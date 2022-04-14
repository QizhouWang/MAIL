from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from wideresnet import *
from resnet import *
from at_loss import at_loss
from mail_loss import *
import numpy as np
import time
import sys

from autoattack import AutoAttack
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=3.5e-3,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=5.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--method', default='trades_mail',
                    help='mart_mail')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--bias', type=float, default=-1.5,
                    help='weighting bias term')
parser.add_argument('--slope', type=float, default=1.0,
                    help='weighting bias term')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data_attack/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.CIFAR10(root='../data_attack/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
                    
        # calculate robust loss
        if 'mail' not in args.method or epoch <= 75:
            loss = at_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            beta=args.beta,
                            method = args.method)
        else:
            loss = mail_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            bias = args.bias, slope = args.slope,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            beta=args.beta,
                            method = args.method)
        loss.backward()
        optimizer.step()

        # print progress
        sys.stdout.write('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    #

def adjust_learning_rate(optimizer, epoch, model, method):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def cwloss(output, target,confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.) 
    loss = torch.sum(loss)
    return loss

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003, loss_fn = 'cent'):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            if loss_fn == 'cent': loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            elif loss_fn == 'cw': loss = cwloss(model(X_pgd), y)
            else: raise RuntimeError('invalid loss_fn')

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def eval_adv_test_whitebox(model, device, test_loader):

    model.eval()
    ce100_err_total, ce20_err_total, cw_err_total, na_err_total = 0, 0, 0, 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_na, err_20 = _pgd_whitebox(model, X, y)
        # err_na, err_100 = _pgd_whitebox(model, X, y, num_steps = 100)
        err_100 = 0
        _, err_cw = _pgd_whitebox(model, X, y, loss_fn='cw')
        
        na_err_total += err_na
        cw_err_total += err_cw
        ce20_err_total += err_20
        ce100_err_total += err_100
    
    print('na %.4f | 20 %.4f | 100 %.4f | cw %.4f ' % (1 - na_err_total / len(test_loader.dataset),1- ce20_err_total / len(test_loader.dataset),1- ce100_err_total / len(test_loader.dataset),1- cw_err_total / len(test_loader.dataset)))

def main():
    model = ResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)   
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, model, args.method)

        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch)

        if epoch % 10 == 9:
            print() 
            eval_adv_test_whitebox(model, device, test_loader)
            x_test = torch.cat([x for (x, y) in test_loader], 0)
            y_test = torch.cat([y for (x, y) in test_loader], 0)
            adversary = AutoAttack(model, norm='Linf', eps=0.031, version='standard')
            adv_complete = adversary.run_standard_evaluation(x_test[:1000], y_test[:1000], bs=500)
            print()
    

if __name__ == '__main__':
    main()
