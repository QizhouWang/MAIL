import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def PM(logit, target):
    eye = torch.eye(10).cuda()
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest = True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
    return  probs_2nd - probs_GT

def weight_assign(logit, target, bias, slope):
    pm = PM(logit, target)
    reweight = ((pm + bias) * slope).sigmoid().detach()
    normalized_reweight = reweight * 3
    return normalized_reweight

def mail_loss(model,
              x_natural,
              y,
              optimizer,
              bias, slope, 
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf',
              method = 'mart'):
    kl = nn.KLDivLoss(reduction='none')
    criterion_kl = nn.KLDivLoss(size_average=False)

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if 'trades' not in method:
                    loss_ce = F.cross_entropy(model(x_adv), y)
                else:
                    loss_ce = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(model(x_natural), dim=1)) 
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    if 'mart' in method:
        logits = model(x_natural)
        logits_adv = model(x_adv)
        norm_weight = weight_assign(logits_adv, y, bias, slope)

        adv_probs = F.softmax(logits_adv, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, y, reduction = 'none') + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y, reduction = 'none')
        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = (loss_adv * norm_weight).mean() + float(beta) * loss_robust
    elif 'trades' in method:
        logits = model(x_natural)
        logits_adv = model(x_adv)
        norm_weight = weight_assign(logits_adv, y, bias, slope)

        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1)), dim=1) * norm_weight)
        loss = loss_natural + float(beta) * loss_robust
    elif 'at' in method:
        logits_adv = model(x_adv)
        norm_weight = weight_assign(logits_adv, y, bias, slope)

        adv_probs = F.softmax(logits_adv, dim=1)
        loss = F.cross_entropy(logits_adv, y, reduction = 'none')
        loss = (loss * norm_weight).mean()
        
    else: raise RuntimeError('invalid method name')
    return loss
