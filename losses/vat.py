# ALL credits for this file are to lyakaap on github
# https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py

# Some light touches were made in accordance to the code of GEMINI

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, pred):

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat,_ = model(x + self.xi * d)
                pred_hat=torch.clamp(pred_hat,min=1e-12)
                logp_hat = torch.log(pred_hat)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat,_ = model(x + r_adv)
            pred_hat = torch.clamp(pred_hat,min=1e-12)
            logp_hat = torch.log(pred_hat)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')
            
        if torch.isnan(lds):
            print(logp_hat,lds,pred)
            import sys
            sys.exit(-1)
        return lds
