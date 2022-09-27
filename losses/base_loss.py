import torch
    
def entropy_loss(y_pred,epsilon=1e-8):
    p_y=torch.clamp(y_pred.mean(0),epsilon,1-epsilon)

    return torch.sum(p_y*torch.log(p_y))