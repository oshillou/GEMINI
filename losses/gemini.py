import torch
import ot

def kl_ovo(y_pred,epsilon=1e-12):
    y_pred=torch.clamp(y_pred,min=epsilon,max=1-epsilon)

    Z=y_pred.mean(0)
    Z_log=torch.log(Z)
    y_log=torch.log(y_pred)
    estimates=Z*(Z_log-y_log)+y_pred*(y_log-Z_log)
    R=torch.mean(torch.sum(estimates,1))
    return -R
    
def kl_ova(y_pred,epsilon=1e-12):
    y_pred=torch.clamp(y_pred,min=epsilon,max=1-epsilon)

    Z=torch.mean(y_pred,0,keepdim=True)
    class_entropy=-torch.sum(Z*torch.log(Z))
    cross_entropy=-torch.sum(y_pred*torch.log(y_pred),1)
    cross_entropy=torch.mean(cross_entropy)
    return cross_entropy-class_entropy
    
def hellinger_ovo(y_pred,epsilon=1e-12):
    y_pred=torch.clamp(y_pred,min=epsilon,max=1-epsilon)

    Z=torch.mean(y_pred,0,keepdim=True)
    estimates=torch.sum(torch.sqrt(y_pred*Z),1,keepdim=True)
    estimates=estimates*estimates
    
    return torch.mean(estimates)-1.0
   

def hellinger_ova(y_pred,epsilon=1e-12):
    y_pred=torch.clamp(y_pred,min=epsilon,max=1-epsilon)

    Z=torch.mean(y_pred,0,keepdim=True)
    estimates=torch.sum(Z-torch.sqrt(y_pred*Z),1)
    
    return -torch.mean(estimates)

def tv_ovo(y_pred,epsilon=1e-12):
    y_pred=torch.clamp(y_pred,min=epsilon,max=1-epsilon)

    Z=torch.mean(y_pred,0,keepdim=True)
    K=y_pred.shape[1]
    u_prime=torch.repeat_interleave(Z,y_pred.shape[0],axis=0).unsqueeze(-1)
    v_prime=torch.reshape(y_pred,(-1,1,K))
    A=u_prime@v_prime
    B=torch.abs(A-A.permute(0,2,1))
    return -0.5*torch.sum(torch.mean(B,0))
    
def tv_ova(y_pred,epsilon=1e-12):
    y_pred=torch.clamp(y_pred,min=epsilon,max=1-epsilon)

    Z=torch.mean(y_pred,0,keepdim=True)
    estimates=torch.sum(torch.abs(y_pred-Z),axis=1)
    
    return -0.5*torch.mean(estimates)

def mmd_ovo(y_pred,K,epsilon=1e-12):
    # Clamp the predictions
    y_pred=torch.clamp(y_pred,epsilon,1-epsilon)
    
    Z=y_pred.mean(0)
    y_pred=y_pred.T
    
    A=(torch.unsqueeze(y_pred,-1)@torch.unsqueeze(y_pred,1)).permute([2,1,0])
    A=torch.unsqueeze(A,axis=-1)@(Z**2).view((1,1,-1))
    
    B=y_pred*torch.unsqueeze(Z,-1)
    B=torch.tensordot(B.view(B.shape+(1,1)),B.view((1,1)+B.shape),dims=[[2,3],[0,1]])
    B=B.permute([1,3,0,2])
    
    K=K.view(K.shape+(1,1))
    
    estimates=K*(A+A.permute([0,1,3,2])-2*B)
    
    estimates=torch.clamp(estimates.mean((0,1)),min=epsilon)
    
    return -estimates.sqrt().sum()

def mmd_ova(y_pred,K,epsilon=1e-12):
    # Clamp the predictions
    y_pred=torch.clamp(y_pred,epsilon,1-epsilon)
    Z=y_pred.mean(0,keepdim=True).T.unsqueeze(-1)
    K_prime=K.unsqueeze(0)
    V_prime=y_pred.T.unsqueeze(-1)
    V_prime_t=V_prime.permute(0,2,1)
    
    M=V_prime@V_prime_t+Z*Z-Z*(V_prime+V_prime_t)
    mmd=M*K_prime
    mmd=torch.sqrt(torch.clamp(torch.mean(mmd,dim=(1,2)),min=epsilon))
    
    return -mmd.sum()

def wasserstein_ovo(y_pred,M,epsilon=1e-12):
    # Clamp the predictions
    y_pred=torch.clamp(y_pred,epsilon,1-epsilon)
    # Compute cluster proportions
    py=y_pred.mean(0)
    
    # These are the importance factors of each sample per cluster
    wy=(y_pred/y_pred.sum(0,keepdim=True)).T.contiguous()
    
    loss=0.0
    
    # To optimise computation performances
    # we simply sum over the upper triangle part of the cluster-wise
    # losses, then multiply by 2 thanks to symetry.
    for y1 in range(y_pred.shape[1]):
        for y2 in range(y1+1,y_pred.shape[1]):
            tmp_loss=2*py[y1]*py[y2]*ot.emd2(wy[y1],wy[y2],M)
            loss=loss+tmp_loss
    
    return -loss

def optimised_wovo(y_pred,M,epsilon=1e-12,N=5):
    # K is the number fo clusters, N the number of studied pairs of clusters
    K=y_pred.shape[1]
    N=min(N,K*(K-1)//2)
    pairs=torch.triu_indices(K,K,1)

    # Sample N pairs of random clusters
    ys=torch.randperm(pairs.shape[1])[:N]
    ys=pairs[:,ys]
    y_pred=y_pred[:,ys] # Shaped n_samples x 2 x N

    # compute importance factors
    total_loss=0.0

    # To rescale the sampled wasserstein ovo, we need a factor
    factor=K*(K-1)/(2*N)
    for i in range(y_pred.shape[2]):
        total_loss+=wasserstein_ovo(y_pred[:,:,i],M,epsilon)
    return factor*total_loss

def wasserstein_ova(y_pred,M,epsilon=1e-12):
    # Clamp the predictions
    y_pred=torch.clamp(y_pred,epsilon,1-epsilon)
    # Compute cluster proportions
    py=y_pred.mean(0)
    
    # These are the importance factors of each sample per cluster
    wy=(y_pred/y_pred.sum(0,keepdim=True)).T.contiguous()
    
    loss=0.0
    
    constant_weights=y_pred.new(y_pred.shape[0]).fill_(1.0)/y_pred.shape[0]
    
    for y1 in range(y_pred.shape[1]):
        loss-=py[y1]*ot.emd2(wy[y1],constant_weights,M)
        
    return loss

class FDivWrapper:
    def __init__(self,fdiv):
        self.fdiv=fdiv
    def __call__(self,*args):
        return self.fdiv(args[0])