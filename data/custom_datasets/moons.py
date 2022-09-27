import numpy as np
import torch

def get_dataset(N=500,tau=0.5,rho=2,sigma=0.05,**kwargs):
    np.random.seed(0)

    angles=np.random.uniform(low=0.0,high=np.pi,size=N)
    ys=np.random.binomial(1,tau,size=N)
    # Angles for class y=0 will be in [-pi,0] and angles of class y=1 in [0,pi]
    angles=angles*(2*ys-1)
    cos_angle=np.cos(angles)
    sin_angle=np.sin(angles)
    angles=np.concatenate([np.expand_dims(cos_angle,axis=-1),np.expand_dims(sin_angle,axis=-1)],axis=-1)
    X=rho*angles

    X+=np.concatenate([np.expand_dims(rho/2*(2*ys-1),axis=-1),-np.expand_dims(rho/4*(2*ys-1),axis=-1)],axis=-1)

    # Add Gaussian noise
    X+=np.random.normal(loc=0,scale=sigma,size=(N,2))

    return torch.utils.data.TensorDataset(torch.Tensor(X),torch.Tensor(ys).long())

def get_train_dataset(**kwargs):
    return get_dataset(**kwargs)

def get_val_dataset(**kwargs):
    return get_dataset(**kwargs)

if __name__=='__main__':
    dataset=get_dataset(N=500,tau=0.5,rho=2,sigma=0.05)

    import matplotlib.pyplot as plt
    plt.scatter(x=dataset.tensors[0][:,0],y=dataset.tensors[0][:,1],c=dataset.tensors[1])
    plt.show()