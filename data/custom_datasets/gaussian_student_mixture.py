import numpy as np
import torch


def get_dataset(N=500,sigma=1,df=3,alpha=2,**kwargs):
    print(f'Args are N={N},sigma={sigma}, df={df}, alpha={alpha}, {kwargs}')
    
    np.random.seed(0)
    py=np.ones(4)/4
    means=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])*alpha
    
    y=np.random.multinomial(1,py,size=N).argmax(1)
    proportions=np.bincount(y)
    y.sort()
    
    covariance=sigma*np.eye(2)
    
    X=[]
    for k in range(3):
        X+=[np.random.multivariate_normal(means[k],covariance,size=proportions[k])]
    
    # Sample from the student t distribution
    nx=np.random.multivariate_normal(np.zeros(2),covariance,size=proportions[-1])
    u=np.random.chisquare(df,proportions[-1]).reshape((-1,1))
    x=np.sqrt(df/u)*nx+np.expand_dims(means[-1],axis=0)
    X+=[x]
    
    X=np.concatenate(X,axis=0)

    return torch.utils.data.TensorDataset(torch.Tensor(X),torch.Tensor(y).long())


def get_train_dataset(**kwargs):
    return get_dataset(**kwargs)

def get_val_dataset(**kwargs):
    return get_dataset(**kwargs)

if __name__=='__main__':
    dataset=get_dataset(N=500,sigma=1,df=0.1,alpha=2)
    import matplotlib.pyplot as plt
    plt.scatter(x=dataset.tensors[0][:,0],y=dataset.tensors[0][:,1],c=dataset.tensors[1])
    plt.show()