import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import ot
import itertools

class Similarity:

    def __init__(self,batch_size=100,tqdm_show=20,**kwargs):
        self.batch_size=batch_size
        self.tqdm_show=tqdm_show

    def distance(self,x,y):
        raise NotImplementedError

    def __call__(self,dataset):
        similarities=torch.zeros(len(dataset),len(dataset))
        B=self.batch_size if self.batch_size>1 else len(dataset)
        dataloader=DataLoader(dataset,shuffle=False,batch_size=B)

        if B>=len(dataset):
            # Proceed all combinations at once
            X=next(iter(dataloader))[0]
            return self.distance(X,X)

        total_combinations=len(dataloader)*(len(dataloader)-1)//2
        if total_combinations>self.tqdm_show:
            iterator=tqdm(itertools.combinations_with_replacement(enumerate(dataloader),2),total=total_combinations)
        else:
            iterator=itertools.combinations_with_replacement(enumerate(dataloader),2)

        for (i,batch1),(j,batch2) in iterator:
                i_min,i_max=B*i,min(B*(i+1),len(dataset))
                j_min,j_max=B*j,min(B*(j+1),len(dataset))
                    
                # delta is an existing function, so compute similarity
                batch_similarity=self.distance(batch1[0],batch2[0])
                # Complete symmetry of this similarities
                similarities[i_min:i_max,j_min:j_max]=batch_similarity
                similarities[j_min:j_max,i_min:i_max]=batch_similarity.T
        return similarities

class EuclideanDistance(Similarity):
    def distance(self, x,y):
        x=torch.flatten(x,start_dim=1)
        y=torch.flatten(y,start_dim=1)
        return torch.sqrt(ot.dist(x,y))

class SQEuclideanDistance(Similarity):
    def distance(self, x,y):
        x=torch.flatten(x,start_dim=1)
        y=torch.flatten(y,start_dim=1)
        return ot.dist(x,y)

class LinearKernel(Similarity):
    def distance(self, x,y):
        x_flat=torch.flatten(x,start_dim=1)
        y_flat=torch.flatten(y,start_dim=1)
        return x_flat@y_flat.T

class GaussianKernel(Similarity):
    def __init__(self,mu=0.1,**kwargs):
        super().__init__()
        self.mu=mu
        self.sqeuclidean=SQEuclideanDistance()

    def distance(self, x,y):
        # Compute squared euclidean distance
        return torch.exp(-self.sqeuclidean.distance(x,y)/self.mu)