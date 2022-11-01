from data import similarities
from torch.utils.data import TensorDataset
import torch

class SIMCLREuclidean(similarities.Similarity):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sub_distance=similarities.EuclideanDistance()
    
    def __call__(self, dataset):
        current_path=__file__[:-len('SIMCLREuclidean.py')]
        if len(dataset)==10000:
            return super().__call__(TensorDataset(torch.load(current_path+'cifar_simclr_validation_features')))
        else:
            return super().__call__(TensorDataset(torch.load(current_path+'cifar_simclr_train_features')))
    
    def distance(self,X,Y):
        return self.sub_distance.distance(X,Y)