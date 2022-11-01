from data.similarities import Similarity
from utils.utils import str2similarity
import torch
from scipy import sparse
import utils.floyd_warshall as fw

class FWSimilarityQuantile(Similarity):
    def __init__(self,sub_similarity='euclidean',threshold=0.1,minimise=True,**kwargs):
        super().__init__(batch_size=-1,**kwargs)
        self.sub_similarity=str2similarity(sub_similarity)
        self.threshold=threshold
        self.minimise=minimise
    
    def distance(self,X,Y):
        D=self.sub_similarity.distance(X,Y)
        quantile=torch.quantile(D,torch.Tensor([self.threshold]))
        if self.minimise:
            weights=(D<quantile).float()
        else:
            weights=(D>=quantile).float()

        graph=sparse.csr_matrix(weights)
        W=torch.Tensor(fw.optimised_floyd_warshall(graph))

        W[W.isinf()]=W.shape[0]*2

        return W