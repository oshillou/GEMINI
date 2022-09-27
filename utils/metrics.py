import torch
from scipy.optimize import linear_sum_assignment as lsa

class AvgMetric:
    def __init__(self,num_clusters):
        self.num_clusters=num_clusters
        self.cmatrix=torch.zeros(1,num_clusters,dtype=torch.int32)
        self.prev_assignments=torch.zeros(1,num_clusters,dtype=torch.int32)
        
    def extend_cmatrix(self,num_classes):
        new_cmatrix=torch.zeros(num_classes,self.num_clusters)
        new_cmatrix[:self.cmatrix.shape[0],:]=self.cmatrix
        self.cmatrix=new_cmatrix
    
    def __call__(self,y_true,y_pred):
        
        num_classes=torch.max(y_true).item()+1
        if num_classes>self.cmatrix.shape[0]:
            self.extend_cmatrix(num_classes)
        for i, j in zip(y_true,y_pred):
            self.cmatrix[i,j]+=1
        self.cmatrix=self.cmatrix.int()
    
    def reset(self):
        self.prev_assignments=self.cmatrix.sum(0)
        self.cmatrix.fill_(0)
    
    def ari(self):
        # Compute the adjusted rand index
        sum_classes=self.cmatrix.sum(1)
        sum_classes_permutations=((sum_classes*(sum_classes-1))/2).sum()
        sum_clusters=self.cmatrix.sum(0)
        sum_clusters_permutations=((sum_clusters*(sum_clusters-1))/2).sum()
        
        cmatrix_permutations=(self.cmatrix*(self.cmatrix-1)/2).sum()
        n=self.cmatrix.sum()
        total_permutations=n*(n-1)/2

        cross_permutations=sum_classes_permutations*sum_clusters_permutations/total_permutations
        numerator=cmatrix_permutations-cross_permutations
        denominator=(sum_clusters_permutations+sum_classes_permutations)/2-cross_permutations

        return numerator/denominator

    def accuracy(self):
        # Perform hungarian algorithm to maximise assignments class <-> cluster
        r,c=lsa(self.cmatrix,True)
        return self.cmatrix[r,c].sum()/self.cmatrix.sum()

    def ucl(self):
        # Compute how many clusters are actually used by looking at non-empty columns of cmatrix
        return (self.cmatrix.max(0)[0]!=0).int().sum()

    def purity(self):
        # Look how cluster distributions do not overlap through classes
        return (self.cmatrix.max(0)[0].sum())/self.cmatrix.sum()
    
    def stability(self):
        # This metric shows how many samples have changed cluster assignments since the last reset
        return (torch.abs(self.cmatrix.sum(0)-self.prev_assignments).sum()/2).int()
    
    def normalised_conditional_entropy(self):
        normalised_cmatrix=self.cmatrix/self.cmatrix.sum()
        cluster_proportions=normalised_cmatrix.sum(0,keepdim=True)

        cmatrix_entropy=-torch.sum(normalised_cmatrix*torch.log(normalised_cmatrix+1e-12))

        conditional_entropy=torch.sum(normalised_cmatrix*(torch.log(normalised_cmatrix+1e-12)-torch.log(cluster_proportions+1e-12)))

        return conditional_entropy/cmatrix_entropy
    
    def __str__(self):
        return str(self.cmatrix)