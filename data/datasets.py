from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import torch
import logging


class SimilarityDataset(Dataset):
    """This class is a wrap-up of any torch Dataset. It extends the original dataset D
    by yielding batches containing samples of the original datasets with their respective similarity.
    In other wors: D={(x0,...,xN)} becomes D'={(x0,...,xN,S(x0),S(...),S(xN))} where each xi is a dataset
    modality. For example, in most common settings, x0 is the data and x1 is the labels.

    Note that the idea of similarity in this context covers both distances and kernels.

    ----
    Args:
    + formal_dataset: the torch Dataset from which simmilarities will be computed and returned batch-wise
    + similarities: a list containing N similarity functions. 1 per modality of the formal_dataset.
        + A None item implies no similarity is computed for the modality of the same index.
        + Similarities extra to the number of modalities are ignored
        + Less similarities than modalities implies that the last modalities do not have any similarity
    
    ---
    Example
    >>> x,y=torch.randn(99,5), torch.rand(99)>0.5 #Build a random dataset
    >>> formal_dataset=TensorDataset(x,y)
    >>> linear_kernel=lambda x:x@x.T # Our Similarity function is a linear kernel for the dataset's x
    >>> ds=SimilarityDataset(ds,[linear_kernel,None]) # No similarity to be computed between labels
    >>> # To make everything work out for now, provide a batch sampler as main sampler of a dataloader
    >>> dataloader=DataLoader(ds,sampler=BatchSampler(SequentialSampler(ds,batch_size=20,drop_last=False)),collate_fn=lambda x: x[0])
    >>> print(next(iter(dataloader)))
    """
    def __init__(self,formal_dataset,similarity=None,similarity_batch=100,**kwargs):
        super().__init__()
        
        assert issubclass(formal_dataset.__class__,Dataset), 'Please provide a dataset inheriting from torch.utils.data.Dataset'
        
        # How many distances/kernel do we have to store?
        self.formal_dataset=formal_dataset
        if similarity is not None:
            self.pre_saved_similarities=similarity(formal_dataset)
        else:
            self.pre_saved_similarities=torch.zeros(len(formal_dataset),len(formal_dataset))
    
    def __getitem__(self,idx):
        # To return in a batch the similarities between all samples of the same batch
        # We need to know all indices of the batch.
        # This means that idx must preferrably be a list or a slice. Typically: BatchSampler of pytorch
        if type(idx)==int:
            # When returning a simple index, we only return the similarity to self.
            return self.formal_dataset[idx]+(self.pre_saved_similarities[idx,idx],)
        
        batch=[]
        for index in idx:
            batch+=[self.formal_dataset[index]+(self.pre_saved_similarities[:,index][idx],)]
        return default_collate(batch)

    def __len__(self):
        return len(self.formal_dataset)
    
    @staticmethod
    def collate_fn(batch):
        """When using batch sampler, since our method __getitem__ already yields complete batches,
        The default collate_fn function of a DataLoader will add another layer of list on top of the batch
        So we need to get rid of this by applying this default collate fn"""
        return batch[0]