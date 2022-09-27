from torch import nn

class DiscriminativeModel(nn.Module):
    def __init__(self,encoder,clustering_head):
        super().__init__()

        self.encoder=encoder
        self.clustering_head=clustering_head
    
    def forward(self,x):
        features=self.encoder(x)
        p_yx=self.clustering_head(features)
        return p_yx,None
    
    def get_features(self,batch):
        return self.encoder(batch)
    
    def predict_clustering(self,features):
        return self.clustering_head(features)