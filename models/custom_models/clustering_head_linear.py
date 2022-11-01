from torch import nn

def get_model(input_shape=2,num_clusters=10,**kwargs):
	return nn.Sequential(nn.Linear(input_shape,num_clusters),nn.Softmax(dim=-1))
