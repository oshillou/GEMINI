from torch import nn

def get_model(input_shape=10,output_shape=10,hidden_dim=10,activation='tanh',**kwargs):
	act=nn.Tanh() if activation=='tanh' else nn.ReLU()
	return nn.Sequential(nn.Linear(input_shape,hidden_dim),act,nn.Linear(hidden_dim,output_shape))