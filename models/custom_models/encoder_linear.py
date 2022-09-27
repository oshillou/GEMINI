from torch import nn

def get_model(input_shape=10,output_shape=10,activation='tanh',**kwargs):
	act=nn.Tanh() if activation=='tanh' else nn.ReLU()
	return nn.Sequential(nn.Linear(input_shape,output_shape),act)