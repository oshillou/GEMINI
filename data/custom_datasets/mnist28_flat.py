import torch
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

def get_train_dataset(data_path='./',**kwargs):
    mnist_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5]),transforms.Lambda(lambda x:torch.flatten(x))])
    train_ds=MNIST(data_path,train=True,transform=mnist_transform)

    return train_ds

def get_val_dataset(data_path='./',**kwargs):
    mnist_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5]),transforms.Lambda(lambda x:torch.flatten(x))])
    val_ds=MNIST(data_path,train=False,transform=mnist_transform)

    return val_ds