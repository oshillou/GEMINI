from torchvision import transforms
from torchvision.datasets import CIFAR10

def get_train_dataset(data_path='./',**kwargs):
    cifar_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
    train_ds=CIFAR10(data_path,train=True,transform=cifar_transform)

    return train_ds

def get_val_dataset(data_path='./',**kwargs):
    mnist_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
    val_ds=CIFAR10(data_path,train=False,transform=mnist_transform)

    return val_ds