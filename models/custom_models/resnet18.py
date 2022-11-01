from torchvision.models import resnet18
from torch import nn

def get_model(**kwargs):
    return resnet18()
