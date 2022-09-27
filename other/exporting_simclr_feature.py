__author__ = 'Louis Ohl'
__email__ = 'louis.ohl@inria.fr'
__doc__ = 'Downloads a SIMCLR model using the library torch lightning in order to extract features for the CIFAR 10' \
          'dataset.' \
          'Usage:' \
          'python exporting_simclr_feature --data+path /path/to/your/data/folder'

from pl_bolts.models.self_supervised import SimCLR
from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True,
                    help='The path to the folder that contains the CIFAR10 dataset')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

simclr.freeze()

simclr_encoder = simclr.encoder.to(device)

# Load the CIFAR10 dataset transforms according to simclr bolt


data_root = args.data_path
training_set = CIFAR10(data_root,
                       transform=transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()]))
validation_set = CIFAR10(data_root, train=False,
                         transform=transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor()]))

train_features = []
train_labels = []
train_loader = DataLoader(training_set, batch_size=250, shuffle=False)
with torch.no_grad():
    for x, y in tqdm(train_loader):
        features = simclr_encoder(x.to(device))[0].cpu()
        train_features += [features]
        train_labels += [y]
train_features = torch.cat(train_features, dim=0)
train_labels = torch.cat(train_labels, dim=0)
train_features_df = TensorDataset(train_features, train_labels)

validation_features = []
validation_labels = []
validation_loader = DataLoader(validation_set, batch_size=250, shuffle=False)
with torch.no_grad():
    for x, y in tqdm(validation_loader):
        features = simclr_encoder(x.to(device))[0].cpu()
        validation_features += [features]
        validation_labels += [y]
validation_features = torch.cat(validation_features, dim=0)
validation_labels = torch.cat(validation_labels, dim=0)
val_features_ds = TensorDataset(validation_features, validation_labels)

torch.save(validation_features, '../data/custom_similarities/cifar_simclr_validation_features')
torch.save(train_features, '../data/custom_similarities/cifar_simclr_train_features')
