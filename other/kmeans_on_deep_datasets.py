# First import all that is necessary
import pandas as pd
from sklearn import metrics, cluster

from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms

import os
import sys

sys.path.append('..')
import torch
from data.custom_datasets import cifar10

import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar_simclr', 'cifar10'], default='mnist')
    parser.add_argument('--K', type=int, default=10, help='Number of clusters to find for KMeans model')
    args = parser.parse_args()

    return args


def get_mnist(data_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mnist_train = MNIST(data_path, train=True, transform=transform)
    mnist_val = MNIST(data_path, train=False, transform=transform)

    return mnist_train, mnist_val


def get_cifar10(data_path):
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
    train_ds = CIFAR10(data_path, train=True, transform=cifar_transform)
    val_ds = CIFAR10(data_path, train=False, transform=cifar_transform)

    return train_ds, val_ds


def get_cifar_simclr(data_path):
    # Load the SIMCLR features
    simclr_train_features = torch.load('../data/custom_similarities/cifar_simclr_train_features')
    simclr_val_features = torch.load('../data/custom_similarities/cifar_simclr_validation_features')
    val_ds = cifar10.get_val_dataset(data_path)
    train_ds = cifar10.get_train_dataset(data_path)
    return simclr_train_features, train_ds.targets, simclr_val_features, val_ds.targets


def expand_and_flatten_data(dataloader):
    batches = [batch for batch in dataloader]
    images = torch.cat(list(map(lambda x: x[0], batches)), dim=0)
    images = torch.flatten(images, start_dim=1)
    labels = list(map(lambda x: x[1], batches))
    labels = torch.cat(labels)

    return images, labels


def main():
    args = get_args()

    print(f'Getting dataset {args.dataset}')
    if args.dataset != 'cifar_simclr':
        if args.dataset == 'mnist':
            train_ds, val_ds = get_mnist(args.data_path)
        else:
            train_ds, val_ds = get_cifar10(args.data_path)
        num_workers = os.cpu_count()
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=250, shuffle=False, drop_last=True,
                                                   pin_memory=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(train_ds, batch_size=250, shuffle=False, drop_last=True,
                                                 pin_memory=True, num_workers=num_workers)

        train_images, train_labels = expand_and_flatten_data(train_loader)
        val_images, val_labels = expand_and_flatten_data(val_loader)
    else:
        train_images, train_labels, val_images, val_labels = get_cifar_simclr(args.data_path)

    print(f'Building model for {args.K} clusters')
    kmeans_model = cluster.KMeans(args.K)

    print('Fitting model')
    kmeans_model.fit(train_images)

    print('Computing results')
    train_pred = kmeans_model.predict(train_images)
    val_pred = kmeans_model.predict(val_images)

    train_ari = metrics.adjusted_rand_score(train_labels, train_pred)
    val_ari = metrics.adjusted_rand_score(val_labels, val_pred)
    print(f'Train ARI is: {train_ari:.3f}')
    print(f'Val ARI is: {val_ari:.3f}')

    print('Exporting results')
    result_df = pd.DataFrame([{'Train': True, 'ARI': train_ari}, {'Train': False, 'ARI': val_ari}])
    result_df['Dataset'] = args.dataset
    result_df['K'] = args.K

    file_path = 'kmeans_results.csv'

    if os.path.exists(file_path):
        print('There is already a file containing KMeans result. I extend it.')
        previous_df = pd.read_csv(file_path, sep=',', index_col=None)
        result_df = pd.concat([previous_df, result_df], ignore_index=True)
    result_df.to_csv(file_path, sep=',', index=False)

    print('Finished')


if __name__ == '__main__':
    main()
