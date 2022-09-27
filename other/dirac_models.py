__author__ = 'Louis Ohl'
__email__ = 'louis.ohl@inria.fr'
__doc__ = 'Generates a set of 3 isotropic gaussian that we try to cluster using a dirac model. The model is trained ' \
          'for a certain number of epochs using either the mmd ova or the mutual information (kl ova). To generate ' \
          'the final clustering figure, use the option --export. '

from sklearn import datasets
import torch
from torch import nn
import sys

sys.path.append('..')
import losses.gemini as gemini
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=100, help='The number of samples to generate')
    parser.add_argument('--seed', type=int, default=0, help='The random seed value. Ignored if it is equal to -1')
    parser.add_argument('--gemini', type=str, choices=['kl', 'mmd'], default='mmd')
    parser.add_argument('--export', default=False, action='store_true',
                        help='if non-empty string, will export the clustering output in a figure')
    parser.add_argument('--n_epochs', type=int, default=1000)
    args = parser.parse_args()
    return args


def scatterplot(X, y, s=1):
    markers = ["v", "o", "x"]
    colors = ['red', 'black', 'blue']
    for i, c in enumerate(np.unique(y)):
        plt.scatter(X[:, 0][y == c], X[:, 1][y == c], c=colors[i], marker=markers[i], s=s)
    plt.axis('off')


def train_params(X, N, K, loss_fct, kernel_values, n_epochs=1000):
    p_theta = nn.Parameter(torch.randn(N, K))
    optimiser = torch.optim.Adam([p_theta], lr=1e-2)
    loss_history = []
    for epoch in tqdm(range(n_epochs)):
        y_pred = torch.softmax(p_theta, dim=-1)

        loss = loss_fct(y_pred, kernel_values)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_history += [loss.item()]
    return p_theta, loss_history


def get_loss_function(loss_name):
    if loss_name == 'kl':
        return gemini.FDivWrapper(gemini.kl_ova)
    else:
        return gemini.mmd_ova


def main():
    args = get_args()

    # Fix the seed
    if args.seed != -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    N, K = max(3, args.N), 3

    # Generate K isotropic Gaussians
    X, y = datasets.make_blobs(n_samples=N, n_features=2, centers=K, cluster_std=0.5, random_state=0)
    X = torch.Tensor(X)
    y = torch.Tensor(y)

    # Create the parameters
    p_theta, history = train_params(X, N, K, get_loss_function(args.gemini), X @ X.T, n_epochs=args.n_epochs)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    scatterplot(X, p_theta.argmax(1))
    plt.title('Clustering output')
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.title(f'{args.gemini} loss')
    plt.suptitle('Final clustering')
    plt.show()

    if args.export:
        plt.figure(figsize=(15, 10))
        scatterplot(X, p_theta.argmax(1), s=1000)
        plt.tight_layout()
        plt.savefig(f'{args.gemini}.svg')
        plt.close()
        print(f'Final GEMINI: {history[-1]}')


if __name__ == '__main__':
    main()
