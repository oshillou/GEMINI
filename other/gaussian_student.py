__author__ = 'Louis Ohl'
__email__ = 'louis.ohl@inria.fr'
__doc__='Trains an oracle, a gmm or a kmeans model on the 3 gaussian distributions + 1 student-t dataset'

import os.path
import sys
sys.path.append('../')
from data.custom_datasets import gaussian_student_mixture as dataset
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd
from utils.metrics import AvgMetric
import numpy as np
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=float, default=5.0,
                        help='Parameter controlling the space between expectations of the distributions')
    parser.add_argument('--df', type=float, default=1.0,
                        help='The degrees of freedom of the Student-t distribution in the mixture')
    parser.add_argument('--N', type=int, default=5000, help='The number of samples to generate in the dataset')

    parser.add_argument('--model', type=str, choices=['oracle', 'kmeans', 'gmm'], default='oracle')
    parser.add_argument('--K', type=int, default=4,
                        help='Number of clusters to find for the model. If oracle model, then it will be 4.')
    parser.add_argument('--cov_type', type=str, choices=['full', 'diag'], default='full',
                        help='The covariance type for the GMM. Please refer to scikit learn\'s GaussianMixture')
    parser.add_argument('--runs', type=int, default=50,
                        help='The number of runs for the model. Useless in case of oracle model.')

    parser.add_argument('--csv', default='results.csv', type=str, help='The CSV in which all results will get exported')

    args = parser.parse_args()

    if args.model == 'oracle':
        args.runs = 1
        args.K = 4

    return args


def train_kmeans(args, X, y):
    metric = AvgMetric(args.K)
    kmeans = KMeans(args.K)
    k_pred = kmeans.fit_predict(X)
    metric(torch.Tensor(y).long(), torch.Tensor(k_pred).long())

    return {'ARI': metric.ari().item(), 'ACC': metric.accuracy().item(), 'PTY': metric.purity().item(),
            'NCE': metric.normalised_conditional_entropy().item(), 'UCL': metric.ucl().item()}


def train_gmm(args, X, y):
    metric = AvgMetric(args.K)
    gmm = GaussianMixture(args.K, covariance_type=args.cov_type, reg_covar=1e-4)
    gmm_pred = gmm.fit_predict(X)
    metric(torch.Tensor(y).long(), torch.Tensor(gmm_pred).long())

    return {'ARI': metric.ari().item(), 'ACC': metric.accuracy().item(), 'PTY': metric.purity().item(),
            'NCE': metric.normalised_conditional_entropy().item(), 'UCL': metric.ucl().item(),
            'covariance': args.cov_type}


def get_oracle_results(args, X, y):
    # The proportions are defined by:
    py = np.ones(args.K) / args.K
    # The means are defined by:
    means = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * args.alpha

    likelihoods = np.zeros((args.N, args.K))
    for i in range(4):
        if i == 3:
            likelihoods[:, i] = stats.multivariate_t.pdf(X, loc=means[i], df=args.df)
        else:
            likelihoods[:, i] = stats.multivariate_normal.pdf(X, mean=means[i])

    # We can now compute the evidence p(x) for both datasets as well as the joint p(x,y)
    X_joint = likelihoods * py
    X_px = X_joint.sum(1, keepdims=True)

    # MAP
    X_map = X_joint / X_px

    metric = AvgMetric(args.K)
    metric(torch.Tensor(y).long(), torch.Tensor(X_map.argmax(1)).long())

    return {'ARI': metric.ari().item(), 'ACC': metric.accuracy().item(), 'PTY': metric.purity().item(),
            'NCE': metric.normalised_conditional_entropy().item(), 'UCL': metric.ucl().item()}


def main():
    args = get_args()

    tmp_dataset = dataset.get_dataset(N=args.N, df=args.df, alpha=args.alpha)
    X=tmp_dataset.tensors[0].numpy()
    y=tmp_dataset.tensors[1].numpy()

    if args.model == 'kmeans':
        train_fct = train_kmeans
    elif args.model == 'gmm':
        train_fct = train_gmm
    else:
        train_fct = get_oracle_results

    results = []
    for i in tqdm(range(args.runs)):
        tmp = train_fct(args, X, y)
        tmp['df'] = args.df
        tmp['alpha'] = args.alpha
        tmp['K'] = args.K
        tmp['model'] = args.model
        results += [tmp]

    csv = pd.DataFrame(results)
    if os.path.exists(args.csv):
        previous_csv = pd.read_csv(args.csv, sep=',', index_col=None)
        csv = pd.concat([previous_csv, csv], ignore_index=True)

    csv.to_csv(args.csv, sep=',', index=False)


if __name__ == '__main__':
    main()
