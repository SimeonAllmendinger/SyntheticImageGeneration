import sys
import os
sys.path.append(os.path.abspath(os.curdir))

import torch
import clip
from PIL import Image
import argparse
import os
import torch
from torchvision import datasets, transforms
import numpy as np
import sys
from scipy import linalg
from tqdm import tqdm


def calc_fcd_score(batch_feature_source_np, batch_feature_test_np):
    mu1 = np.mean(batch_feature_source_np, axis=0)
    sigma1 = np.cov(batch_feature_source_np, rowvar=False)
    mu2 = np.mean(batch_feature_test_np, axis=0)
    sigma2 = np.cov(batch_feature_test_np, rowvar=False)
    curr_fcd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return curr_fcd


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    # Numpy implementation of the Frechet Distance.
    # The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    # and X_2 ~ N(mu_2, C_2) is
    # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    # Stable version by Dougal J. Sutherland.
    
    
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

    
def __get_fcd_score__(feature_np_source, feature_np_test):
    
    
    
    res = calc_fcd_score(feature_np_source, feature_np_test)
    return res
