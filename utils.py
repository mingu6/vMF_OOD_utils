import numpy as np
import torch
from scipy.special import logsumexp

from spherecluster import VonMisesFisherMixture
from spherecluster.von_mises_fisher_mixture import _vmf_log, _vmf_log_asymptotic


def fit_vmf_mixture(X: torch.Tensor, n_clusters: int, **spherecluster_params):
    """
    Fits a vMF mixture distribution to a set of embedding vectors. Vectors are
    l2-normalized inside of this function so no need to preprocess. Returns
    the estimator (similar to sklearn k-Means style) containing fitted
    parameters which will be used for likelihood evaluation.

    Args:
        X (Tensor): n_obs x n_features containing embedding vectors
        n_clusters (int): number of mixture components in model
        spherecluster_params: additional parameters used to initialize the VonMisesFisherMixture
                              class from the spherecluster package. Important params are
                              posterior_type ('soft'/'hard') and init ('random-class'/'spherical-k-means' etc.)
    Returns:
        sklearn estimator containing fitted vMF distribution
    """
    X_np = X.detach().cpu().numpy()
    vmf = VonMisesFisherMixture(n_clusters=n_clusters, **spherecluster_params)
    vmf.fit(X_np)
    return vmf




def _log_likelihood(X, centers, weights, concentrations):
    """
    taken directly from the spherecluster library (von_mises_fisher_mixture.py)
    but actually computes the log-likelihood, not the posterior allocation probabilities?!?!?!
    """
    if len(np.shape(X)) != 2:
        X = X.reshape((1, len(X)))

    n_examples, n_features = np.shape(X)
    n_clusters, _ = centers.shape

    if n_features <= 50:  # works up to about 50 before numrically unstable
        vmf_f = _vmf_log
    else:
        vmf_f = _vmf_log_asymptotic

    f_log = np.zeros((n_clusters, n_examples))
    for cc in range(n_clusters):
        f_log[cc, :] = vmf_f(X, concentrations[cc], centers[cc, :])

    posterior = np.zeros((n_clusters, n_examples))
    weights_log = np.log(weights)
    posterior = np.tile(weights_log.T, (n_examples, 1)).T + f_log
    for ee in range(n_examples):
        posterior[:, ee] = logsumexp(posterior[:, ee])

    return posterior.sum(axis=0)



def log_likelihood(X: torch.Tensor, vmf: VonMisesFisherMixture):
    """
    Evaluates the log-likelihood of a set of embedding vectors given a vMF mixture
    distribution passed as an sklearn estimator.
    """
    X_np = X.detach().cpu().numpy()
    llhood = _log_likelihood(X_np, vmf.cluster_centers_, vmf.weights_, vmf.concentrations_)
    return torch.from_numpy(llhood)

# test

X_train = torch.randn((500, 128))  # (n_obs x n_features)
print('fitting using soft assignments...')
vmf_soft = fit_vmf_mixture(X_train, n_clusters=5, posterior_type='soft', init='spherical-k-means', n_jobs=4)
print('fitting using hard assignments...')
vmf_hard = fit_vmf_mixture(X_train, n_clusters=5, posterior_type='hard', init='spherical-k-means', n_jobs=4)

X_test = torch.randn((100, 128))  # (n_obs x n_features)
print('evaluating log-likelihood using soft trained model')
llhood_test_soft = log_likelihood(X_test, vmf_soft)
print('evaluating log-likelihood using hard trained model')
llhood_test_hard = log_likelihood(X_test, vmf_hard)
import pdb; pdb.set_trace()

assert llhood_test_soft.shape == (100,)
assert llhood_test_hard.shape == (100,)
assert not torch.allclose(llhood_test_hard, llhood_test_soft)