import numpy as np
from numpy.linalg import cholesky, slogdet
from scipy.special import gammaln, zeta, gamma
from scipy.special import kv as Knu  # modified Bessel K
from scipy.integrate import quad
import sys 

sys.path.insert(1, '../')

from utils.elliptical_distributions import MultivariateGaussian

def importance_sampling_GVI(potential: callable, m_hat: np.ndarray, cov_hat: np.ndarray, f: callable, n_samples: int = 2000):
    """
    Perform self-normalised importance sampling using a Gaussian variational proposal.
    
    Samples are drawn from q(y) = N(m_hat, cov_hat) and weighted to approximate
    expectations under the target distribution p(y).
    
    Args:
        potential (callable): Log-density function log p(y) evaluated at y of shape (dim,) or (dim, n)
        m_hat (array-like): Mean vector of shape (dim,)
        cov_hat (array-like): Covariance matrix of shape (dim, dim)
        f (callable): Function evaluated on samples of shape (dim, n), returning (n,) or (n, k)
        n_samples (int, optional): Number of importance samples
        
    Returns:
        float: Importance sampling estimate
        float: Effective sample size (ESS)
    """
    gaussian_distribution = MultivariateGaussian(m_hat[:, None], cov_hat)
    y    = gaussian_distribution.sample(n_samples)
    logq = -gaussian_distribution.potential(y)
    logp = potential(y)

    logw = logp - logq
    m = np.max(logw)
    w = np.exp(logw - m)
    w_norm = w / np.sum(w)

    fx = f(y)
    estimate = np.sum(w_norm * fx)

    ess = 1.0 / np.sum(w_norm**2)

    return float(estimate), float(ess)


##### Importance sampling using radVI
def importance_sampling_radvi(potential, radvi_obj, f, dim, n_samples=2000):
    """
    Perform self-normalised importance sampling using a radial transport proposal.
    
    Samples are generated via:
        z ~ N(0, I)
        y = T(z)
    
    The proposal density is computed using:
        log q(y) = log ρ(z) - log |det J_T(z)|
    
    Args:
        potential (callable): Log-density function log p(y) evaluated at y of shape (dim,) or (dim, n)
        radvi_obj: Object implementing push_forward(z) and _log_det_jacobian(z)
        f (callable): Function evaluated on samples of shape (dim, n), returning (n,) or (n, k)
        dim (int): Dimensionality
        n_samples (int, optional): Number of importance samples
        
    Returns:
        float: Importance sampling estimate
        float: Effective sample size (ESS)
    """
    standard_gaussian_distribution = MultivariateGaussian(np.zeros(dim)[:, None], np.eye(dim))
    zs = standard_gaussian_distribution.sample(n_samples)
    ys = radvi_obj.push_forward(zs)
    
    logqs = -standard_gaussian_distribution.potential(zs) - radvi_obj._log_det_jacobian(zs)
    logps = potential(ys)

    logw = logps - logqs
    m = np.max(logw)
    w = np.exp(logw - m)
    w_norm = w / np.sum(w)

    fx = f(ys)
    estimate = np.sum(w_norm * fx)

    ess = 1.0 / np.sum(w_norm**2)

    return float(estimate), float(ess)