import numpy as np
from numpy.linalg import cholesky, slogdet
import autograd.numpy as anp
from autograd.scipy.special import gammaln # Logarithm of the absolute value of the gamma function.
from scipy.special import gammaln, zeta, gamma
from scipy.special import kv as Knu  # Modified Bessel K
from scipy.integrate import quad

# ---------- utilities ----------
def _as_column(x: np.ndarray, type: str = 'numpy'):
    """
    Convert input to column vector format.
    
    Args:
        x (np.ndarray): Input array of any shape
        type (str): Type of array to convert to ('numpy' or 'autograd')
    Returns:
        np.ndarray: Array reshaped to (d, n) where d is the dimension and n is the number of samples
    """
    if type == 'numpy':
        x = np.asarray(x)
    elif type == 'autograd':
        x = anp.asarray(x)
    else:
        raise ValueError("Invalid type. Must be 'numpy' or 'autograd'.")

    return x[:, None] if x.ndim == 1 else x

def _chol_and_prec(cov: np.ndarray):
    """
    Compute Cholesky decomposition and precision matrix from covariance.
    
    Args:
        cov (np.ndarray): Covariance matrix of shape (d, d)
        
    Returns:
        tuple: (L, precision) where L is the Cholesky factor (Σ = L L^T) and precision is the inverse covariance
    """
    L = cholesky(cov)
    precision = np.linalg.inv(cov)
    
    return L, precision

def _mahalanobis_radius(x: np.ndarray, mean: np.ndarray, precision: np.ndarray):
    """
    Compute Mahalanobis distance for each sample.
    
    r(x) = sqrt((x-μ)^T Σ^{-1}(x-μ)) for input x of shape (d,n).
    
    Args:
        x (np.ndarray): Input data of shape (d, n) where d is dimension, n is number of samples
        mean (np.ndarray): Mean vector of shape (d, 1)
        precision (np.ndarray): Precision matrix (inverse covariance) of shape (d, d)
        
    Returns:
        np.ndarray: Mahalanobis radii of shape (n,) for each sample
    """
    xm = x - mean
    quad = np.einsum('ik,ij,jk->k', xm, precision, xm)

    return np.sqrt(quad + 1e-300)

def _mahalanobis_radius_anp(x: anp.ndarray, mean: anp.ndarray, precision: anp.ndarray):
    """
    Compute Mahalanobis distance for each sample using autograd numpy.
    
    r(x) = sqrt((x-μ)^T Σ^{-1}(x-μ)) for input x of shape (d,n).
    
    Args:
        x (anp.ndarray): Input data of shape (d, n) where d is dimension, n is number of samples
        mean (anp.ndarray): Mean vector of shape (d, 1)
        precision (anp.ndarray): Precision matrix (inverse covariance) of shape (d, d)
        
    Returns:
        anp.ndarray: Mahalanobis radii of shape (n,) for each sample
    """
    xm = x - mean
    quad = anp.einsum('ik,ij,jk->k', xm, precision, xm)

    return anp.sqrt(quad + 1e-300)


class MultivariateLaplace:
    """
    Symmetric Multivariate Laplace distribution with Bessel-K form.
    
    This class implements a multivariate Laplace distribution with elliptical contours.
    The density is given by:
    
    f(x) ∝ (r^2/2)^(-ν/2) * K_ν(sqrt(2)*r) / sqrt(det(Σ))
    
    where r = sqrt((x-μ)^T Σ^{-1}(x-μ)) is the Mahalanobis radius,
    K_ν is the modified Bessel function of the second kind, and ν = (2-dim)/2.
    
    Attributes:
        mean (np.ndarray): Mean vector of shape (dim, 1)
        covariance (np.ndarray): Covariance matrix of shape (dim, dim)
        precision (np.ndarray): Precision matrix (inverse covariance) of shape (dim, dim)
        L (np.ndarray): Cholesky factor of covariance matrix
        dim (int): Dimensionality of the distribution
        nu (float): Bessel-K order parameter
    """
    
    def __init__(self, mean, covariance):
        self.mean              = np.asarray(mean).reshape(-1, 1)
        self.covariance        = np.asarray(covariance)
        self.L, self.precision = _chol_and_prec(self.covariance)
        self.dim               = self.mean.shape[0]
        self.nu                = (2.0 - self.dim) / 2.0               # Bessel-K order

        sign, logdet = slogdet(self.covariance)

        assert sign > 0, "Covariance must be PD."

        self._log_c = np.log(2.0) - (self.dim/2.0)*np.log(2*np.pi) - 0.5*logdet

    def potential(self, x: np.ndarray):
        """
        Compute the potential (negative log-density) of the multivariate Laplace distribution.
        
        V(x) = - (nu/2) * log(r^2/2) - log K_nu(sqrt(2) * r) + const

        Args:
            x (np.ndarray): Input data of shape (dim, n)

        Returns:
            np.ndarray: Potential values of shape (n,)
        """
        x = _as_column(x)
        r = _mahalanobis_radius(x, self.mean, self.precision)
        z = np.sqrt(2.0) * r
        val = -self._log_c - (self.nu/2.0)*np.log((r*r)/2.0 + 1e-300) - np.log(Knu(self.nu, z) + 1e-300)

        return val[0] if val.shape == (1,) else val

    def grad_potential(self, x: np.ndarray):
        """
        Compute the gradient of the multivariate Laplace potential.
        
        Args:
            x (np.ndarray): Input data of shape (dim, n)

        Returns:
            np.ndarray: Gradient of shape (dim, n)
        """
        x     = _as_column(x)
        r     = _mahalanobis_radius(x, self.mean, self.precision)
        z     = np.sqrt(2.0) * r
        ratio = Knu(self.nu - 1.0, z) / (Knu(self.nu, z) + 1e-300)
        coeff = (np.sqrt(2.0) * ratio) / (r + 1e-300)
        xm    = x - self.mean
        grad  = (self.precision @ xm) * coeff[None, :]

        return grad[:, 0] if grad.shape[1] == 1 else grad
    
    def hess_potential(self, x: np.ndarray):
        """
        Compute the Hessian of the multivariate Laplace potential.

        Args:
            x (np.ndarray): Input data of shape (dim, n)
        
        Returns:
            np.ndarray: Hessian of shape (dim, dim) for a single point (dim,), or (dim, dim, n) for batched input (dim, n).
        """
        x  = _as_column(x)
        xm = x - self.mean
        A  = self.precision

        r = _mahalanobis_radius(x, self.mean, A)
        z = np.sqrt(2.0) * r

        # Bessel ratios
        Kn   = Knu(self.nu, z)
        Knm1 = Knu(self.nu - 1.0, z)
        eps  = 1e-300
        R    = Knm1 / (Kn + eps) # R(z) = K_{nu-1}/K_{nu}

        # Radial derivatives of V
        phi1 = np.sqrt(2.0) * R  # phi'(r)
        phi2 = 2.0 * (-1.0 + R**2 + ((2.0*self.nu - 1.0)/(z + eps)) * R)  # phi''(r)

        invr  = 1.0 / (r + eps)
        invr2 = invr**2
        invr3 = invr**3

        w_A     = phi1 * invr  # coefficient for A
        w_rank1 = phi2 * invr2 - phi1 * invr3  # coefficient for A xm xm^T A

        Axm   = A @ xm
        outer = np.einsum('ik,jk->ijk', Axm, Axm)

        H = A[:, :, None] * w_A[None, None, :] + outer * w_rank1[None, None, :]

        return H[:, :, 0] if H.shape[2] == 1 else H

    def sample(self, n_samples: int):
        """
        Generate samples from the multivariate Laplace distribution.
        
        Use the decomposition: X = μ + sqrt(Y) * (Z @ L^T)
        where Y ~ Exp(1) and Z ~ N(0, I).
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            np.ndarray: Samples of shape (dim, n_samples) where dim is the dimension
        """
        d = self.dim
        Y = np.random.exponential(scale=1.0, size=n_samples)
        Z = np.random.randn(n_samples, d)
        X = self.mean.flatten() + (np.sqrt(Y)[:, None] * (Z @ self.L.T))

        return X.T


class MultivariateLogistic:
    """
    Elliptical Multivariate Logistic distribution.
    
    This class implements a multivariate logistic distribution with elliptical contours.
    The density is given by: 

    f(x) = exp(-r/s)/(s * (1 + exp(-r/s))^2)

    where r = sqrt((x-μ)^T Σ^{-1}(x-μ)) is the Mahalanobis radius.
    
    Attributes:
        mean (np.ndarray): Mean vector of shape (d, 1)
        covariance (np.ndarray): Covariance matrix of shape (d, d)
        precision (np.ndarray): Precision matrix (inverse covariance) of shape (d, d)
        L (np.ndarray): Cholesky factor of covariance matrix
        dim (int): Dimensionality of the distribution
        s (float): Scale parameter
    """
    
    def __init__(self, mean, covariance, scale):
        self.mean              = np.asarray(mean).reshape(-1, 1)      # (d,1)
        self.covariance        = np.asarray(covariance)
        self.L, self.precision = _chol_and_prec(self.covariance)
        self.dim               = self.mean.shape[0]
        self.s                 = float(scale)
        
        assert self.s > 0
    
    def log_normalizing_constant(self):
        """
        Compute the log of the normalizing constant for the multivariate logistic distribution.
        
        Returns:
            float: Log of the normalizing constant
        """
        d, s = self.dim, self.s
        term1 = 0.5 * np.log(np.linalg.det(self.covariance))
        term2 = np.log(2) + (d/2)*np.log(np.pi) - gammaln(d/2)
        term3 = d * np.log(s)
        term4 = gammaln(d)
        term5 = np.log(1 - 2**(1-d))
        term6 = np.log(zeta(d, 1))

        return term1 + term2 + term3 + term4 + term5 + term6

    def potential(self, x: np.ndarray):
        """
        Compute the potential (negative log-density) of the multivariate logistic distribution.
        
        The potential is given by V(x) = r/s + 2*log(1 + exp(-r/s)) + const
        
        Args:
            x (np.ndarray): Input data of shape (dim, n)
            
        Returns:
            np.ndarray: Potential values of shape (n,)
        """
        x  = _as_column(x)
        r  = _mahalanobis_radius(x, self.mean, self.precision)
        s  = self.s
        val = (r / s) + 2.0 * np.log(1.0 + np.exp(-r / s))
        val +=  self.log_normalizing_constant()

        return val[0] if val.shape == (1,) else val

    def grad_potential(self, x: np.ndarray):
        """
        Compute the gradient of the multivariate logistic potential.
        
        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)
            
        Returns:
            np.ndarray: Gradient of shape (dim, n)
        """
        x  = _as_column(x)
        r  = _mahalanobis_radius(x, self.mean, self.precision)
        s  = self.s
        coeff = (np.tanh(r / (2.0 * s)) / (s * (r + 1e-300)))
        xm    = x - self.mean
        grad  = (self.precision @ xm) * coeff[None, :]

        return grad[:, 0] if grad.shape[1] == 1 else grad
    
    def hess_potential(self, x: np.ndarray):
        """
        Compute the Hessian of the multivariate logistic potential.

        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)

        Returns:
            np.ndarray: Hessian of shape (dim, dim) for a single point (dim,), or (dim, dim, n) for batched input (dim, n).
        """
        x  = _as_column(x)
        xm = x - self.mean
        A  = self.precision
        s  = self.s

        r = _mahalanobis_radius(x, self.mean, A)
        eps = 1e-300
        u = r / (2.0 * s)

        # Radial derivatives
        t = np.tanh(u)                                      # tanh(r/(2s))
        phi1 = t / s                                        # phi'(r)
        sech2 = 1.0 - t**2                                  # sech^2(r/(2s))
        phi2 = 0.5 * sech2 / (s**2)                         # phi''(r)

        invr  = 1.0 / (r + eps)
        invr2 = invr**2
        invr3 = invr**3

        w_A     = phi1 * invr                               # coeff for A
        w_rank1 = phi2 * invr2 - phi1 * invr3               # coeff for A xm xm^T A

        Axm   = A @ xm
        outer = np.einsum('ik,jk->ijk', Axm, Axm)

        H = A[:, :, None] * w_A[None, None, :] + outer * w_rank1[None, None, :]

        return H[:, :, 0] if H.shape[2] == 1 else H

    def sample(self, n_samples: int, max_trials_factor: int = 5):
        """
        Generate samples from the multivariate logistic distribution using 
        rejection sampling with:
        - Proposal: R ~ Gamma(shape=dim, scale=s)
        - Accept with probability: 1/(1+e^{-r/s})^2
        
        Args:
            n_samples (int): Number of samples to generate
            max_trials_factor (int, optional): Multiplier for maximum rejection trials. Defaults to 5.
            
        Returns:
            np.ndarray: Samples of shape (dim, n_samples)
        """
        d, s = self.dim, self.s

        max_draws = int(max_trials_factor * n_samples)
        Rs = np.random.gamma(shape=d, scale=s, size=max_draws)
        accept = np.random.rand(max_draws) < 1.0 / (1.0 + np.exp(-Rs / s))**2

        Rs = Rs[accept]

        if Rs.size < n_samples:
            while Rs.size < n_samples:
                R_more = np.random.gamma(shape=d, scale=s, size=max_draws)
                acc_more = np.random.rand(max_draws) < 1.0 / (1.0 + np.exp(-R_more / s))**2
                Rs = np.concatenate([Rs, R_more[acc_more]])

        Rs = Rs[:n_samples]
       
        U = np.random.randn(n_samples, d)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        X = self.mean.flatten() + (U * Rs[:, None]) @ self.L.T
        
        return X.T


class MultivariateGaussian:
    """
    Multivariate Gaussian (Normal) distribution.
    
    This class implements a multivariate Gaussian distribution with elliptical contours.
    The distribution is parameterized by a mean vector and covariance matrix.
    The density is given by:
    
    f(x) = (2π)^{-d/2} * det(Σ)^{-1/2} * exp(-0.5 * r^2),

    where r = sqrt((x-μ)^T Σ^{-1}(x-μ)) is the Mahalanobis radius.
    
    Attributes:
        mean (np.ndarray): Mean vector
        covariance (np.ndarray): Covariance matrix of shape (dim, dim)
        precision (np.ndarray): Precision matrix (inverse covariance) of shape (dim, dim)
        dim (int): Dimensionality of the distribution
    """
    
    def __init__(self, mean, covariance):
        self.mean       = mean
        self.covariance = covariance
        self.precision  = np.linalg.inv(covariance)
        self.dim        = mean.shape[0]

    def potential(self, x: np.ndarray):
        """
        Compute the potential (negative log-density) of the multivariate Gaussian distribution.
        
        The potential is given by V(x) = 0.5 * r^2 + const
        
        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)
            
        Returns:
            np.ndarray: Potential values of shape (n,)
        """
        x = _as_column(x, type='autograd')

        r_squared = _mahalanobis_radius_anp(x, self.mean, self.precision)**2
        const     = 0.5 * (self.dim * anp.log(2 * anp.pi) - anp.log(anp.linalg.det(self.precision)))
        result    = 0.5 * r_squared + const

        return result[0] if result.shape == (1,) else result

    def grad_potential(self, x: np.ndarray):
        """
        Compute the gradient of the multivariate Gaussian potential.
                
        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)
            
        Returns:
            np.ndarray: Gradient of shape (dim, n)
        """
        x = _as_column(x, type='autograd')

        grad = self.precision @ (x - self.mean)

        return grad[:, 0] if grad.shape[1] == 1 else grad
    
    def hess_potential(self, x: np.ndarray | None = None):
        """
        Compute the Hessian of the multivariate Gaussian potential.
                
        Args:
            x (np.ndarray | None): Input data. Not used for Gaussian as Hessian is constant.
            
        Returns:
            np.ndarray: Hessian matrix of shape (dim, dim)
        """
        return self.precision

    def sample(self, n_samples: int):
        """
        Generate samples from the multivariate Gaussian distribution.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            np.ndarray: Samples of shape (dim, n_samples)
        """
        samples = np.random.multivariate_normal(mean=self.mean.flatten(), cov=self.covariance, size=n_samples)
       
        return samples.T


class MultivariateStudent:
    """
    Multivariate Student-t distribution.
    
    This class implements a multivariate Student-t distribution with elliptical contours.
    The distribution is parameterized by a mean vector, scale matrix, and degrees of freedom.
    The density is given by:
    
    f(x) ∝ [1 + (1/ν)r^2]^{-(ν+dim)/2}

    where r = sqrt((x-μ)^T Σ^{-1}(x-μ)) is the Mahalanobis radius and ν is the degrees of freedom parameter.
    
    Attributes:
        mean (np.ndarray): Mean vector
        scale (np.ndarray): Scale matrix of shape (dim, dim)
        precision (np.ndarray): Precision matrix (inverse scale) of shape (dim, dim)
        dim (int): Dimensionality of the distribution
        dof (float): Degrees of freedom parameter
    """
    
    def __init__(self, mean, scale, dof):
        self.mean      = mean
        self.scale     = scale
        self.precision = anp.linalg.inv(scale)
        self.dim       = mean.shape[0]
        self.dof       = float(dof)

    def potential(self, x: np.ndarray):
        """
        Compute the potential (negative log-density) of the multivariate Student-t distribution.
        
        The potential is given by V(x) = 0.5*(ν+d) * log(1 + r^2/ν) + const
        
        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)
            
        Returns:
            np.ndarray: Potential values of shape (n,)
        """
        x = _as_column(x, type='autograd')
        r_squared = _mahalanobis_radius_anp(x, self.mean, self.precision)**2
        const     = -0.5 * anp.log(anp.linalg.det(self.precision)) + 0.5 * self.dim * anp.log(self.dof * anp.pi) + gammaln(self.dof / 2.0) - gammaln((self.dof + self.dim) / 2.0)
        result    = 0.5 * (self.dof + self.dim) * anp.log1p(r_squared / self.dof) + const
        
        return result[0] if result.shape == (1,) else result

    def grad_potential(self, x: np.ndarray):
        """
        Compute the gradient of the multivariate Student-t potential.
                
        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)
            
        Returns:
            np.ndarray: Gradient of shape (dim, n)
        """
        x = _as_column(x)
        xm        = x - self.mean
        r_squared = _mahalanobis_radius_anp(x, self.mean, self.precision)**2
        weight    = (self.dof + self.dim) / (self.dof + r_squared)
        grad      = (self.precision @ xm) * weight[None, :]
        return grad[:, 0] if grad.shape[1] == 1 else grad
    
    def hess_potential(self, x: np.ndarray):
        """
        Compute the Hessian of the multivariate Student-t potential.
        
        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)
            
        Returns:
            np.ndarray: Hessian of shape (dim, dim) for a single point or (dim, dim, n) for batched input
        """
        x = _as_column(x, type='autograd')
        A         = self.precision
        xm        = x - self.mean
        r_squared = _mahalanobis_radius_anp(x, self.mean, self.precision)**2
        c         = (self.dof + self.dim)
        w1        = c / (self.dof + r_squared)
        w2        = 2.0 * c / (self.dof + r_squared)**2
        Axm       = A @ xm
        outer     = anp.einsum('ik,jk->ijk', Axm, Axm)
        H = A[:, :, None] * w1[None, None, :] - outer * w2[None, None, :]

        return H[:, :, 0] if H.shape[2] == 1 else H

    def sample(self, n_samples: int):
        """
        Generate samples from the multivariate Student-t distribution.
        
        Uses the transformation method:
        1. Generate u ~ χ²(ν)
        2. Generate z ~ N(0, Σ)
        3. Transform to Student-t: x = μ + z * √(ν/u)
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            np.ndarray: Samples of shape (dim, n_samples)
        """
        z = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.scale, size=n_samples)
        u = np.random.chisquare(self.dof, size=n_samples)   
        s = anp.sqrt(self.dof / u)[:, None]                
        samples = self.mean.flatten() + z * s    

        return samples.T


class NealsFunnel:
    """
    Neal's funnel distribution.

    This class implements Neal's funnel, a hierarchical distribution in dimension dim >= 2.
    The variable y = (z, x_1, ..., x_{dim-1}) in R^dim is defined by:

        z ~ N(0, sigma^2)
        x_i | z ~ N(0, exp(z)),  i = 1, ..., dim-1

    Attributes:
        dim (int): Dimensionality of the distribution (dim >= 2)
        sigma (float): Scale parameter for the latent z
    """

    def __init__(self, dim, sigma):
        assert dim >= 2, "Neal's funnel needs dim >= 2"
        self.dim   = int(dim)
        self.sigma = float(sigma)

    def potential(self, x: np.ndarray):
        """
        Compute the potential (negative log-density) of Neal's funnel distribution.

        The potential is given by V(y) = 0.5*z^2/sigma^2 + 0.5*exp(-z)*||x||^2 + 0.5*(dim-1)*z + const.

        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)

        Returns:
            np.ndarray: Potential values of shape (n,)
        """
        x = _as_column(x, type='autograd')
        z  = x[0, :]
        xs = x[1:, :]
        k  = self.dim - 1

        S = anp.sum(xs**2, axis=0)
        const = 0.5 * (self.dim * anp.log(2 * anp.pi)) + anp.log(self.sigma)
        V = 0.5 * (z**2 / self.sigma**2) + 0.5 * (anp.exp(-z) * S) + 0.5 * (self.dim-1) * z + const

        return V[0] if V.shape == (1,) else V

    def grad_potential(self, x: np.ndarray):
        """
        Compute the gradient of the Neal's funnel potential, ∇V(x).

        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)

        Returns:
            np.ndarray: Gradient of shape (dim, n)
        """
        x = _as_column(x, type='autograd')
        z  = x[0, :]
        xs = x[1:, :]

        S  = anp.sum(xs**2, axis=0)
        e_z = anp.exp(-z)

        # Build Derivatives
        dz  = z / (self.sigma**2) - 0.5 * e_z * S + 0.5 * (self.dim-1) # dV/dz
        dxs = e_z[None, :] * xs                                        # dV/dx_i

        grad = anp.vstack([dz[None, :], dxs])

        return grad[:, 0] if grad.shape[1] == 1 else grad

    def hess_potential(self, x: np.ndarray):
        """
        Compute the Hessian of the Neal's funnel potential, ∇^2 V(x).

        Args:
            x (np.ndarray): Input data of shape (dim,) or (dim, n)

        Returns:
            np.ndarray: Hessian of shape (dim, dim) for a single point (dim,), or (dim, dim, n) for batched input (dim, n).
        """
        x = _as_column(x, type='autograd')
        z  = x[0, :]
        xs = x[1:, :]

        S = anp.sum(xs**2, axis=0)
        e_z = anp.exp(-z)

        n = x.shape[1]
        d = self.dim

        H = anp.zeros((d, d, n))

        # Build Hessian
        H[0, 0, :] = 1.0 / (self.sigma**2) + 0.5 * e_z * S         # d^2 V / dz^2

        H[0, 1:, :] = -e_z[None, :] * xs                           # cross terms d^2 V / dz dx_i
        H[1:, 0, :] = -e_z[None, :] * xs

        for i in range(d - 1):
            H[1 + i, 1 + i, :] = e_z                               # diagonal d^2 V / dx_i^2

        return H[:, :, 0] if n == 1 else H

    def sample(self, n_samples: int):
        """
        Generate samples from Neal's funnel distribution.

        Uses the hierarchical definition: 
        - z ~ N(0, sigma^2)
        - x_i | z ~ N(0, exp(z)), i = 1, ..., dim-1

        Args:
            n_samples (int): Number of samples to generate

        Returns:
            np.ndarray: Samples of shape (dim, n_samples)
        """
        z    = np.random.normal(loc=0.0, scale=self.sigma, size=n_samples)
        stds = np.exp(0.5 * z)[:, None]
        xs   = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, self.dim-1)) * stds

        y = np.concatenate([z[:, None], xs], axis=1)

        return y.T