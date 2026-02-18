import numpy as np
from typing import Callable, Optional, Tuple
from scipy.linalg import sqrtm, inv
import sys
sys.path.insert(1,'../')
from utils.elliptical_distributions import MultivariateGaussian

class Gaussian_FBVI:
    """
    Forward-backward Gaussian variational inference algorithm.
    """

    def __init__(self, 
                 learning_rate: float, 
                 dim: int, 
                 grad_V: Callable, 
                 hess_V: Callable, 
                 mean: Optional[np.ndarray] = None, 
                 covariance: Optional[np.ndarray] = None):

        self.dim           = dim
        self.learning_rate = learning_rate

        # Whitening parameters
        self.mean    = np.zeros((dim, 1)) if mean is None else mean
        self.sigma   = np.eye(dim) if covariance is None else covariance

        self.grad_V = grad_V
        self.hess_V = hess_V

    def fit(self, num_iterations: int, sample_size: int, print_freq: int = 1000):
        print("\n" + "-"*50)
        print("       Fitting Gaussian FBVI Approximation ")
        print("-"*50)
        print(f"Optimizer settings:")
        print(f"• Learning rate   : {self.learning_rate}")
        print(f"• Iterations      : {num_iterations}")
        print(f"• Sample size     : {sample_size}")
        print("-"*50 + "\n")

        for iteration in range(num_iterations):
            sqrtcov = sqrtm(self.sigma)
            
            x = self.mean + sqrtcov @ np.random.randn(self.dim, 1)

            hat_nabla1 = self.grad_V(x).reshape(-1,1)
            hat_nabla2 = self.hess_V(x)

            self.mean = self.mean - self.learning_rate * hat_nabla1
            self.mean = np.real(self.mean)

            M_half     = np.eye(self.dim) - self.learning_rate * hat_nabla2
            Sigma_half = M_half @ self.sigma @ M_half

            sqrt_matrix  = sqrtm(Sigma_half @ (Sigma_half + 4 * self.learning_rate * np.eye(self.dim)))
            sqrt_matrix  = np.real(sqrt_matrix)
            self.sigma   = 0.5 * (Sigma_half + 2 * self.learning_rate * np.eye(self.dim) + sqrt_matrix)

            if (iteration + 1) % print_freq == 0:
                print(f"Iteration {iteration + 1} of {num_iterations}")

        print(f'Completed fitting Gaussian FBVI approximation.\n')
    