import numpy as np
from scipy.optimize import minimize
from autograd import hessian
from typing import Callable, Optional, Tuple


class Gaussian_MFVI:
    """
    Mean-field Gaussian variational inference algorithm.
    """

    def __init__(self, 
                 stepsize_mean: float, 
                 stepsize_cov: float, 
                 dim: int, 
                 grad_V: Callable,
                 mean: Optional[np.ndarray] = None, 
                 diagcovariance: Optional[np.ndarray] = None):

        self.dim                = dim
        self.learning_rate_mean = stepsize_mean
        self.learning_rate_cov  = stepsize_cov

        # Whitening parameters
        self.mean       = np.zeros((dim, 1)) if mean is None else mean
        self.diagLambda = np.ones((dim, 1)) if diagcovariance is None else diagcovariance

        self.grad_V = grad_V

    def fit(self, num_iterations: int, sample_size: int, print_freq: int = 1000):
        print("\n" + "-"*50)
        print("       Fitting Gaussian MFVI Approximation ")
        print("-"*50)
        print(f"Optimizer settings:")
        print(f"• Learning rate for mean   : {self.learning_rate_mean}")
        print(f"• Learning rate for covariance   : {self.learning_rate_cov}")
        print(f"• Iterations      : {num_iterations}")
        print(f"• Sample size     : {sample_size}")
        print("-"*50 + "\n")

        for iteration in range(num_iterations):
            x = np.random.randn(self.dim, sample_size)

            transport_samples = self.diagLambda * x + self.mean
            grad_V_samples    = self.grad_V(transport_samples)

            grad_kl_m   = np.mean(grad_V_samples, axis=1).reshape(self.dim, 1)
            grad_kl_cov = (x * grad_V_samples).mean(axis=1, keepdims=True) - 1/self.diagLambda

            next_mean       = self.mean - self.learning_rate_mean * grad_kl_m            
            next_diagLambda = self.diagLambda - self.learning_rate_cov * grad_kl_cov

            self.mean       = next_mean
            self.diagLambda = next_diagLambda

            if (iteration + 1) % print_freq == 0:
                print(f"Iteration {iteration} of {num_iterations}")
    
        print(f'Completed fitting Gaussian MFVI approximation.\n')