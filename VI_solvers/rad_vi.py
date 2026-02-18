from typing import Callable, Optional, Tuple
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from math import lgamma, exp
import numpy as np
import sys

sys.path.insert(1,'../')

from utils.basis_functions import build_psi, build_dpsi, build_gram_matrix

class RadVI:
    def __init__(self, 
                 truncation: float, 
                 mesh: float, 
                 dim: int, 
                 alpha: float, 
                 isotropic: bool,
                 V: Optional[Callable] = None, 
                 grad_V: Optional[Callable] = None, 
                 lambda_arr: Optional[np.ndarray] = None, 
                 mean: Optional[np.ndarray] = None, 
                 covariance: Optional[np.ndarray] = None,
                 radial_ot_map: Optional[Callable] = None,
                 N_mc: int = 10**5):

        # radVI parameters
        self.truncation = truncation
        self.mesh       = mesh
        self.dim        = dim
        self.alpha      = alpha
        self.isotropic  = isotropic
        self.J          = int(2 * self.truncation / self.mesh) + 1

        # SPGD parameters
        self.V          = V
        self.grad_V     = grad_V
        self.N_mc       = N_mc
        self.lambda_arr = np.ones(self.J) if lambda_arr is None else lambda_arr

        # Wasserstein distance parameters
        self.radial_ot_map = radial_ot_map

        # Whitening parameters
        self.mean        = mean
        self.covariance  = covariance
        self.is_whitened = not self.isotropic

        # Build dictionary of basis functions
        self.psi_functions, _ = build_psi(truncation = self.truncation, mesh = self.mesh, dim = self.dim)
        self.dpsi_functions   = build_dpsi(truncation = self.truncation, mesh = self.mesh, dim = self.dim)
        
        # Build Gram matrix and its pseudo-inverse
        self.Q, self.Qinv = build_gram_matrix(truncation = self.truncation, 
                                              mesh = self.mesh, 
                                              dim = self.dim, 
                                              psi_function_means = np.zeros(len(self.psi_functions)))
    
    def _compute_whitened_gradient(self, x: np.ndarray):
        """
        Computes the gradient of the whitened potential function tilde{V}

        ∇tilde{V} = Σ^{-1/2} ∇V(Σ^{1/2} x + μ)

        Args:
            x (np.ndarray): Input data of shape (dim, n_samples)

        Returns:
            np.ndarray: Gradient of the whitened potential function of shape (dim, n_samples)
        """
        Sigma_half = sqrtm(self.covariance)

        return Sigma_half @ self.grad_V(Sigma_half @ x + self.mean)

    def _grad_v_of_transport_map(self, x: np.ndarray):
        """
        Computes the gradient of the potential function acting  on the push-forward: ∇V(T(x)). If distribution 
        is anisotropic, we compute the whitened gradient.

        Args:
            x (np.ndarray): Input data of shape (dim, n_samples)

        Returns:
            np.ndarray: Gradient of the potential function acting on the push-forward of shape (dim, n_samples)
        """
        radii = np.linalg.norm(x, axis=0) 

        psi_vals = np.vstack([psi(radii) for psi in self.psi_functions])

        transport_map = (self.alpha + np.dot(self.lambda_arr, psi_vals)/radii) * x

        if self.isotropic:
            G = self.grad_V(transport_map)
        else:
            G = self._compute_whitened_gradient(transport_map) 

        x_dot_g = np.einsum('dn,dn->n', x, G)

        return (psi_vals * x_dot_g) / radii

    def _log_det_jacobian(self, x: np.ndarray):
        radii     = np.linalg.norm(x, axis=0)
        psi_vals  = np.array([fn(radii) for fn in self.psi_functions])
        dpsi_vals = np.array([fn(radii) for fn in self.dpsi_functions])

        term1 = (self.dim - 1) * np.log(self.alpha + np.dot(self.lambda_arr, psi_vals)/radii )
        term2 = np.log(self.alpha + np.dot(self.lambda_arr, dpsi_vals))

        return term1 + term2

    def _grad_log_det_mc(self, x: np.ndarray):
        """
        Compute gradient of log determinant of the derivative of the transport map: ∇_λ log det DT(x)
        
        Args:
            x: Input data of shape (dim, n_samples)
            
        Returns:
            np.ndarray: ∇_λ log det DT(x) with shape (n_basis_functions, n_samples)
        """
        radii     = np.linalg.norm(x, axis=0)
        psi_vals  = np.array([fn(radii) for fn in self.psi_functions])
        dpsi_vals = np.array([fn(radii) for fn in self.dpsi_functions])

        term1 = ((self.dim - 1) * psi_vals) / (self.alpha*radii + np.dot(self.lambda_arr, psi_vals))
        term2 = (dpsi_vals)/(self.alpha + np.dot(self.lambda_arr, dpsi_vals))

        return term1 + term2

    def _compute_kl_gradient(self, x: np.ndarray):
        """
        Compute ∇_λ KL(T(x) || π), for π the target distribution, via Monte Carlo integration.

        Args:
            x (np.ndarray): Input data of shape (dim, n_samples)

        Returns:
            np.ndarray: ∇_λ log det DT(x) with shape (n_basis_functions, n_samples)
        """
        g1 = np.mean(self._grad_v_of_transport_map(x), axis=1)
        g2 = np.mean(self._grad_log_det_mc(x), axis=1)

        return g1 - g2

    def _compute_wasserstein_distance(self):
        """
        Compute the Wasserstein distance between the radVI map and the true radial map.

        Returns:
            float: Wasserstein distance
        """
        samples_gaussian = np.random.randn(self.dim, self.N_mc)
        r_samples_radial = np.linalg.norm(samples_gaussian, axis=0)
        
        radvi_pushforward_samples = self._construct_radial_part()(r_samples_radial)
        otmap_pushforward_samples = self.radial_ot_map(r_samples_radial)

        return np.mean((radvi_pushforward_samples - otmap_pushforward_samples)**2)
        
    def _qproj(self, target: np.ndarray, previous: np.ndarray):
        """
        Project the current lambda_arr onto the Q-space.

        Args:
            target (np.ndarray): Target lambda_arr of shape (n_basis_functions,)
            previous (np.ndarray): Previous lambda_arr of shape (n_basis_functions,)

        Returns:
            np.ndarray: Projected lambda_arr of shape (n_basis_functions,)
        """
        M = len(self.Q)
        bounds = [(0.0, None)] * M
        try:
            res = minimize(fun=lambda l: np.dot(l - target, self.Q @ (l - target)),
                           x0=previous,
                           jac=lambda l: self.Q @ (l - target),
                           bounds=bounds,
                           method='L-BFGS-B')
            return res.x
            
        except Exception as e:
            print(f"Qproj error: {e}")

            return np.maximum(target, 0.0)
    
    def _construct_radial_part(self):
        """
        Constucts the radial component of the transport map.
        For r = ||x||, T_rad(x) = α*r + <λ,ψ(r)>

        Returns:
            Callable: Radial transport map
        """
        def T_rad(r: np.ndarray):
            psi_vals = np.array([fn(r) for fn in self.psi_functions])

            return (self.alpha*r + np.dot(self.lambda_arr, psi_vals))

        return T_rad

    def _construct_radial_transport_map(self):
        """
        Consuct the full transport map. For r = ||x||, T(x) = (α + <λ,ψ(r)>/r)*x

        Returns:
            Callable: Full transport map
        """
        def T(x: np.ndarray):
            radii    = np.linalg.norm(x, axis=0)
            psi_vals = np.array([fn(radii) for fn in self.psi_functions])
            return (self.alpha + np.dot(self.lambda_arr, psi_vals)/radii) * x
        return T
    
    def _construct_composite_transport_map(self):
        """
        Function which computes the composite transport map. This is the 
        transport map following whitening. 
        T_comp(x) = Σ^{1/2}(T_rad(x)) + μ

        Returns:
            Callable: Composite transport map
        """
        def T_comp(x: np.ndarray):
            T_rad         = self._construct_radial_transport_map()
            x_transformed = T_rad(x)
            Sigma_half    = sqrtm(self.covariance)

            return np.dot(Sigma_half, x_transformed) + self.mean

        return T_comp
    
    def _construct_transport_map(self):
        """
        Function which constructs the transport map. If the distribution is isotropic,
        we use the radial transport map. Otherwise, we use the composite transport map.

        Returns:
            Callable: Transport map
        """
        if not self.is_whitened:
            return self._construct_radial_transport_map()
        
        return self._construct_composite_transport_map()

    def fit(self, 
            learning_rate: float, 
            num_iterations: int, 
            sample_size: int, 
            compute_wasserstein: bool = False, 
            log_w2_freq: int = 1,
            print_freq: int = 1000):
        """
        Stochastic gradient descent optimization algorithm for learning the lambdas.

        Args:
            learning_rate (float): Learning rate
            num_iterations (int): Number of iterations
            sample_size (int): Sample size
            compute_wasserstein (bool, optional): Whether to compute the Wasserstein distance. Defaults to False.
            log_w2_freq (Optional[int], optional): Frequency of logging the Wasserstein distance. Defaults to None.
            print_freq (Optional[int], optional): Frequency of printing the iteration number. Defaults to 1000.
        """
        print("\n" + "-"*50)
        print("       Fitting radVI Approximation ")
        print("-"*50)
        print(f"• Mesh            : {self.mesh}")
        print(f"• Truncation      : {self.truncation}")
        print(f"• Alpha           : {self.alpha}")
        print(f"• Num. basis (J)  : {self.J}")
        print(f"• Whitening       : {'Yes' if self.is_whitened else 'No'}\n")
        print(f"Optimizer settings:")
        print(f"• Learning rate   : {learning_rate}")
        print(f"• Iterations      : {num_iterations}")
        print(f"• Sample size     : {sample_size}")
        print("-"*50 + "\n")

        if compute_wasserstein:
            print(f'Computing Wasserstein distance every {log_w2_freq} iterations\n')
            self.wasserstein_history = []

        for iteration in range(num_iterations):
            x = np.random.randn(self.dim, sample_size)
            grad_kl = self._compute_kl_gradient(x)
            
            step = self.Qinv @ grad_kl  
            
            next_lambda = self.lambda_arr - learning_rate * step

            if (next_lambda < 0).any():
                self.lambda_arr = self._qproj(next_lambda, self.lambda_arr)
            else:
                self.lambda_arr = next_lambda

            if (iteration + 1) % print_freq == 0:
                print(f"Iteration {iteration + 1} of {num_iterations}")

            if compute_wasserstein and (iteration % log_w2_freq == 0 or iteration == num_iterations - 1):
                wasserstein = self._compute_wasserstein_distance()
                self.wasserstein_history.append(wasserstein)

        self.transport_map = self._construct_transport_map()

        print(f'Completed fitting radVI approximation.\n')
            
    def push_forward(self, x: np.ndarray):
        """
        Computes the push-forward of the data through the transport map.

        Args:
            x (np.ndarray): Input data of shape (dim, n_samples)

        Returns:
            np.ndarray: Push-forward of the data through the transport map of shape (dim, n_samples)
        """
        if self.transport_map is None:
            raise ValueError("Model must be fitted before calling push_forward")
        
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected input with shape (dim={self.dim}, n_samples), got {x.shape}")
        
        return self.transport_map(x)
    
    def get_lambdas(self):
        """
        Returns the most recent lambdas values

        Returns:
            np.ndarray: Lambdas of shape (n_basis_functions,)
        """
        
        return self.lambda_arr.copy()
