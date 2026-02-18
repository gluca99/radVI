import sys, time
import numpy as np
from scipy.stats import chi2, f
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import brentq, root_scalar
from scipy.special import gammainc, betaincinv, kv, gamma
from scipy.stats import chi2


sys.path.insert(1,'../')

def identity_transport_map(r: np.ndarray):
    """
    Computes the transport map from a standard Gaussian to a standard Gaussian.

    Args:
        r (np.ndarray): The radius values to transport.

    Returns:
        np.ndarray: The transported radius values.
    """
    
    return r

def student_t_map(r: np.ndarray, dof: float, dim: int):
    """
    Computes the radial transport map from a standard Gaussian to the isotropic Student-t distribution.

    T(r) = sqrt(dimension * F^{-1}_{d, nu}(F_{chi^{2}_{d}}(r^2)))

    Args:
        r (np.ndarray): The radius values to transport.
        dof (float): The degrees of freedom of the Student-t distribution.
        dim (int): The dimension of the Student-t distribution.

    Returns:
        np.ndarray: The transported radius values.
    """        
    chi2_cdf_vals = chi2.cdf(r**2, df=dim) # F(r^2)

    f_ppf_vals = f.ppf(chi2_cdf_vals, dfn=dim, dfd=dof) # F^{-1}_{d, nu}( F(r^2) )

    new_radius = np.sqrt(dim * f_ppf_vals) # sqrt( dimension * F^{-1}_{d, nu}( F_{chi^{2}_{d}}(r^2) ) )

    return new_radius

class GaussianToLogisticOTMap:
    """
    Computes the radial transport map from a standard Gaussian to the isotropic 
    logistic distribution using an interpolation method.
    """
    def __init__(self, dim: int, scale: float = 1.0, grid_size: int = 10000):
        self.dim = dim
        self.scale = scale

        # 1. Create a grid of R values.
        upper_bound = self.dim * self.scale + 30 * np.sqrt(2 * self.dim) * self.scale
        R_grid = np.linspace(0, upper_bound, grid_size)

        # 2. Evaluate the probability density on this grid
        pdf_values = self._log_space_integrand(R_grid)
        
        # 3. Compute the cumulative integral (unnormalized CDF)
        unnormalized_cdf = cumulative_trapezoid(pdf_values, R_grid, initial=0)
        
        # 4. Normalize the CDF
        self.denominator = unnormalized_cdf[-1]
        cdf_values = unnormalized_cdf / self.denominator

        # 5. Create the inverse CDF interpolator
        self.inverse_cdf_interpolator = interp1d(cdf_values, 
                                                 R_grid,
                                                 kind='linear',
                                                 bounds_error=False,
                                                 fill_value=(0, upper_bound))

    def _log_space_integrand(self, s: np.ndarray):
        log_val = np.full(s.shape, -np.inf, dtype=np.float64)
        nonzero_mask = s > 1e-9
        s_nz = s[nonzero_mask]
        
        arg = s_nz / self.scale
        log_val_nz = (self.dim - 1) * np.log(s_nz) - arg - 2 * np.log(1 + np.exp(-arg))
        
        log_val[nonzero_mask] = log_val_nz
        return np.exp(log_val)

    def transform(self, r: np.ndarray):
        # 1. Compute the CDF values of the source distribution
        source_cdf_vals = chi2.cdf(r**2, df=self.dim)

        # 2. Use the inverse CDF interpolator to compute the transported values
        R  = self.inverse_cdf_interpolator(source_cdf_vals)

        # 3. Mask out the values that are too small
        nonzero_mask = r > 1e-9
        
        return R[nonzero_mask]

class GaussianToLaplaceOTMap:
    """
    Computes the OT map from a standard Gaussian to an isotropic Laplace
    distribution using an interpolation method.
    """
    def __init__(self, dim: int, grid_size: int = 10000):
        self.dim = dim

        # 1. Create a grid of R values.
        upper_bound = 15 * self.dim
        R_grid = np.linspace(0, upper_bound, grid_size)

        # 2. Evaluate the probability density on this grid
        pdf_values = self._log_space_integrand_laplace(R_grid)
        
        # 3. Compute the cumulative integral (unnormalized CDF)
        unnormalized_cdf = cumulative_trapezoid(pdf_values, R_grid, initial=0)
        
        # 4. Normalize the CDF
        total_integral = unnormalized_cdf[-1]
        cdf_values = unnormalized_cdf / total_integral

        # 5. Create the inverse CDF interpolator
        self.inverse_cdf_interpolator = interp1d(cdf_values, 
                                                 R_grid,
                                                 kind='linear',
                                                 bounds_error=False,
                                                 fill_value=(0, upper_bound))

    def _log_space_integrand_laplace(self, s: np.ndarray):
        log_val = np.full(s.shape, -np.inf, dtype=np.float64)
        nonzero_mask = s > 1e-9
        s_nz = s[nonzero_mask]
        
        v = abs(1.0 - self.dim / 2.0)
        z = np.sqrt(2) * s_nz
        
        kv_values = kv(v, z)
        valid_kv_mask = kv_values > 0
        
        final_mask = nonzero_mask.copy()
        final_mask[nonzero_mask] = valid_kv_mask
        
        if np.any(final_mask):
            s_final  = s[final_mask]
            kv_final = kv_values[valid_kv_mask]
            
            log_val_final = (self.dim / 2.0) * np.log(s_final) + np.log(kv_final) 
            
            log_val[final_mask] = log_val_final
            
        return np.exp(log_val)

    def transform(self, r: np.ndarray):
        # 1. Compute the CDF values of the source distribution
        source_cdf_vals = chi2.cdf(r**2, df=self.dim)
        
        # 2. Use the inverse CDF interpolator to compute the transported values
        R = self.inverse_cdf_interpolator(source_cdf_vals)
        
        # 3. Mask out the values that are too small
        nonzero_mask = r > 1e-9
        
        return R[nonzero_mask]

