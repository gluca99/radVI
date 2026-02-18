import numpy as np
from scipy.special import gammainc, gamma
from scipy import special

def chi_tail_moment(n: float, a: float, d: int):
    """
    Compute the tail moment E[R^n 1_{R >= a}] for R ~ chi(d).

    Args:
        n (float): Order of the moment (can be 0, 1, 2, etc.)
        a (float): Lower bound for the tail region (can be -∞)
        d (int): Degrees of freedom parameter for chi distribution
        
    Returns:
        float: E[R^n 1_{R >= a}] where R ~ chi(d)
    """
    n        = float(n)
    s        = 0.5 * (d + n)
    upper_inc = special.gammaincc(s, 0.5 * a**2) * special.gamma(s)  # Γ(s, a^2/2)

    return np.power(2.0, 0.5 * n) * upper_inc / special.gamma(0.5 * d)

def chi_trunc_moment(n: float, a: float, b: float, d: int):
    """
    Compute the truncated moment E[R^n 1_{a <= R <= b}] for R ~ chi(d).
    
    Uses the difference of upper incomplete gamma functions to compute moments
    of the chi distribution truncated to the interval [a, b].
    
    Args:
        n (float): Order of the moment (can be 0, 1, 2, etc.)
        a (float): Lower bound (can be -∞)
        b (float): Upper bound (can be +∞)
        d (int): Degrees of freedom parameter for chi distribution
        
    Returns:
        float: E[R^n 1_{a <= R <= b}] where R ~ chi(d)
    """
    n          = float(n)
    s          = 0.5 * (d + n)

    # Γ(s, a^2/2) - Γ(s, b^2/2)
    gamma_diff = (special.gammaincc(s, 0.5 * a**2) - special.gammaincc(s, 0.5 * b**2)) * special.gamma(s)

    return np.power(2.0, 0.5 * n) * gamma_diff / special.gamma(0.5 * d)

def chi_pdf(d: int, r: np.ndarray):
    """
    Compute the probability density function of the chi distribution.
    
    Args:
        d (int): Degrees of freedom parameter
        r (np.ndarray): Input values where to evaluate the PDF
        
    Returns:
        np.ndarray: PDF values f(r) for the chi(d) distribution
    """
    r    = np.asarray(r)
    coef = 1.0 / (2.0**(d/2 - 1) * gamma(d/2))
    out  = coef * np.where(r >= 0, (r**(d-1)) * np.exp(-0.5 * r**2), 0.0)

    return out

def chi_cdf(d: int, r: np.ndarray):
    """
    Compute the cumulative distribution function of the chi distribution.

    Args:
        d (int): Degrees of freedom parameter
        r (np.ndarray): Input values where to evaluate the CDF
        
    Returns:
        np.ndarray: CDF values F(r) = P(R <= r) for the chi(d) distribution
    """
    r = np.asarray(r)
    x = 0.5 * np.clip(r, 0, np.inf)**2

    return special.gammainc(d/2, x)  # regularized lower incomplete gamma

def chi_mass(d: int, a: float, b: float):
    """    
    Computes P(a <= R <= b) for R ~ chi(d).
    
    Args:
        d (int): Degrees of freedom parameter
        a (float): Lower bound of the interval
        b (float): Upper bound of the interval (can be +∞)
        
    Returns:
        float: Probability mass P(a <= R <= b) for the chi(d) distribution
    """
    if a < 0:
        a = 0.0
    
    Fa = float(chi_cdf(d, a))
    Fb = 1.0 if np.isinf(b) else float(chi_cdf(d, b))

    return Fb - Fa

def gauss_legendre_chi_integral(d: int, f: callable, a: float, b: float, n: int = 5):
    """
    Compute ∫_a^b f(r) p_d(r) dr using Gauss-Legendre quadrature.
    
    Args:
        d (int): Degrees of freedom parameter for the chi distribution
        f (callable): Function to integrate (should accept numpy arrays)
        a (float): Lower bound of integration
        b (float): Upper bound of integration
        n (int, optional): Number of quadrature points. Defaults to 5.
        
    Returns:
        float: Approximate value of ∫_a^b f(r) p_d(r) dr
    """
    x, w = np.polynomial.legendre.leggauss(n)  # nodes/weights on (-1,1)
    r    = 0.5 * (b - a) * x + 0.5 * (b + a)
    fr   = f(r) * chi_pdf(d, r)
    
    return 0.5 * (b - a) * float(np.dot(w, fr))

