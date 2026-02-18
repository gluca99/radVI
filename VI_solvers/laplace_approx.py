import numpy as np
from scipy.optimize import minimize
from autograd import hessian
from typing import Callable, Optional

# ---------- utilities ----------
def hvp(gradV, x, v, eps=None):
    """Central-difference Hessian-vector product using only gradV."""
    x = np.asarray(x); v = np.asarray(v)
    if eps is None:
        # scale-invariant step, robust in practice for float64
        nv = np.linalg.norm(v) or 1.0
        eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(x)) / nv
    return (gradV(x + eps * v) - gradV(x - eps * v)) / (2.0 * eps)

def hessian_full(gradV, x):
    """Assemble full Hessian via HVPs (small dim)."""
    d = x.size
    H = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = 1.0
        H[:, i] = hvp(gradV, x, e)
    # symmetrize for numerical niceness
    return 0.5 * (H + H.T)

# ---------- Laplace approximation function ----------
def laplace_from_potential(V: Callable, grad_V: Callable, method: str, x0: np.ndarray, hess_V: Optional[Callable] = None):
    """
    Compute the Laplace approximation for a density proportional to exp(-V(x)).

    Since log(exp(-V(x))) = -V(x), maximising the log-density is equivalent
    to minimising the potential function V(x).

    Args:
        V (Callable): The potential function V(x) of the density exp(-V(x))
        grad_V (Callable): The gradient of the potential function V(x)
        hess_V (Callable): The Hessian of the potential function V(x)
        method (str): The method to use for the optimiser
        x0 (np.ndarray): Initial guess for the optimiser

    Returns:
        mode (np.ndarray): The approximated mode of the density exp(-V(x))
        cov (np.ndarray): The approximated covariance matrix of the density exp(-V(x))
    """

    print("-"*50)
    print("       Fitting Laplace Approximation ")
    print("-"*50)
    print(f"• Method: {method}")
    print("-"*50 + "\n")
    
    # Find the mode by minimizing the potential function
    mode = minimize(V, x0=np.array(x0), method=method).x

    if method == 'Powell':
        print("Computing numerical approximation of the Hessian of the potential function using finite differences")
        hess = hessian_full(grad_V, mode)
        cov = np.linalg.inv(hess)
    else:
        if hess_V is not None:
            print("Using provided closed-form Hessian of the potential function")
            hess = hess_V(mode)
        else:
            print("Computing numerical approximation of the Hessian of the potential function using autograd")
            hess = hessian(V)(mode)
        cov  = np.linalg.inv(hess)

    print(f'Completed fitting Laplace approximation.\n')

    return mode.reshape(-1, 1), cov