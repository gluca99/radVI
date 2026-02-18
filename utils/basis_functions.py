import numpy as np
import sys

sys.path.insert(1,'../')

from utils.integrals import chi_trunc_moment

def ramp_function(index: int, truncation: float, mesh: float, dim: int):
    """
    Ramp function defines a ramp edge function starting at a shifted location
    along the real line.
    Let a = sqrt(dim) - truncation. Then, the ramp function is defined as:
    Ramp function = 
        - 0 for r < a
        - (r - a) / mesh for a <= r <= a + index * mesh
        - 1 for r > a + index * mesh
    
    Args:
        index (int): Index of the psi function
        truncation (float): Truncation parameter
        mesh (float): Mesh size
        dim (int): Dimension
    
    Returns:
        lambda function: Ramp function psi_k(r)
    """
    a = (np.sqrt(dim) - truncation) + index * mesh

    return lambda r: np.minimum(1.0, np.maximum(0.0, (r - a) / mesh))

def build_psi(truncation: float, mesh: float, dim: int):
    """
    Build psi basis functions, and compute the means of the psi functions.
    The kth psi basis function is defined as:
    psi_k(r) = ramp_function(k, truncation, mesh, dim)

    Args:
        truncation (float): Truncation parameter
        mesh (float): Mesh size
        dim (int): Dimension
    
    Returns:
        psi_functions (list): List of psi functions
        means (np.ndarray): Means of the psi functions
    """
    num_bins = int(2 * truncation / mesh)
    means = []

    astart = np.sqrt(dim) - truncation
    mean0 = (chi_trunc_moment(n=1, a=0, b=astart, d=dim) - astart * chi_trunc_moment(n=0, a=0, b=astart, d=dim)) / astart + chi_trunc_moment(n=0, a=astart, b=np.inf, d=dim)
    means.append(mean0)

    for j in range(num_bins):
        a = astart + j * mesh
        mean = (
            (chi_trunc_moment(n=1, a=a, b=a + mesh, d=dim) - a * chi_trunc_moment(n=0, a=a, b=a + mesh, d=dim)) / mesh
            + chi_trunc_moment(n=0, a=a + mesh, b=np.inf, d=dim))
    
        means.append(mean)

    psi_functions = []
    psi_functions.append( lambda r: np.minimum(1.0, np.maximum(0.0, r/astart)) )

    for j in range(num_bins):
        raw = ramp_function(j, truncation, mesh, dim)
        psi_functions.append(raw)

    return psi_functions, np.array(means)

def build_dpsi(truncation: float, mesh: float, dim: int):
    """
    Build derivative of psi basis functions. The derivative of the kth psi basis 
    function is defined as:
    dpsi_k(r) =
        - 0 for r < a_k and r > a_k + mesh
        - 1 / mesh for a_k <= r <= a_k + mesh

    Args:
        truncation (float): Truncation parameter
        mesh (float): Mesh size
        dim (int): Dimension
    
    Returns:
        dpsi_functions (list): List of derivative of psi functions
    """
    num_bins = int(2 * truncation / mesh)
    dpsi_functions = []
    dpsi_functions.append(lambda r: np.where(r <= (np.sqrt(dim) - truncation), 1.0 / (np.sqrt(dim) - truncation), 0.0))
    for j in range(num_bins):
        a = (np.sqrt(dim) - truncation) + j * mesh
        dpsi_functions.append(lambda r, a=a, w=mesh: np.where((r >= a) & (r <= a + w), 1.0 / w, 0.0))

    return dpsi_functions

def build_gram_matrix(truncation: float, mesh: float, dim: int, psi_function_means: np.ndarray):
    """
    Build Gram matrix Q_{ij} = ∫ (ψ_i - m_i)(ψ_j - m_j) dμ, including the base ψ_0.
    Exploits that each ψ is piecewise in {0, affine, 1} with at most two breakpoints.
    
    Args:
        truncation (float): Truncation parameter
        mesh (float): Mesh size
        dim (int): Dimension
        psi_function_means (np.ndarray): Means of the psi functions
        
    Returns:
        Q (np.ndarray): Gram matrix
        Qinv (np.ndarray): Inverse of the Gram matrix
    """
    astart = np.sqrt(dim) - truncation
    M = len(psi_function_means) #num_bins #+ 1                               # total basis including base ψ_0
    assert psi_function_means.shape[0] == M, "psi_function_means must have length 1 + num_bins"

    # helper: ∫ (c0 + c1 r)(d0 + d1 r) dμ over [a,b]
    def affine_prod_integral(c0, c1, d0, d1, a, b):
        I0 = chi_trunc_moment(n=0, a=a, b=b, d=dim)
        I1 = chi_trunc_moment(n=1, a=a, b=b, d=dim)
        I2 = chi_trunc_moment(n=2, a=a, b=b, d=dim)
        return c0*d0*I0 + (c0*d1 + c1*d0)*I1 + c1*d1*I2

    # coefficients (c0, c1) of ψ_i on an interval [lo, hi] where form is constant
    # ψ_0: 0≤r≤astart -> r/astart (c0=0, c1=1/astart); r≥astart -> 1 (1,0); (r<0 -> 0)
    # ψ_k (k≥1): r≤a -> 0; a≤r≤a+mesh -> (r-a)/mesh (c0=-a/mesh,c1=1/mesh); r≥a+mesh -> 1
    def coeffs(i, lo, hi):
        if hi <= 0:
            return (0.0, 0.0)  # no support
        if i == 0:
            # base ψ_0
            if hi <= astart:                 # entirely inside [0, astart]
                return (0.0, 1.0/astart)
            if lo >= astart:                 # entirely in tail
                return (1.0, 0.0)
            # interval crosses astart -> caller must split at breakpoints; we do split globally below
            # but if it happens, we can return None to catch mistakes
            return None
        else:
            a = astart + (i-1)*mesh
            if hi <= a:                       # entirely in head (zero)
                return (0.0, 0.0)
            if lo >= a + mesh:                # entirely in tail (one)
                return (1.0, 0.0)
            if lo >= a and hi <= a + mesh:    # entirely inside ramp
                return (-a/mesh, 1.0/mesh)
            # otherwise, interval crosses a or a+mesh -> caller must split; return None if that happens
            return None

    # Build the global breakpoint set so every subinterval has constant form for both ψ_i and ψ_j
    # Potential breakpoints: 0, astart, and for each ramp k: a_k, a_k+mesh
    breaks = [0.0, astart]
    for k in range(1, M):
        a_k = astart + (k-1)*mesh
        breaks.append(a_k)
        breaks.append(a_k + mesh)
    # add an artificial large "cap" that we'll interpret as ∞ when integrating (use np.inf)
    # sorting unique finite breakpoints
    breaks = sorted(set(b for b in breaks if np.isfinite(b)))

    # function to integrate ψ_i ψ_j by summing over subintervals [bk, bk+1] and the tail [last, ∞)
    def pair_integral(i, j):
        total = 0.0
        # finite segments
        for u, v in zip(breaks[:-1], breaks[1:]):
            if v <= u:
                continue
            ci = coeffs(i, u, v)
            cj = coeffs(j, u, v)
            # If either returns None, this interval crosses a breakpoint of that ψ;
            # but by construction of breaks, that shouldn't happen. Guard just in case:
            if ci is None or cj is None:
                # split further at the relevant local breakpoints (rare). For robustness, do a tiny epsilon split.
                mid = (u + v)/2
                ci1 = coeffs(i, u, mid); cj1 = coeffs(j, u, mid)
                ci2 = coeffs(i, mid, v);  cj2 = coeffs(j, mid, v)
                if ci1 and cj1:
                    total += affine_prod_integral(ci1[0], ci1[1], cj1[0], cj1[1], a=u, b=mid)
                if ci2 and cj2:
                    total += affine_prod_integral(ci2[0], ci2[1], cj2[0], cj2[1], a=mid, b=v)
                continue
            if (ci[0], ci[1]) == (0.0, 0.0) or (cj[0], cj[1]) == (0.0, 0.0):
                continue  # integral is zero on this segment
            total += affine_prod_integral(ci[0], ci[1], cj[0], cj[1], a=u, b=v)

        # tail segment [last_break, ∞)
        last = breaks[-1]
        # coefficients on the tail are always constants (1,0) for all ψ (including ψ_0) once beyond max(astart, a_k+mesh)
        # BUT if last < astart (can’t happen here) or some a_k+mesh is larger, we chose last = max of all breakpoints,
        # so tail is truly constant 1 for every ψ_i.
        total += chi_trunc_moment(n=0, a=last, b=np.inf, d=dim)  # ∫ 1·1 dμ

        return total

    Q = np.zeros((M, M))
    for j in range(M):
        for i in range(j+1):
            I = pair_integral(i, j) - psi_function_means[i] * psi_function_means[j]
            Q[i, j] = I
            Q[j, i] = I

    Qinv = np.linalg.pinv(Q)
    
    return Q, Qinv