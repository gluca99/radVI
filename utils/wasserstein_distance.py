import numpy as np


def w2_squared_radial_from_points(X: np.ndarray, Y: np.ndarray):
    """
    Squared 2-Wasserstein distance between two radial distributions in R^d,
    given d-dimensional samples X ~ P and Y ~ Q.

    This returns the exact W2^2 between the empirical radial laws
    μ_n = (1/n) Σ δ_{||X_i||}, ν_m = (1/m) Σ δ_{||Y_j||},
    which equals the W2^2 between P and Q in the radial setting.

    Args:
        X (np.ndarray): Samples from the target distribution of shape (d, n)
        Y (np.ndarray): Samples from the pushforward distribution of shape (d, m)

    Returns:
        float: W2^2(μ_n, ν_m).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D arrays of shape (d, n) and (d, m).")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same ambient dimension d.")
        
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError("Both sample sets must be non-empty.")

    # Radii (empirical radial measures)
    R = np.linalg.norm(X, axis=0)
    S = np.linalg.norm(Y, axis=0)
    R.sort(); S.sort()
    n, m = R.size, S.size

    # Fast path when sizes match: pair order statistics
    if n == m:
        return float(np.mean((R - S) ** 2))