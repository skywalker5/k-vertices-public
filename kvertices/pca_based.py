import numpy as np
from numpy.linalg import svd

def alg_2_vertices(X: np.ndarray):
    """
    Implements the Fast 2-Vertices algorithm via PCA (Algorithm 2).

    Finds the optimal k=2 solution for the k-vertices problem:
      min ||X - WH||_F^2, s.t. H>=0, 1^T H = 1^T

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
       Data matrix (features x samples).

    Returns
    -------
    W : np.ndarray, shape (m, 2)
        Matrix W with two vertices.
    H : np.ndarray, shape (2, n)
        Matrix H with soft memberships (columns sum to 1).
    eta : np.ndarray, shape (n,)
        1D representation (embedding) of data points, corresponding to the first row of H.
          eta_j = (v_max - v1_j) / (v_max - v_min)
    """
    m, n = X.shape
    if n == 0:
        # Handle empty cluster case
        return np.zeros((m, 2)), np.zeros((2, 0)), np.zeros(0)
    if n == 1:
        # Handle single point cluster case
        w = X[:, 0]
        return np.stack([w, w], axis=1), np.array([[0.5], [0.5]]), np.array([0.5])
        
    s = np.mean(X, axis=1, keepdims=True) # Keep dims for broadcasting
    X_centered = X - s

    try:
        U, S, Vt = svd(X_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        print("Warning: SVD computation failed in alg_2_vertices. Returning degenerate solution.")
        # Return a degenerate solution (e.g., centroid duplicated, uniform H)
        w_centroid = s.flatten()
        return np.stack([w_centroid, w_centroid], axis=1), np.full((2, n), 0.5), np.full(n, 0.5)

    if S.shape[0] == 0: # Handle case where SVD returns empty S (e.g., all points are identical)
        print("Warning: SVD returned empty singular values in alg_2_vertices. Returning degenerate solution.")
        w_centroid = s.flatten()
        return np.stack([w_centroid, w_centroid], axis=1), np.full((2, n), 0.5), np.full(n, 0.5)
        
    sigma1 = S[0]
    u1 = U[:, 0]
    v1 = Vt[0, :]

    v_min = np.min(v1)
    v_max = np.max(v1)

    # Construct vertices using the formula from Appendix A / Theorem 1
    w1 = s + sigma1 * v_min * u1.reshape(m, 1)
    w2 = s + sigma1 * v_max * u1.reshape(m, 1)
    W = np.hstack([w1, w2])

    # Handle potential degeneracy (all points project to the same value on PC1)
    if np.abs(v_max - v_min) < 1e-10:
        eta = 0.5 * np.ones(n) # Assign uniform membership
        # The vertices W might be identical in this case, which is fine.
    else:
        # Compute the 1D embedding eta based on v1
        # eta_j should be 1 if v1_j=v_min, and 0 if v1_j=v_max
        eta = (v_max - v1) / (v_max - v_min)

    # Construct H using eta
    H = np.vstack((eta, 1 - eta))

    # Ensure H values are clipped between 0 and 1 due to potential floating point issues
    np.clip(H, 0, 1, out=H)
    # Renormalize columns slightly if needed to ensure sum is exactly 1
    col_sums = H.sum(axis=0, keepdims=True)
    # Avoid division by zero for columns that might be all zero after clipping (unlikely but safe)
    H[:, col_sums[0,:] > 1e-9] /= col_sums[:, col_sums[0,:] > 1e-9]
    H[:, col_sums[0,:] <= 1e-9] = 0.5 # Fallback for zero-sum columns
    
    # Return W, H, and the first row of H (eta)
    return W, H, H[0, :] 