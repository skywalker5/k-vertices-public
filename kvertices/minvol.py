"""Minimum-volume k-vertices: numpy implementation."""
import numpy as np

from .nnls import nnlsm_blockpivot


def encode(W, X, beta=1e4):
    """Simplex memberships H for data X given vertices W.

    Solves min_{H>=0} ||X - WH||_F^2 + beta ||1^T H - 1^T||^2 (soft
    sum-to-one) via block-pivot NNLS. W: (m, k), X: (m, n) -> H: (k, n).
    """
    m, n = X.shape
    _, k = W.shape
    s = np.sqrt(beta)
    X_aug = np.vstack([X, np.full((1, n), s)])
    W_aug = np.vstack([W, np.full((1, k), s)])
    H, _ = nnlsm_blockpivot(W_aug, X_aug)
    return H


def _volume_logdet(W, epsilon=1e-8):
    k = W.shape[1]
    if k == 1:
        return 0.0
    P = W[:, :-1] - W[:, [-1]]
    G = P.T @ P
    G = 0.5 * (G + G.T)
    sign, logdet = np.linalg.slogdet(G + max(epsilon, 1e-12) * np.eye(k - 1))
    return logdet if sign > 0 else 0.0


def _grad_volume_logdet(W, epsilon=1e-8):
    m, k = W.shape
    if k == 1:
        return np.zeros_like(W)
    P = W[:, :-1] - W[:, [-1]]
    G = P.T @ P
    G = 0.5 * (G + G.T)
    try:
        Ginv = np.linalg.inv(G + max(epsilon, 1e-12) * np.eye(k - 1))
    except np.linalg.LinAlgError:
        return np.zeros_like(W)
    dP = 2.0 * P @ Ginv
    grad = np.zeros_like(W)
    grad[:, :-1] = dP
    grad[:, -1] = -dP.sum(axis=1)
    return grad


def _update_W(W, H, X, lambda_, steps=80, lr=1e-2, epsilon=1e-8):
    """Adam steps on ||X - WH||_F^2 + lambda * logdet volume penalty."""
    m_adam = np.zeros_like(W)
    v_adam = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    HHt = H @ H.T
    XHt = X @ H.T
    for t in range(1, steps + 1):
        grad = 2.0 * (W @ HHt - XHt) + lambda_ * _grad_volume_logdet(W, epsilon)
        if not np.all(np.isfinite(grad)):
            break
        m_adam = b1 * m_adam + (1 - b1) * grad
        v_adam = b2 * v_adam + (1 - b2) * grad ** 2
        W = W - lr * (m_adam / (1 - b1 ** t)) / (np.sqrt(v_adam / (1 - b2 ** t)) + eps)
    return W


def refine_minvol(X, W_init, lambda_=1e-2, beta=1e4, max_outer_iter=80,
                  steps_w=80, lr_w=1e-2, epsilon=1e-8, tol=1e-4):
    """Alternating H/W optimization of the min-volume objective from W_init.

    X: (m, n), W_init: (m, k). Returns (W, H).
    """
    W = W_init.copy()
    H = encode(W, X, beta)
    prev = np.inf
    for it in range(max_outer_iter):
        W = _update_W(W, H, X, lambda_, steps=steps_w, lr=lr_w, epsilon=epsilon)
        H = encode(W, X, beta)
        obj = (np.linalg.norm(X - W @ H, 'fro') ** 2
               + lambda_ * _volume_logdet(W, epsilon)
               + beta * np.linalg.norm(H.sum(axis=0) - 1.0) ** 2)
        if np.isfinite(prev) and abs(obj - prev) / (abs(prev) + 1e-9) < tol and it > 0:
            break
        prev = obj
    return W, H


def minvol_kvs(X, k, lambda_=1e-2, beta=1e4, random_seed=0, **kwargs):
    """Min-volume k-vertices from random initialization. Returns (W, H)."""
    rng = np.random.RandomState(random_seed)
    W = rng.randn(X.shape[0], k) * 0.1
    return refine_minvol(X, W, lambda_=lambda_, beta=beta, **kwargs)
