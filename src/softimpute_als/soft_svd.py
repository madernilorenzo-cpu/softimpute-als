"""Algorithm 2.1 of the paper: rank-restricted soft SVD of a complete matrix.

Solves (Theorem 1)

    minimize_{Z: rank(Z) <= r}  1/2 ||X - Z||_F^2 + lambda ||Z||_*

by alternating (multiresponse) ridge regressions, maintaining the iterates in
"SVD form" A = U D, B = V D (so M = A B^T = U D^2 V^T).  Theorem 2 links this
to the factorization problem

    minimize_{A, B}  1/2 ||X - A B^T||_F^2 + lambda/2 (||A||_F^2 + ||B||_F^2).
"""

from __future__ import annotations

import numpy as np

from .low_rank import SoftSVD, soft_threshold


def soft_svd_reference(X: np.ndarray, lam: float, rank: int | None = None) -> SoftSVD:
    """Closed-form solution of Theorem 1 via a full SVD of X.

    Z = U_r S_lambda(D_r) V_r^T with U_r D_r V_r^T the (truncated) SVD of X.
    """
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    r = len(d) if rank is None else min(rank, len(d))
    d_soft = soft_threshold(d[:r], lam)
    keep = d_soft > 0
    return SoftSVD(U[:, :r][:, keep], d_soft[keep], Vt[:r].T[:, keep])


def balanced_factors_reference(X: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Theorem 2 solution: A = U_r S_lambda(D_r)^{1/2}, B = V_r S_lambda(D_r)^{1/2}."""
    S = soft_svd_reference(X, lam)
    sqrt_d = np.sqrt(S.d)
    return S.u * sqrt_d, S.v * sqrt_d


def _final_cleanup(XV: np.ndarray, V: np.ndarray, lam: float) -> SoftSVD:
    """Step 7: M = X V = U D R^T; output U, V <- V R and S_lambda(D)."""
    Um, Dm, Vtm = np.linalg.svd(XV, full_matrices=False)
    d_soft = soft_threshold(Dm, lam)
    keep = d_soft > 1e-12 * max(1.0, Dm[0] if len(Dm) else 1.0)
    return SoftSVD(Um[:, keep], d_soft[keep], (V @ Vtm)[:, keep])


def soft_svd_als(
    X: np.ndarray,
    lam: float,
    rank: int,
    tol: float = 1e-7,
    max_iter: int = 500,
    seed: int = 0,
    trace: list | None = None,
) -> tuple[SoftSVD, dict]:
    """Algorithm 2.1: alternating ridge regression for the rank-restricted soft SVD.

    2. B~^T = (D^2 + lam I)^{-1} D U^T X                       (Eq. 16)
    3. SVD of B~ D = V~ D~^2 R^T;  V <- V~, D <- D~, B = V D
    4. A~   = X V D (D^2 + lam I)^{-1}                         (Eq. 18)
    5. SVD of A~ D = U~ D~^2 R^T;  U <- U~, D <- D~, A = U D
    6. repeat 2-5 until convergence of A B^T (criterion of Eq. 19)
    7. final cleanup via :func:`_final_cleanup`
    """
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    r = min(rank, min(m, n))
    rng = np.random.default_rng(seed)

    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V = np.zeros((n, r))
    D = np.ones(r)

    n_iter = max_iter
    for it in range(1, max_iter + 1):
        U_old, D_old, V_old = U, D.copy(), V

        shrink = D / (D**2 + lam)
        Bt = shrink[:, None] * (U.T @ X)      # B~^T  (r x n)
        Bd = Bt.T * D[None, :]                # B~ D  (n x r)
        Vh, s, Rt = np.linalg.svd(Bd, full_matrices=False)
        # B~ D = V~ D~^2 R^T.  The ridge iterate is M' = U R D~^2 V~^T, so the
        # rotation R (in practice mostly sign flips) must be carried onto U
        # to keep the SVD-form representation exact.
        V = Vh
        D = np.sqrt(np.maximum(s, 0.0))       # singular values of B~D are D~^2
        U = U @ Rt.T

        shrink = D / (D**2 + lam)
        At = (X @ V) * shrink[None, :]        # A~    (m x r)
        Ad = At * D[None, :]                  # A~ D  (m x r)
        Uh, s, Wt = np.linalg.svd(Ad, full_matrices=False)
        # A~ D = U~ D~^2 R2^T;  the iterate is U~ D~^2 (V R2)^T.
        U = Uh
        D = np.sqrt(np.maximum(s, 0.0))
        V = V @ Wt.T

        if trace is not None:
            trace.append(_criterion_complete(X, U, D, V, lam))

        # Eq. 19: relative Frobenius change between M_old and M_new, both in
        # SVD form M = U D^2 V^T, computed with r x r operations only.
        d2o, d2n = D_old**2, D**2
        denom = float(np.sum(d2o**2))
        cross = float(np.trace((d2o[:, None] * (U_old.T @ U)) @ (d2n[:, None] * (V.T @ V_old))))
        num = float(np.sum(d2o**2) + np.sum(d2n**2) - 2.0 * cross)
        if denom <= 1e-300 and float(np.sum(d2n**2)) <= 1e-300:
            n_iter = it
            break
        if num <= (tol**2) * max(denom, 1e-300):
            n_iter = it
            break

    return _final_cleanup(X @ V, V, lam), {"n_iter": n_iter, "converged": n_iter < max_iter}


def _criterion_complete(X: np.ndarray, U: np.ndarray, D: np.ndarray, V: np.ndarray, lam: float) -> float:
    """Value of problem (9): 1/2 ||X - U D^2 V^T||_F^2 + lam ||U D^2 V^T||_*.

    Uses ||X - M||_F^2 = ||X||_F^2 - 2 tr(D^2 U^T X V) + tr(D^4) and
    ||M||_* = sum(D^2), all O(m n r + r) operations.
    """
    cross = float(np.sum(D**2 * np.einsum("ij,ij->j", (X.T @ U), V)))
    fro = float(np.sum(X**2)) - 2.0 * cross + float(np.sum(D**4))
    return 0.5 * fro + lam * float(np.sum(D**2))
