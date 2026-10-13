"""softImpute: the original algorithm of Mazumder, Hastie and Tibshirani (2010).

Baseline method of the paper (Section 1, steps 1-2):

    1. X~ = P(X) + P_perp(M)          (fill in the missing entries)
    2. M = U S_lambda(D) V^T          (soft-thresholded SVD of X~)

Repeated until convergence.  The SVD of the filled matrix is the computational
bottleneck that softImpute-ALS removes.  Here the filled matrix is never
materialized: X~ = P(X - M) + M is stored as *sparse + low rank* (Eq. 5) and
its rank-r SVD is computed with a Lanczos-type method (scipy PROPACK solver)
through a :class:`~scipy.sparse.linalg.LinearOperator`.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, svds

from .data import IncompleteMatrix
from .low_rank import SoftSVD, soft_threshold


def _filled_operators(X: IncompleteMatrix, M: SoftSVD):
    """LinearOperators for X~ = P(X - M) + M and its transpose.

    The sparse piece is the residual at the observed entries; the low-rank
    piece is M = U diag(d) V^T.
    """
    fit = np.einsum("ij,ij->i", M.u[X.rows] * M.d, M.v[X.cols])
    res = X.vals - fit
    R = sparse.csr_matrix((res, (X.rows, X.cols)), shape=X.shape)
    m, n = X.shape

    def rmat(W):  # X~ @ W
        W = np.asarray(W)
        W2 = W if W.ndim == 2 else W.reshape(-1, 1)
        out = R @ W2 + (M.u * M.d) @ (M.v.T @ W2)
        return out if W.ndim == 2 else out[:, 0]

    def lmat(W):  # X~^T @ W
        W = np.asarray(W)
        W2 = W if W.ndim == 2 else W.reshape(-1, 1)
        out = R.T @ W2 + (M.v * M.d) @ (M.u.T @ W2)
        return out if W.ndim == 2 else out[:, 0]

    return LinearOperator((m, n), matvec=rmat, rmatvec=lmat), float(res @ res)


def soft_impute(
    X: IncompleteMatrix,
    lam: float,
    rank: int = 10,
    tol: float = 1e-5,
    max_iter: int = 200,
    warm_start: SoftSVD | None = None,
    trace: list | None = None,
    time_trace: list | None = None,
) -> tuple[SoftSVD, dict]:
    """Fit the nuclear-norm-regularized matrix completion problem (1).

    minimize_M  1/2 ||P(X - M)||_F^2 + lam ||M||_*

    Returns the solution in soft-thresholded SVD form.
    """
    m, n = X.shape
    r = min(rank, min(m, n) - 1) if min(m, n) > 1 else min(rank, min(m, n))
    r = max(r, 1)

    if warm_start is not None and warm_start.d.size:
        M = warm_start
    else:
        M = SoftSVD(np.zeros((m, 0)), np.zeros(0), np.zeros((n, 0)))

    t0 = time.perf_counter()
    n_iter = max_iter
    for it in range(1, max_iter + 1):
        op, res_norm_sq = _filled_operators(X, M)
        if trace is not None:
            trace.append(0.5 * res_norm_sq + lam * float(np.sum(M.d)))
        if time_trace is not None:
            time_trace.append(time.perf_counter() - t0)

        if r < min(m, n):
            try:
                U, d, Vt = svds(op, k=r, which="LM", solver="propack", random_state=0)
                order = np.argsort(-d)
                U, d, Vt = U[:, order], d[order], Vt[order]
            except Exception:
                # Lanczos failed to converge: dense fallback on the filled matrix.
                Xd = X.to_sparse().toarray() + (~X.observed_mask()) * M.to_dense()
                U, d, Vt = np.linalg.svd(Xd, full_matrices=False)
                U, d, Vt = U[:, :r], d[:r], Vt[:r]
        else:
            # tiny problems: dense SVD of the filled matrix
            Xd = X.to_sparse().toarray() + (~X.observed_mask()) * M.to_dense()
            U, d, Vt = np.linalg.svd(Xd, full_matrices=False)
            U, d, Vt = U[:, :r], d[:r], Vt[:r]

        d = soft_threshold(d, lam)
        M_new = SoftSVD(U[:, d > 0], d[d > 0], Vt.T[:, d > 0])

        # Eq. 19-style relative change between two SVD-form iterates,
        # computed with small dense algebra only.
        d1, d2 = M.d, M_new.d
        if d1.size == 0:
            delta = float(np.sqrt(np.sum(d2**2))) if d2.size else 0.0
        elif d2.size == 0:
            delta = 1.0  # moved from M to the zero solution
        else:
            cross = float(np.sum(d1[None, :] * d2[:, None] * ((M.u.T @ M_new.u) * (M_new.v.T @ M.v))))
            num = float(np.sum(d1**2) + np.sum(d2**2) - 2.0 * cross)
            delta = np.sqrt(max(num, 0.0)) / np.sqrt(max(float(np.sum(d1**2)), 1e-12))
        M = M_new
        if delta < tol:
            n_iter = it
            break

    info = {"n_iter": n_iter, "converged": n_iter < max_iter,
            "time": time.perf_counter() - t0}
    return M, info
