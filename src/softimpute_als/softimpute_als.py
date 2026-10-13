"""softImpute-ALS: the algorithm proposed in the paper (Algorithms 3.1 and 5.1).

Bridges nuclear-norm-regularized matrix approximation (softImpute) and
maximum-margin matrix factorization (ALS).  Instead of a soft-thresholded SVD
per iteration, each half-step is a *single multiresponse ridge regression*

    X~ = P(X) + P_perp(A B^T) = P(X - A B^T) + A B^T          (Eq. 24-25)
    A  <- X~ B (B^T B + lam I)^-1                             (Alg 5.1, step 3)
    B  <- X~^T A (A^T A + lam I)^{-1}                         (Alg 5.1, step 5)

The regression operator is the same for every row/column, so the update costs
O(|Omega| r + (m + n) r^2) -- a factor of r cheaper than vanilla ALS.  The
filled matrix X~ lives in sparse-plus-low-rank form: the sparse piece is the
residual at the observed entries, refreshed after each half-step.

At convergence the criterion of problem (6) equals that of the convex
problem (1) (for rank at least the rank of its solution), and the final
soft-thresholded SVD (Algorithm 3.1, step 5) cleans up the rank.
"""

from __future__ import annotations

import time

import numpy as np

from .data import IncompleteMatrix
from .low_rank import SoftSVD, soft_threshold
from .operators import FilledMatrix


def soft_impute_als(
    X: IncompleteMatrix,
    lam: float,
    rank: int,
    tol: float = 1e-5,
    max_iter: int = 200,
    warm_start: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = 0,
    trace: list | None = None,
    time_trace: list | None = None,
    soft_threshold_output: bool = True,
) -> tuple[SoftSVD, dict]:
    """Fit matrix completion problem (6)/(8) by softImpute-ALS.

    Parameters
    ----------
    X : observed entries in triplet form.
    lam : regularization parameter lambda >= 0 (set a small positive value,
        or rely on the built-in jitter, if a row/column has fewer than
        ``rank`` observations).
    rank : number of columns of A and B (r in the paper).
    warm_start : optional (A0, B0) factors, e.g. from a larger lambda
        (Section 1: "the iterative algorithms can use the solution at
        lambda_prev, padded with some additional dimensions").
    soft_threshold_output : apply step 5 of Algorithm 3.1 to the final
        solution (reveals the exact rank of the solution).
    """
    m, n = X.shape
    r = min(rank, min(m, n))
    rng = np.random.default_rng(seed)
    jitter = 1e-10 if lam == 0 else 0.0

    if warm_start is not None:
        A, B = (np.asarray(w, dtype=np.float64) for w in warm_start)
        if A.shape[1] != B.shape[1]:
            raise ValueError("warm-start factors must have the same rank")
    else:
        # Algorithm 3.1, step 1: A = U D with U random orthonormal, D = I;
        # B = V D with V = 0.
        A = np.linalg.qr(rng.standard_normal((m, r)))[0]
        B = np.zeros((n, r))

    filled = FilledMatrix(X)
    filled.set_factors(A, B)

    t0 = time.perf_counter()
    n_iter = max_iter
    for it in range(1, max_iter + 1):
        A_old, B_old = A, B

        # --- update B given A (Algorithm 3.1, step 2) ---
        AtA = A.T @ A
        rhs = filled.lmult(A)                      # X~^T A = R^T A + B (A^T A)
        B = rhs @ np.linalg.inv(AtA + (lam + jitter) * np.eye(r))
        filled.set_factors(A, B)                   # refresh residual R

        # --- update A given B (Algorithm 3.1, step 3) ---
        BtB = B.T @ B
        rhs = filled.rmult(B)                      # X~ B = R B + A (B^T B)
        A = rhs @ np.linalg.inv(BtB + (lam + jitter) * np.eye(r))
        filled.set_factors(A, B)

        if trace is not None:
            trace.append(_criterion(filled, A, B, lam))
        if time_trace is not None:
            time_trace.append(time.perf_counter() - t0)

        # relative Frobenius change of M = A B^T (Eq. 19 spirit), r x r only
        num = _mdiff_sq(A_old, B_old, A, B)
        denom = float(np.sum((A_old.T @ A_old) * (B_old.T @ B_old)))
        if denom <= 1e-300 and num <= 1e-300:
            n_iter = it
            break
        if np.sqrt(max(num, 0.0)) < tol * max(np.sqrt(max(denom, 0.0)), 1e-12):
            n_iter = it
            break

    if soft_threshold_output:
        out = filled.soft_svd(lam)                 # Algorithm 3.1, step 5
    else:
        out = _to_svd_raw(A, B)
    info = {
        "n_iter": n_iter,
        "converged": n_iter < max_iter,
        "time": time.perf_counter() - t0,
        "criterion": _criterion(filled, A, B, lam),
    }
    return out, info


def _criterion(filled: FilledMatrix, A: np.ndarray, B: np.ndarray, lam: float) -> float:
    """F(A, B) = 1/2 ||P(X - A B^T)||_F^2 + lam/2 (||A||_F^2 + ||B||_F^2)."""
    return (
        0.5 * filled.residual_norm_sq()
        + 0.5 * lam * (float(np.sum(A * A)) + float(np.sum(B * B)))
    )


def _mdiff_sq(A0, B0, A1, B1) -> float:
    """||A1 B1^T - A0 B0^T||_F^2 from r x r Gram matrices."""
    t1 = float(np.sum((A0.T @ A0) * (B0.T @ B0)))
    t2 = float(np.sum((A1.T @ A1) * (B1.T @ B1)))
    tc = float(np.sum((A1.T @ A0) * (B0.T @ B1)))
    return t1 + t2 - 2.0 * tc


def _to_svd_raw(A: np.ndarray, B: np.ndarray) -> SoftSVD:
    """Balanced SVD form of M = A B^T without soft-thresholding."""
    Qa, Ra = np.linalg.qr(A)
    W, d, Vt = np.linalg.svd(Ra @ B.T, full_matrices=False)
    return SoftSVD(Qa @ W, d, Vt.T)
