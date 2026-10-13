"""Vanilla Alternating Least Squares (Algorithm 5.2; Rennie and Srebro 2005).

Solves the maximum-margin matrix factorization (MMMF) problem (6):

    minimize_{A, B}  1/2 ||P(X - A B^T)||_F^2 + lam/2 (||A||_F^2 + ||B||_F^2)

by block coordinate descent: with B fixed, the problem for A decouples into m
separate ridge regressions (one per row of X, using only the observed entries
of that row), and symmetrically for B.  Each update costs O(|Omega| r^2),
roughly r times the per-iteration cost of softImpute-ALS -- this is the
comparison baseline of Section 6.1 of the paper.
"""

from __future__ import annotations

import time

import numpy as np

from .data import IncompleteMatrix
from .operators import FilledMatrix


def _grouped_indices(idx: np.ndarray, size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-group observation positions: for each group, positions in the
    triplet arrays and the *other* index of each observation."""
    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    bounds = np.searchsorted(idx_sorted, np.arange(size + 1))
    return order, bounds


def als(
    X: IncompleteMatrix,
    lam: float,
    rank: int,
    tol: float = 1e-5,
    max_iter: int = 100,
    seed: int = 0,
    jitter: float = 1e-10,
    trace: list | None = None,
    time_trace: list | None = None,
) -> tuple[SoftSVD, dict]:
    """Vanilla ALS for problem (6), with per-row / per-column ridge updates.

    ``jitter`` is added to ``lam`` to keep the r x r normal equations
    invertible when a row (column) observes fewer than r entries.
    """
    m, n = X.shape
    r = min(rank, min(m, n))
    rng = np.random.default_rng(seed)

    # Both factors start nonzero (orthonormal), otherwise the very first
    # row sweep would map A to zero and the iteration would collapse.
    A = np.linalg.qr(rng.standard_normal((m, r)))[0]
    B = np.linalg.qr(rng.standard_normal((n, r)))[0]

    rows, cols, vals = X.rows, X.cols, X.vals
    row_order, row_bounds = _grouped_indices(rows, m)
    col_order, col_bounds = _grouped_indices(cols, n)
    lam_eff = lam + jitter

    t0 = time.perf_counter()
    n_iter = max_iter
    for it in range(1, max_iter + 1):
        A_old, B_old = A.copy(), B.copy()  # updates below are in-place

        # --- update rows of A:  A_i = (sum_{j in Om_i} x_ij B_j)(sum B_j B_j^T + lam I)^-1
        for i in range(m):
            pos = row_order[row_bounds[i]:row_bounds[i + 1]]
            if pos.size == 0:
                A[i] = 0.0
                continue
            Bj = B[cols[pos]]
            G = Bj.T @ Bj + lam_eff * np.eye(r)
            A[i] = np.linalg.solve(G, vals[pos] @ Bj)

        # --- update columns of B (symmetric) ---
        for j in range(n):
            pos = col_order[col_bounds[j]:col_bounds[j + 1]]
            if pos.size == 0:
                B[j] = 0.0
                continue
            Ai = A[rows[pos]]
            G = Ai.T @ Ai + lam_eff * np.eye(r)
            B[j] = np.linalg.solve(G, vals[pos] @ Ai)

        if trace is not None:
            fit = np.einsum("ij,ij->i", A[rows], B[cols])
            res = vals - fit
            trace.append(0.5 * float(res @ res) + lam * (float(np.sum(A**2)) + float(np.sum(B**2))) / 2)
        if time_trace is not None:
            time_trace.append(time.perf_counter() - t0)

        # relative Frobenius change of M = A B^T, via r x r Grams
        def _mdiff(A0, B0, A1, B1):
            t1 = float(np.sum((A0.T @ A0) * (B0.T @ B0)))
            t2 = float(np.sum((A1.T @ A1) * (B1.T @ B1)))
            tc = float(np.sum((A1.T @ A0) * (B0.T @ B1)))
            return t1 + t2 - 2.0 * tc

        denom = float(np.sum((A_old.T @ A_old) * (B_old.T @ B_old)))
        num = _mdiff(A_old, B_old, A, B)
        if denom <= 1e-300 and num <= 1e-300:
            n_iter = it
            break
        if np.sqrt(max(num, 0.0)) < tol * max(np.sqrt(max(denom, 0.0)), 1e-12):
            n_iter = it
            break

    # Final cleanup, same as Algorithm 3.1 step 5: soft-thresholded SVD of
    # X~ V for the current factors (reveals the rank of the solution to the
    # equivalent convex problem (1)).
    filled = FilledMatrix(X)
    filled.set_factors(A, B)
    out = filled.soft_svd(lam)

    info = {"n_iter": n_iter, "converged": n_iter < max_iter,
            "time": time.perf_counter() - t0}
    return out, info
