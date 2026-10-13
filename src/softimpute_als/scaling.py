"""Centering and scaling of an incomplete matrix (Section 9 of the paper).

Model (Eqs. 58-62):

    X~_ij = (X_ij - mu_i - nu_j) / (sigma_i * tau_j)

with row parameters (mu_i, sigma_i) and column parameters (nu_j, tau_j)
estimated by the method-of-moments algorithm: the transformed observed data
are required to have row/column means zero and variances one, and the four
estimating equations (64)-(67) are solved in alternation until all four
conditions hold simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import IncompleteMatrix


@dataclass
class BiScale:
    """Fitted centering/scaling parameters (Section 9)."""

    row_mean: np.ndarray  # mu, shape (m,)
    col_mean: np.ndarray  # nu, shape (n,)
    row_scale: np.ndarray  # sigma, shape (m,)
    col_scale: np.ndarray  # tau, shape (n,)

    def transform(self, X: IncompleteMatrix) -> IncompleteMatrix:
        vals = (X.vals - self.row_mean[X.rows] - self.col_mean[X.cols]) / (
            self.row_scale[X.rows] * self.col_scale[X.cols]
        )
        return IncompleteMatrix(X.rows, X.cols, vals, X.shape)

    def inverse_transform_values(self, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> np.ndarray:
        return vals * self.row_scale[rows] * self.col_scale[cols] + self.row_mean[rows] + self.col_mean[cols]


def biscale(
    X: IncompleteMatrix,
    scale: bool = True,
    center: bool = True,
    max_iter: int = 20,
    tol: float = 1e-5,
) -> BiScale:
    """Alternating method-of-moments estimation of (mu, nu, sigma, tau).

    Iterates the four blocks of estimating equations (64)-(67):
    row means, column means, row scales, column scales, each solved exactly
    given the other three.  Weights are inversely proportional to the scale
    parameters, exactly as in the paper.
    """
    m, n = X.shape
    rows, cols, vals = X.rows, X.cols, X.vals
    mu = np.zeros(m)
    nu = np.zeros(n)
    sigma = np.ones(m)
    tau = np.ones(n)

    # per-row / per-column observation slices (assumes sorted triplets)
    def _stats(idx: np.ndarray, size: int):
        order = np.argsort(idx, kind="stable")
        bounds = np.searchsorted(idx[order], np.arange(size + 1))
        counts = np.diff(bounds)
        return order, bounds, counts

    row_order, row_bounds, row_counts = _stats(rows, m)
    col_order, col_bounds, col_counts = _stats(cols, n)

    prev = np.inf
    for _ in range(max_iter):
        if center:
            # Eq. (64): mu_i = sum_{j in Om_i} tau_j^{-1}(X_ij - nu_j) / sum tau_j^{-1}
            w = 1.0 / np.maximum(tau[cols], 1e-4)
            num = np.bincount(rows, weights=w * (vals - nu[cols]), minlength=m)
            den = np.bincount(rows, weights=w, minlength=m)
            mu = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

            # Eq. (65): nu_j = sum_{i in Om^j} sigma_i^{-1}(X_ij - mu_i) / sum sigma_i^{-1}
            w = 1.0 / np.maximum(sigma[rows], 1e-4)
            num = np.bincount(cols, weights=w * (vals - mu[rows]), minlength=n)
            den = np.bincount(cols, weights=w, minlength=n)
            nu = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)

        if scale:
            resid = vals - mu[rows] - nu[cols]
            # Eq. (66): sigma_i^2 = (1/n_i) sum_{j in Om_i} resid^2 / tau_j^2
            # Zero-variance rows/columns (e.g. a user rating everything the
            # same) are floored to keep the weights finite.
            w = 1.0 / np.maximum(tau[cols], 1e-4) ** 2
            num = np.bincount(rows, weights=w * resid**2, minlength=m)
            var = num / np.maximum(row_counts, 1)
            sigma = np.where(row_counts > 0, np.sqrt(np.maximum(var, 1e-8)), 1.0)
            # Eq. (67): tau_j^2 = (1/m_j) sum_{i in Om^j} resid^2 / sigma_i^2
            w = 1.0 / np.maximum(sigma[rows], 1e-4) ** 2
            num = np.bincount(cols, weights=w * resid**2, minlength=n)
            var = num / np.maximum(col_counts, 1)
            tau = np.where(col_counts > 0, np.sqrt(np.maximum(var, 1e-8)), 1.0)

        # crude convergence check on the mean shift of parameters
        cur = float(np.abs(np.mean(mu) - 0.0)) + float(np.abs(np.mean(nu)))
        if center and abs(prev - cur) < tol:
            break
        prev = cur

    return BiScale(mu, nu, sigma, tau)
