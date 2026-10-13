"""Soft-thresholding operator and the soft-SVD solution container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Elementwise soft-thresholding S_lambda(x) = sign(x) * (|x| - lam)_+."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


@dataclass
class SoftSVD:
    """A matrix in (soft-thresholded) SVD form  M = U diag(d) V^T.

    This is the output format of every algorithm in the package.  ``d`` holds
    the (already soft-thresholded) singular values, so ``rank`` is the number
    of nonzero entries of ``d``.
    """

    u: np.ndarray  # m x r, orthonormal columns
    d: np.ndarray  # r,  nonnegative, nonincreasing
    v: np.ndarray  # n x r, orthonormal columns

    def __post_init__(self) -> None:
        if self.d.size == 0:
            self.u = self.u.reshape(self.u.shape[0], 0)
            self.v = self.v.reshape(self.v.shape[0], 0)

    @property
    def rank(self) -> int:
        if self.d.size == 0:
            return 0
        thresh = 1e-10 * max(1.0, float(self.d[0]))
        return int(np.sum(self.d > thresh))

    def to_dense(self) -> np.ndarray:
        if self.d.size == 0:
            return np.zeros((self.u.shape[0], self.v.shape[0]))
        return (self.u * self.d) @ self.v.T

    def factors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (A, B) with M = A B^T (Theorem 2 balanced factorization)."""
        sqrt_d = np.sqrt(self.d)
        return self.u * sqrt_d, self.v * sqrt_d

    def predict(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Evaluate the fitted matrix at arbitrary index pairs."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if self.d.size == 0:
            return np.zeros(len(rows))
        return np.einsum("ij,ij->i", self.u[rows] * self.d, self.v[cols])

    def warm_start(self, rank: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Factors (A, B) padded with fresh directions, for warm starts.

        Following Section 1 of the paper: solutions along a path of decreasing
        lambda increase in rank, so the previous solution is padded with extra
        dimensions to warm-start the next lambda.
        """
        r_old = self.u.shape[1]
        if rank <= r_old:
            A, B = self.factors()
            return A[:, :rank], B[:, :rank]
        m, n = self.u.shape[0], self.v.shape[0]
        rng = np.random.default_rng(seed)
        A_old, B_old = self.factors()
        # Random orthonormal completions, orthogonalized against the current
        # singular subspaces.
        Ux = np.linalg.qr(np.hstack([self.u, rng.standard_normal((m, rank - r_old))]))[0]
        Vx = np.linalg.qr(np.hstack([self.v, rng.standard_normal((n, rank - r_old))]))[0]
        scale = 1e-1 * (np.mean(self.d) if self.d.size else 1.0)
        A = np.hstack([A_old, scale * Ux[:, r_old:]])
        B = np.hstack([B_old, scale * Vx[:, r_old:]])
        return A, B
