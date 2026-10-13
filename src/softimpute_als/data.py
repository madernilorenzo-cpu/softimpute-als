"""Core data structures for incomplete matrices.

The paper (Hastie, Mazumder, Lee and Zadeh, JMLR 2015) considers an m x n matrix X
whose entries are observed only on an index set Omega = {(i, j) : X_ij observed}.
P(X) denotes the projection that keeps observed entries and zeroes the rest, and
P_perp its complement.  Everything in this package works with the sparse
representation of P(X): an (row, col, val) triplet list plus the matrix shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class IncompleteMatrix:
    """An m x n matrix observed only on the index set Omega.

    Attributes
    ----------
    rows, cols, vals : arrays of length N = |Omega|
        Triplet representation of the observed entries.
    shape : tuple (m, n)
    """

    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        self.rows = np.asarray(self.rows, dtype=np.int64)
        self.cols = np.asarray(self.cols, dtype=np.int64)
        self.vals = np.asarray(self.vals, dtype=np.float64)
        if not (len(self.rows) == len(self.cols) == len(self.vals)):
            raise ValueError("rows, cols and vals must have the same length")
        m, n = self.shape
        if len(self.vals) and (self.rows.min() < 0 or self.rows.max() >= m
                               or self.cols.min() < 0 or self.cols.max() >= n):
            raise ValueError("indices out of bounds")

    @property
    def n_obs(self) -> int:
        return len(self.vals)

    @classmethod
    def from_dense(cls, X: np.ndarray) -> "IncompleteMatrix":
        """Build from a dense matrix that uses NaN for missing entries."""
        X = np.asarray(X, dtype=np.float64)
        obs = ~np.isnan(X)
        r, c = np.nonzero(obs)
        return cls(r, c, X[r, c], X.shape)

    def to_sparse(self) -> sparse.csr_matrix:
        """P(X): sparse m x n matrix with the observed entries (zeros elsewhere)."""
        return sparse.csr_matrix(
            (self.vals, (self.rows, self.cols)), shape=self.shape
        )

    def to_dense(self) -> np.ndarray:
        """Dense matrix with NaN at unobserved positions."""
        X = np.full(self.shape, np.nan)
        X[self.rows, self.cols] = self.vals
        return X

    def observed_mask(self) -> np.ndarray:
        M = np.zeros(self.shape, dtype=bool)
        M[self.rows, self.cols] = True
        return M

    def split(self, frac: float, seed: int = 0) -> tuple["IncompleteMatrix", "IncompleteMatrix"]:
        """Randomly split Omega into two disjoint IncompleteMatrix objects."""
        rng = np.random.default_rng(seed)
        n_val = int(round(frac * self.n_obs))
        perm = rng.permutation(self.n_obs)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train = IncompleteMatrix(
            self.rows[train_idx], self.cols[train_idx], self.vals[train_idx], self.shape
        )
        val = IncompleteMatrix(
            self.rows[val_idx], self.cols[val_idx], self.vals[val_idx], self.shape
        )
        return train, val


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """Root mean squared error between two arrays of equal length."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))
