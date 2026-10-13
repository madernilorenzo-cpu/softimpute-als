"""Sparse-plus-low-rank operators for the filled matrix.

Key identity from the paper (Eqs. 24-25):

    X~ = P(X) + P_perp(A B^T) = P(X - A B^T) + A B^T

The first piece is as sparse as X (it is the *residual* at the observed
entries) and the second piece is low rank.  Both storing and left/right
multiplying X~ is therefore cheap -- this is the computational heart of
softImpute-ALS.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .data import IncompleteMatrix
from .low_rank import SoftSVD, soft_threshold


class FilledMatrix:
    """X~ = P(X - AB^T) + AB^T  stored as sparse + low rank.

    Supports the only two operations the algorithms need:

    * right multiplication  X~ @ W     (W is n x k)
    * left multiplication   X~^T @ W    (W is m x k)

    plus recomputing the sparse residual after A or B changes.
    """

    def __init__(self, X: IncompleteMatrix):
        self.X = X
        self.A: np.ndarray | None = None  # m x r
        self.B: np.ndarray | None = None  # n x r
        # Sparse residual R = P(X - AB^T) in triplet form (as CSR for matmuls).
        self._res: sparse.csr_matrix = sparse.csr_matrix(X.shape)

    def set_factors(self, A: np.ndarray, B: np.ndarray) -> None:
        """Update the low-rank piece and refresh the sparse residual."""
        self.A, self.B = A, B
        fit = np.einsum("ij,ij->i", A[self.X.rows], B[self.X.cols])
        res = self.X.vals - fit
        self._res = sparse.csr_matrix(
            (res, (self.X.rows, self.X.cols)), shape=self.X.shape
        )

    def residual_norm_sq(self) -> float:
        """||P(X - AB^T)||_F^2 -- the loss term of problems (1)/(6)."""
        d = self._res.data
        return float(d @ d)

    def rmult(self, W: np.ndarray) -> np.ndarray:
        """X~ @ W  =  R @ W + A (B^T W)."""
        return self._res @ W + self.A @ (self.B.T @ W)

    def lmult(self, W: np.ndarray) -> np.ndarray:
        """X~^T @ W  =  R^T @ W + B (A^T W)."""
        return self._res.T @ W + self.B @ (self.A.T @ W)

    def soft_svd(self, lam: float) -> SoftSVD:
        """Algorithm 3.1, step 5 (final cleanup).

        The balanced SVD form of the current fit gives the right singular
        subspace V; the soft-thresholded solution is the SVD of X~ V with
        singular values passed through S_lambda.  At convergence this reveals
        the exact rank of the solution to problem (1).
        """
        if self.A.shape[1] == 0 or self.B.shape[1] == 0:
            return SoftSVD(self.A, np.zeros(0), self.B)
        Qa, Ra = np.linalg.qr(self.A)                    # A = Qa Ra
        W, d, Vt = np.linalg.svd(Ra @ self.B.T, full_matrices=False)
        V = Vt.T                                         # right vectors of A B^T
        XV = self.rmult(V)                               # M = X~ V
        Um, Dm, Vtm = np.linalg.svd(XV, full_matrices=False)
        d_soft = soft_threshold(Dm, lam)
        keep = d_soft > 1e-12 * max(1.0, float(Dm[0]) if Dm.size else 1.0)
        V_out = V @ Vtm                                  # V <- V R
        return SoftSVD(Um[:, keep], d_soft[keep], V_out[:, keep])
