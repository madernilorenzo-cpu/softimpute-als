import numpy as np
import pytest

from softimpute_als.data import IncompleteMatrix
from softimpute_als.low_rank import SoftSVD
from softimpute_als.operators import FilledMatrix


@pytest.fixture
def X_incomplete():
    rng = np.random.default_rng(42)
    m, n = 40, 30
    U = np.linalg.qr(rng.standard_normal((m, 5)))[0]
    V = np.linalg.qr(rng.standard_normal((n, 5)))[0]
    M = U @ np.diag([10, 6, 3, 1, 0.5]) @ V.T
    mask = rng.random((m, n)) < 0.5
    rows, cols = np.nonzero(mask)
    vals = M[rows, cols] + 0.1 * rng.standard_normal(len(rows))
    return IncompleteMatrix(rows, cols, vals, (m, n)), M


def test_filled_matrix_matches_dense(X_incomplete):
    X, M = X_incomplete
    rng = np.random.default_rng(0)
    A = rng.standard_normal((X.shape[0], 3))
    B = rng.standard_normal((X.shape[1], 3))
    filled = FilledMatrix(X)
    filled.set_factors(A, B)

    # dense reference: X~ = P(X) + P_perp(AB^T)
    Xd = np.zeros(X.shape)
    Xd[X.rows, X.cols] = X.vals
    Xd_dense = Xd + (~X.observed_mask()) * (A @ B.T)

    W = rng.standard_normal((X.shape[1], 4))
    np.testing.assert_allclose(filled.rmult(W), Xd_dense @ W, atol=1e-10)
    W = rng.standard_normal((X.shape[0], 4))
    np.testing.assert_allclose(filled.lmult(W), Xd_dense.T @ W, atol=1e-10)

    # residual norm matches the loss term
    res = Xd_dense - Xd  # = -P_perp... careful: loss is at observed entries
    obs = X.observed_mask()
    expected = np.sum((Xd_dense[obs] - Xd[obs]) ** 2) * 0  # Xd_dense==Xd on Omega
    # the loss is ||P(X - AB^T)||_F^2:
    loss = np.sum((X.vals - np.einsum("ij,ij->i", A[X.rows], B[X.cols])) ** 2)
    assert filled.residual_norm_sq() == pytest.approx(loss, rel=1e-10)


def test_soft_svd_predict_shape(X_incomplete):
    X, _ = X_incomplete
    S = SoftSVD(np.eye(X.shape[0])[:, :2], np.array([3.0, 1.0]), np.eye(X.shape[1])[:, :2])
    pred = S.predict(X.rows[:5], X.cols[:5])
    assert pred.shape == (5,)
