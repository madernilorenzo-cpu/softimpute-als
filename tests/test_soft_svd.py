"""Tests for Algorithm 2.1 (rank-restricted soft SVD) and Theorems 1-2."""

import numpy as np
import pytest

from softimpute_als.soft_svd import (
    balanced_factors_reference,
    soft_svd_als,
    soft_svd_reference,
)


@pytest.fixture
def X_dense():
    rng = np.random.default_rng(7)
    m, n = 60, 40
    U = np.linalg.qr(rng.standard_normal((m, 6)))[0]
    V = np.linalg.qr(rng.standard_normal((n, 6)))[0]
    return U @ np.diag([12, 8, 5, 2, 1, 0.3]) @ V.T


@pytest.mark.parametrize("lam,rank", [(0.5, 6), (3.0, 6), (3.0, 10), (6.0, 4)])
def test_alg21_matches_theorem1(X_dense, lam, rank):
    """Algorithm 2.1 must reproduce the soft-thresholded SVD (Theorem 1)."""
    ref = soft_svd_reference(X_dense, lam, rank)
    fit, info = soft_svd_als(X_dense, lam, rank, tol=1e-10, max_iter=2000)
    assert fit.rank == ref.rank
    np.testing.assert_allclose(np.sort(fit.d)[::-1], np.sort(ref.d)[::-1], atol=1e-6)
    np.testing.assert_allclose(fit.to_dense(), ref.to_dense(), atol=1e-5)


def test_theorem2_balanced_factors_solve_problem14(X_dense):
    """A = U S(D)^{1/2}, B = V S(D)^{1/2} must be optimal for problem (14)."""
    lam = 2.0
    A, B = balanced_factors_reference(X_dense, lam)

    def obj(A, B):
        return 0.5 * np.sum((X_dense - A @ B.T) ** 2) + lam / 2 * (
            np.sum(A**2) + np.sum(B**2)
        )

    f0 = obj(A, B)
    # local perturbations must not decrease the objective
    rng = np.random.default_rng(1)
    for _ in range(20):
        dA = rng.standard_normal(A.shape) * 1e-3
        dB = rng.standard_normal(B.shape) * 1e-3
        assert obj(A + dA, B) >= f0 - 1e-10
        assert obj(A, B + dB) >= f0 - 1e-10


def test_alg21_criterion_decreases(X_dense):
    """With the rotation carried between half-steps, Algorithm 2.1 performs
    exact majorize-minimize steps and its criterion is non-increasing."""
    trace = []
    soft_svd_als(X_dense, lam=2.0, rank=8, trace=trace, max_iter=100)
    crit = np.array(trace)
    assert crit[-1] < 0.9 * crit[0]
    assert np.all(crit[1:] <= crit[:-1] + 1e-9 * np.abs(crit[:-1]))
