"""Tests for the three matrix-completion algorithms on incomplete data.

Key claims of the paper validated here:

1. On a *complete* matrix all three methods solve the same convex problem (1)
   and must agree with the soft-thresholded SVD.
2. On *incomplete* data, softImpute (problem 1) and softImpute-ALS
   (problems 6/8) converge to the same solution when the rank is large
   enough -- the equivalence that unifies nuclear-norm and MMMF approaches.
3. softImpute-ALS decreases its criterion monotonically.
"""

import numpy as np
import pytest

from softimpute_als.als import als
from softimpute_als.data import IncompleteMatrix, rmse
from softimpute_als.soft_impute import soft_impute
from softimpute_als.softimpute_als import soft_impute_als
from softimpute_als.soft_svd import soft_svd_reference


@pytest.fixture(scope="module")
def X_obs():
    rng = np.random.default_rng(3)
    m, n = 50, 40
    U = np.linalg.qr(rng.standard_normal((m, 4)))[0]
    V = np.linalg.qr(rng.standard_normal((n, 4)))[0]
    M = U @ np.diag([9, 6, 3, 1]) @ V.T
    mask = rng.random((m, n)) < 0.55
    rows, cols = np.nonzero(mask)
    vals = M[rows, cols] + 0.05 * rng.standard_normal(len(rows))
    return IncompleteMatrix(rows, cols, vals, (m, n))


def _criterion(X, S: "SoftSVD-like", lam):
    pred = S.predict(X.rows, X.cols)
    return 0.5 * float(np.sum((X.vals - pred) ** 2)) + lam * float(np.sum(S.d))


def test_complete_matrix_all_three_agree():
    """All methods recover S_lambda(SVD(X)) when X is fully observed."""
    rng = np.random.default_rng(11)
    m, n = 40, 30
    U = np.linalg.qr(rng.standard_normal((m, 4)))[0]
    V = np.linalg.qr(rng.standard_normal((n, 4)))[0]
    Xd = U @ np.diag([8, 5, 2, 0.5]) @ V.T
    X = IncompleteMatrix(*np.nonzero(np.ones((m, n))), Xd.ravel().copy(), (m, n))
    lam = 1.0

    ref = soft_svd_reference(Xd, lam)

    fit_si, _ = soft_impute(X, lam, rank=10, tol=1e-9, max_iter=500)
    fit_sals, _ = soft_impute_als(X, lam, rank=10, tol=1e-10, max_iter=1000)
    fit_als, _ = als(X, lam, 10, tol=1e-8, max_iter=500)

    np.testing.assert_allclose(np.sort(fit_si.d)[::-1], np.sort(ref.d)[::-1], atol=1e-4)
    np.testing.assert_allclose(np.sort(fit_sals.d)[::-1], np.sort(ref.d)[::-1], atol=1e-3)
    np.testing.assert_allclose(np.sort(fit_als.d)[::-1], np.sort(ref.d)[::-1], atol=5e-3)
    np.testing.assert_allclose(fit_sals.to_dense(), ref.to_dense(), atol=1e-3)


def test_softimpute_als_matches_softimpute_on_incomplete(X_obs):
    """Fixed-point equivalence: with enough rank, ALS output ~ softImpute."""
    lam = 3.0
    fit_si, _ = soft_impute(X_obs, lam, rank=15, tol=1e-7, max_iter=300)
    fit_sals, _ = soft_impute_als(X_obs, lam, rank=15, tol=1e-7, max_iter=600)

    np.testing.assert_allclose(
        np.sort(fit_si.d)[::-1], np.sort(fit_sals.d)[::-1], atol=5e-2
    )
    pred_si = fit_si.predict(X_obs.rows, X_obs.cols)
    pred_als = fit_sals.predict(X_obs.rows, X_obs.cols)
    # both criteria near the common optimum
    assert _criterion(X_obs, fit_si, lam) == pytest.approx(
        _criterion(X_obs, fit_sals, lam), rel=5e-3
    )
    assert rmse(pred_si, pred_als) < 0.1


def test_als_vanilla_reaches_comparable_objective(X_obs):
    """Vanilla ALS (Alg 5.2) and softImpute-ALS solve the same problem (6)."""
    lam = 2.0
    fit_v, info_v = als(X_obs, lam, rank=12, tol=1e-6, max_iter=300)
    fit_s, info_s = soft_impute_als(X_obs, lam, rank=12, tol=1e-6, max_iter=600)

    def crit(S):
        pred = S.predict(X_obs.rows, X_obs.cols)
        return 0.5 * float(np.sum((X_obs.vals - pred) ** 2)) + lam * float(np.sum(S.d))

    assert crit(fit_v) <= crit(fit_s) * (1 + 0.02)
    # softImpute-ALS should converge in fewer or comparable wall time here
    assert info_s["converged"] and info_v["converged"]


def test_criterion_monotone_softimpute_als(X_obs):
    trace = []
    soft_impute_als(X_obs, lam=2.0, rank=10, max_iter=60, trace=trace)
    crit = np.array(trace)
    # descent property (Section 5): majorize-minimize never increases F
    assert np.all(crit[1:] <= crit[:-1] + 1e-9)


def test_rank_zero_above_lambda_max(X_obs):
    """For lambda >= largest singular value of P(X), the solution is M = 0."""
    from softimpute_als.path import lambda_max

    lam = lambda_max(X_obs) * 1.01
    fit, _ = soft_impute_als(X_obs, lam, rank=10, tol=1e-7, max_iter=200)
    assert fit.rank == 0
    assert np.allclose(fit.predict(X_obs.rows, X_obs.cols), 0.0)


def test_warm_start_accelerates(X_obs):
    """Warm start from an adjacent lambda must need fewer iterations.

    Warm starting pays off along a gradual path of decreasing lambda, which is
    how :func:`softimpute_als.fit_path` uses it.
    """
    lam1, lam2 = 5.0, 4.0
    _, info_cold = soft_impute_als(X_obs, lam2, rank=12, tol=1e-6, max_iter=400)
    fit1, _ = soft_impute_als(X_obs, lam1, rank=12, tol=1e-6, max_iter=400)
    warm = fit1.warm_start(12)
    _, info_warm = soft_impute_als(
        X_obs, lam2, rank=12, tol=1e-6, max_iter=400, warm_start=warm
    )
    assert info_warm["n_iter"] < info_cold["n_iter"]
