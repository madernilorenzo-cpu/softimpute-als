"""Tests for the lambda path, cross-validation and biscale."""

import numpy as np
import pytest

from softimpute_als.data import IncompleteMatrix, rmse
from softimpute_als.path import cv_lambda, fit_path, lambda_max, lambda_path
from softimpute_als.scaling import biscale


@pytest.fixture(scope="module")
def X_obs():
    rng = np.random.default_rng(5)
    m, n = 60, 50
    U = np.linalg.qr(rng.standard_normal((m, 4)))[0]
    V = np.linalg.qr(rng.standard_normal((n, 4)))[0]
    M = U @ np.diag([7, 4, 2, 0.5]) @ V.T
    mask = rng.random((m, n)) < 0.6
    rows, cols = np.nonzero(mask)
    vals = M[rows, cols] + 0.05 * rng.standard_normal(len(rows))
    return IncompleteMatrix(rows, cols, vals, (m, n))


def test_lambda_path_is_decreasing(X_obs):
    lams = lambda_path(X_obs, n_lambda=5)
    assert np.all(np.diff(lams) < 0)
    assert lams[0] == pytest.approx(lambda_max(X_obs), rel=1e-6)


def test_fit_path_warm_starts(X_obs):
    lams = lambda_path(X_obs, n_lambda=4)
    path = fit_path(X_obs, lams, rank_max=10, rank_step=4)
    assert len(path) == 4
    # training RMSE decreases as lambda decreases (less bias), and the rank
    # of the solution grows -- the path behaviour described in Section 1
    assert path[-1]["train_rmse"] < path[0]["train_rmse"]


def test_cv_lambda_imputes_well(X_obs):
    """CV-selected lambda should reconstruct the signal entries well."""
    res = cv_lambda(X_obs, val_frac=0.15, n_lambda=6, rank_max=12)
    pred = res["fit"].predict(X_obs.rows, X_obs.cols)
    # rank-0 predictor (mean of nothing = 0) RMSE for comparison
    base = rmse(np.zeros(len(X_obs.vals)), X_obs.vals)
    assert res["best_val_rmse"] < base
    assert rmse(pred, X_obs.vals) < base


def test_biscale_centers_rows_and_columns():
    rng = np.random.default_rng(9)
    m, n = 30, 25
    row_eff = rng.standard_normal(m) * 2
    col_eff = rng.standard_normal(n) * 1.5
    mask = rng.random((m, n)) < 0.8
    rows, cols = np.nonzero(mask)
    vals = row_eff[rows] + col_eff[cols] + rng.standard_normal(len(rows)) * 0.1
    X = IncompleteMatrix(rows, cols, vals, (m, n))

    bs = biscale(X, scale=False, center=True)
    Xt = bs.transform(X)

    # after centering, observed row and column means should be ~zero
    for i in range(m):
        sel = Xt.rows == i
        if sel.sum() > 2:
            assert abs(np.mean(Xt.vals[sel])) < 0.1
    for j in range(n):
        sel = Xt.cols == j
        if sel.sum() > 2:
            assert abs(np.mean(Xt.vals[sel])) < 0.1


def test_biscale_scales_to_unit_variance():
    rng = np.random.default_rng(13)
    m, n = 40, 30
    mask = rng.random((m, n)) < 0.85
    rows, cols = np.nonzero(mask)
    vals = rng.standard_normal(len(rows)) * 3.0
    X = IncompleteMatrix(rows, cols, vals, (m, n))
    bs = biscale(X, scale=True, center=True)
    Xt = bs.transform(X)
    var = float(np.var(Xt.vals))
    assert 0.5 < var < 2.0
