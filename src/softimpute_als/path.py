"""Warm-start lambda paths and cross-validation.

The paper (Section 1) notes that solutions along a *path* of decreasing
lambda tend to increase in rank, and that the solution at lambda_prev > lambda
makes an excellent warm start for the next lambda, "padded with some
additional dimensions".  This module implements that strategy and a simple
K-fold / hold-out cross-validation of lambda.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import svds

from .data import IncompleteMatrix, rmse
from .low_rank import SoftSVD
from .softimpute_als import soft_impute_als


def lambda_max(X: IncompleteMatrix) -> float:
    """Largest singular value of P(X): for lambda >= lambda_max the rank-0
    solution M = 0 is optimal for problem (1)."""
    S = X.to_sparse()
    m, n = X.shape
    if min(m, n) > 1:
        try:
            return float(svds(S, k=1, which="LM", solver="propack", random_state=0)[1][0])
        except Exception:
            pass
    return float(np.linalg.norm(S.toarray(), 2))


def lambda_path(
    X: IncompleteMatrix,
    n_lambda: int = 10,
    lambda_min_ratio: float = 0.01,
    lambda_max_value: float | None = None,
) -> np.ndarray:
    """Decreasing, log-spaced path of lambda values."""
    lmax = lambda_max_value if lambda_max_value is not None else lambda_max(X)
    return np.exp(np.linspace(np.log(lmax), np.log(lambda_min_ratio * lmax), n_lambda))


def fit_path(
    X: IncompleteMatrix,
    lambdas: np.ndarray,
    rank_max: int = 20,
    rank_step: int = 5,
    tol: float = 1e-5,
    max_iter: int = 200,
    seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Fit softImpute-ALS along a decreasing lambda path with warm starts.

    At each lambda the previous solution (padded by ``rank_step`` extra
    dimensions, up to ``rank_max``) warm-starts the next fit.
    """
    results = []
    warm: tuple[np.ndarray, np.ndarray] | None = None
    rank = min(rank_step, rank_max)
    for lam in np.sort(np.asarray(lambdas))[::-1]:
        M, info = soft_impute_als(
            X, lam, rank, tol=tol, max_iter=max_iter, warm_start=warm, seed=seed
        )
        train_rmse = rmse(M.predict(X.rows, X.cols), X.vals)
        results.append({"lambda": float(lam), "fit": M, "info": info, "train_rmse": train_rmse})
        warm = M.warm_start(min(rank + rank_step, rank_max), seed=seed)
        rank = min(rank + rank_step, rank_max)
        if verbose:
            print(f"lambda={lam:.4g}  rank={M.rank:3d}  iter={info['n_iter']:3d}  "
                  f"train_rmse={train_rmse:.4f}")
    return results


def cv_lambda(
    X: IncompleteMatrix,
    val_frac: float = 0.1,
    n_lambda: int = 10,
    rank_max: int = 20,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Hold-out cross-validation of lambda along a warm-start path.

    Fits the path on the training split and evaluates the held-out RMSE,
    then refits at the best lambda on all of Omega.
    """
    train, val = X.split(val_frac, seed=seed)
    lambdas = lambda_path(train, n_lambda=n_lambda)
    path = fit_path(train, lambdas, rank_max=rank_max, seed=seed, verbose=verbose)

    scores = []
    for entry in path:
        pred = entry["fit"].predict(val.rows, val.cols)
        scores.append(rmse(pred, val.vals))
        if verbose:
            print(f"lambda={entry['lambda']:.4g}  val_rmse={scores[-1]:.4f}")

    best_idx = int(np.argmin(scores))
    best_lambda = path[best_idx]["lambda"]

    full = fit_path(X, np.array([best_lambda]), rank_max=rank_max, seed=seed)
    return {
        "lambdas": lambdas,
        "val_rmse": scores,
        "best_lambda": best_lambda,
        "best_val_rmse": scores[best_idx],
        "cv_path": path,
        "fit": full[0]["fit"],
    }
