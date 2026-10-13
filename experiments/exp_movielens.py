"""MovieLens 100K experiment: warm-start lambda path + held-out RMSE.

Applies the paper's methodology to real data:
  1. center rows and columns of the incomplete ratings matrix (Section 9,
     ``biscale``);
  2. fit softImpute-ALS along a decreasing lambda path with warm starts
     (Section 1) and rank growing by ``rank_step`` at each step;
  3. evaluate held-out RMSE on the u1.test split of MovieLens 100K;
  4. compare ALS / softImpute-ALS / softImpute at the best lambda.

Data: https://files.grouplens.org/datasets/movielens/ml-100k.zip
(auto-downloaded into experiments/data if missing).

Run:  python experiments/exp_movielens.py
Outputs: results/movielens_rmse_vs_lambda.png, results/movielens_summary.json
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from softimpute_als.als import als
from softimpute_als.data import IncompleteMatrix, rmse
from softimpute_als.path import fit_path, lambda_path
from softimpute_als.scaling import biscale
from softimpute_als.soft_impute import soft_impute
from softimpute_als.softimpute_als import soft_impute_als

RESULTS = ROOT / "results"
DATA_DIR = ROOT / "experiments" / "data"
RESULTS.mkdir(exist_ok=True)

N_USERS, N_ITEMS = 943, 1682


def load_ml100k() -> tuple[IncompleteMatrix, IncompleteMatrix]:
    base = DATA_DIR / "ml-100k"
    if not base.exists():
        zip_path = DATA_DIR / "ml-100k.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(
                "https://files.grouplens.org/datasets/movielens/ml-100k.zip", zip_path
            )
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)

    def read_split(fname: str) -> IncompleteMatrix:
        arr = np.loadtxt(base / fname, dtype=np.int64)
        # ratings are 1..5; use 0-based user/item ids
        return IncompleteMatrix(arr[:, 0] - 1, arr[:, 1] - 1, arr[:, 2].astype(float),
                                (N_USERS, N_ITEMS))

    return read_split("u1.base"), read_split("u1.test")


def main():
    train, test = load_ml100k()
    print(f"MovieLens 100K: {train.n_obs} train / {test.n_obs} test ratings")

    # --- Section 9: center (and scale) rows and columns ---
    scaler = biscale(train, scale=True, center=True)
    train_s = scaler.transform(train)

    # --- warm-start lambda path (Section 1) ---
    lambdas = lambda_path(train_s, n_lambda=12, lambda_min_ratio=0.005)
    path = fit_path(train_s, lambdas, rank_max=40, rank_step=6, tol=1e-5,
                    max_iter=300, verbose=True)

    test_rmse, test_rmse_unscaled = [], []
    for entry in path:
        pred_s = entry["fit"].predict(test.rows, test.cols)
        pred = scaler.inverse_transform_values(test.rows, test.cols, pred_s)
        test_rmse.append(rmse(pred, test.vals))

    test_rmse = np.asarray(test_rmse)
    best = int(np.argmin(test_rmse))
    print(f"\nbest lambda = {path[best]['lambda']:.4f}, "
          f"test RMSE = {test_rmse[best]:.4f} (rank {path[best]['fit'].rank})")

    # --- method comparison at the best lambda ---
    lam_b = path[best]["lambda"]
    rank_b = max(path[best]["fit"].rank + 5, 10)
    comparison = {}
    for method, fitfn in [
        ("ALS", als),
        ("softImpute-ALS", soft_impute_als),
        ("softImpute", soft_impute),
    ]:
        f, info = fitfn(train_s, lam_b, rank=rank_b, tol=1e-5, max_iter=300)
        pred = scaler.inverse_transform_values(test.rows, test.cols,
                                                f.predict(test.rows, test.cols))
        comparison[method] = {
            "test_rmse": rmse(pred, test.vals),
            "rank": int(f.rank),
            "n_iter": info["n_iter"],
            "seconds": info["time"],
        }
        print(f"{method:15s} test RMSE {comparison[method]['test_rmse']:.4f} "
              f"(rank {f.rank}, {info['n_iter']} iters, {info['time']:.1f}s)")

    # baseline: global mean of the training ratings
    mean_pred = np.full(len(test.vals), train.vals.mean())
    print(f"global-mean baseline test RMSE = {rmse(mean_pred, test.vals):.4f}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambdas, test_rmse, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (on biscaled ratings)")
    ax.set_ylabel("held-out RMSE (u1.test, original scale)")
    ax.set_title("softImpute-ALS on MovieLens 100K (warm-start path)")
    ax.axvline(lambdas[best], ls="--", c="gray", label=f"best lambda={lambdas[best]:.3g}")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "movielens_rmse_vs_lambda.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    with open(RESULTS / "movielens_summary.json", "w") as f:
        json.dump({
            "lambdas": lambdas.tolist(),
            "test_rmse": test_rmse.tolist(),
            "best_lambda": float(lambdas[best]),
            "best_test_rmse": float(test_rmse[best]),
            "global_mean_rmse": rmse(mean_pred, test.vals),
            "comparison_at_best_lambda": comparison,
        }, f, indent=2)
    print(f"saved {RESULTS / 'movielens_summary.json'}")


if __name__ == "__main__":
    main()
