"""Timing comparison of the three algorithms (reproduces Section 6.1 of the
paper: ALS vs softImpute-ALS vs softImpute).

Simulation settings follow the paper's figure captions:
  - (300, 200) matrix, 70% missing, rank-15 signal, rank r = 25
  - (800, 600) matrix, 90% missing, rank-31 signal, rank r = 50

For each method we plot the *relative objective*
    (F_k - F*) / F*
against wall-clock seconds, where F* is the best criterion found by a long
softImpute-ALS run.

Run:  python experiments/exp_timing.py
Outputs: results/timing_<setting>.png, results/timing_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from softimpute_als.als import als
from softimpute_als.data import IncompleteMatrix
from softimpute_als.path import lambda_max
from softimpute_als.soft_impute import soft_impute
from softimpute_als.softimpute_als import soft_impute_als

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def make_dataset(m: int, n: int, rank_signal: int, na_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((m, rank_signal)))[0]
    V = np.linalg.qr(rng.standard_normal((n, rank_signal)))[0]
    M = U @ np.diag(np.linspace(1.0, 0.3, rank_signal)) @ V.T
    mask = rng.random((m, n)) < (1 - na_frac)
    rows, cols = np.nonzero(mask)
    vals = M[rows, cols] + 0.1 * rng.standard_normal(len(rows))
    return IncompleteMatrix(rows, cols, vals, (m, n))


def run_setting(name: str, m: int, n: int, rank_signal: int, na_frac: float, r: int, seed: int):
    X = make_dataset(m, n, rank_signal, na_frac, seed=seed)
    lam = 0.3 * lambda_max(X)

    # reference optimum: long softImpute-ALS run
    fit_star, _ = soft_impute_als(X, lam, rank=r, tol=1e-10, max_iter=2000)
    pred = fit_star.predict(X.rows, X.cols)
    f_star = 0.5 * float(np.sum((X.vals - pred) ** 2)) + lam * float(np.sum(fit_star.d))

    runners = {
        "ALS": lambda tr, tt: als(X, lam, rank=r, tol=0, max_iter=50,
                                  trace=tr, time_trace=tt),
        "softImpute-ALS": lambda tr, tt: soft_impute_als(
            X, lam, rank=r, tol=0, max_iter=200, trace=tr, time_trace=tt),
        "softImpute": lambda tr, tt: soft_impute(
            X, lam, rank=r, tol=0, max_iter=50, trace=tr, time_trace=tt),
    }
    runs = {}
    for method, run in runners.items():
        tr, tt = [], []
        run(tr, tt)
        crit = np.maximum(np.asarray(tr) - f_star, 1e-16) / f_star
        runs[method] = {"t": np.asarray(tt).tolist(), "rel_obj": crit.tolist()}

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, spec in runs.items():
        ax.plot(spec["t"], spec["rel_obj"], marker="o", ms=3, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock seconds")
    ax.set_ylabel("relative objective (log scale)")
    ax.set_title(f"{name}:  m={m}, n={n}, {int(na_frac*100)}% NA, lam={lam:.1f}, r={r}")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / f"timing_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

    summary = {}
    for method, spec in runs.items():
        t = np.asarray(spec["t"])
        rel = np.asarray(spec["rel_obj"])
        per_iter = np.median(np.diff(t)) if len(t) > 1 else t[-1]
        summary[method] = {
            "n_iters": len(t),
            "seconds_per_iter": float(per_iter),
            "final_rel_obj": float(rel[-1]),
            "total_seconds": float(t[-1]),
        }
    return summary


def main():
    settings = [
        ("300x200", 300, 200, 15, 0.70, 25, 1),
        ("800x600", 800, 600, 31, 0.90, 50, 2),
    ]
    all_summaries = {}
    for name, m, n, rk, na, r, seed in settings:
        print(f"--- {name} ---")
        all_summaries[name] = run_setting(name, m, n, rk, na, r, seed)
        for method, s in all_summaries[name].items():
            print(f"  {method:15s} {s['seconds_per_iter']:.4f} s/iter, "
                  f"final rel. obj {s['final_rel_obj']:.2e} in {s['total_seconds']:.2f} s")
    with open(RESULTS / "timing_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"saved {RESULTS / 'timing_summary.json'}")


if __name__ == "__main__":
    main()
