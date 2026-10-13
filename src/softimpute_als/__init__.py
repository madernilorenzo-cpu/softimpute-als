"""softimpute-als: matrix completion and low-rank SVD via fast alternating
least squares.

A Python implementation of:

    Trevor Hastie, Rahul Mazumder, Jason D. Lee and Reza Zadeh (2015),
    "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares",
    Journal of Machine Learning Research 16, 3367-3402.

Main entry points
-----------------
- :func:`soft_impute_als` -- the paper's algorithm (Algorithms 3.1 / 5.1)
- :func:`soft_impute` -- the original softImpute baseline (Mazumder et al. 2010)
- :func:`als` -- vanilla alternating least squares / MMMF (Algorithm 5.2)
- :func:`soft_svd_als` -- rank-restricted soft SVD of a complete matrix (Algorithm 2.1)
- :func:`fit_path` / :func:`cv_lambda` -- warm-start lambda path and cross-validation
- :func:`biscale` -- row/column centering and scaling (Section 9)
"""

from .als import als
from .data import IncompleteMatrix, rmse
from .low_rank import SoftSVD, soft_threshold
from .path import cv_lambda, fit_path, lambda_max, lambda_path
from .scaling import BiScale, biscale
from .soft_impute import soft_impute
from .soft_svd import (
    balanced_factors_reference,
    soft_svd_als,
    soft_svd_reference,
)
from .softimpute_als import soft_impute_als

__version__ = "0.1.0"

__all__ = [
    "IncompleteMatrix",
    "SoftSVD",
    "BiScale",
    "als",
    "biscale",
    "cv_lambda",
    "fit_path",
    "lambda_max",
    "lambda_path",
    "rmse",
    "soft_impute",
    "soft_impute_als",
    "soft_svd_als",
    "soft_svd_reference",
    "balanced_factors_reference",
    "soft_threshold",
]
