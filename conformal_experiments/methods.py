"""Interval constructors at nominal level ``1 - alpha`` (default 90%).

Conformal machinery
-------------------
* :func:`conformal_quantile` -- the finite-sample-corrected empirical quantile
  ``Quantile(ceil((1-alpha)(n+1))/n)`` of calibration scores. Returns ``inf``
  when the corrected rank exceeds ``n`` (the honest 'infinite interval' case).
* :func:`conformal_quantile_sorted` -- same, given pre-sorted scores (used by
  the per-step online loop, where one sort is shared by several methods).
* :func:`aci_update` -- the Adaptive Conformal Inference recursion of
  Gibbs & Candes (2021): ``alpha_{t+1} = alpha_t + gamma (alpha - err_t)``.
  Note this updates the *level* ``alpha_t`` (the original formulation), not the
  threshold; when ``alpha_t <= 0`` the interval is all of R (always covers) and
  when ``alpha_t >= 1`` it is empty (never covers) -- exactly the conventions
  the ACI long-run coverage theorem requires. (The source blog draft writes the
  update on the threshold ``q_hat`` with a sign that *narrows* on a miss; that
  is a sign error -- see tests/analysis.)

Model fitting
-------------
* :func:`fit_predict_block` -- fits a point regressor (squared loss) and two
  quantile regressors at ``alpha/2``/``1-alpha/2`` (sklearn
  ``GradientBoostingRegressor`` with quantile loss, the CQR setup of
  Romano, Patterson & Candes 2019) on a training slice and batch-predicts a
  contiguous index range. All hyperparameters travel through
  :class:`GBRParams` and are recorded in the run metadata.

Interval families built in ``simulate.py`` from these pieces:
  split_abs   -- split conformal, absolute residual score |y - mu_hat|
  split_norm  -- split conformal, normalized score |y - mu_hat| / sigma_hat_EWMA
  cqr         -- conformalized quantile regression
  aci_*       -- ACI on top of split_abs / split_norm, gamma swept
  raw_qr      -- UNconformalized quantile regression (baseline)
  param_gauss -- Gaussian interval mu_hat +/- z_{1-alpha/2} sigma_hat from the
                 QML-fitted EWMA vol model (parametric baseline)
  oracle      -- the DGP's true conditional interval (efficiency benchmark)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# --------------------------------------------------------------------------- #
# conformal quantiles
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of calibration scores."""
    return conformal_quantile_sorted(np.sort(np.asarray(scores, dtype=float)), alpha)


def conformal_quantile_sorted(sorted_scores: np.ndarray, alpha: float) -> float:
    """As :func:`conformal_quantile`, for an ascending pre-sorted array.

    Returns ``+inf`` if the corrected rank exceeds ``n`` (always-cover) and
    ``-inf`` if ``alpha >= 1`` (empty interval). Both cases matter for ACI,
    whose effective level wanders outside (0, 1).
    """
    if alpha >= 1.0:
        return -np.inf
    n = sorted_scores.size
    if n == 0:
        return np.inf
    k = math.ceil((1.0 - alpha) * (n + 1))
    if k > n:
        return np.inf
    return float(sorted_scores[k - 1])


def aci_update(alpha_t: float, err: float, gamma: float, alpha_target: float) -> float:
    """One ACI step (Gibbs & Candes 2021): widen after a miss, tighten after a hit."""
    return alpha_t + gamma * (alpha_target - err)


# --------------------------------------------------------------------------- #
# model fitting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GBRParams:
    """Hyperparameters of the point/quantile gradient-boosting learners."""

    n_estimators: int = 100
    max_depth: int = 2
    learning_rate: float = 0.08

    def kwargs(self) -> dict:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate}


@dataclass(frozen=True)
class BlockPredictions:
    """Predictions of the three learners over a contiguous index range."""

    mu_hat: np.ndarray   # point predictions
    qlo_hat: np.ndarray  # quantile-regression alpha/2 predictions
    qhi_hat: np.ndarray  # quantile-regression 1-alpha/2 predictions


def fit_predict_block(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: slice,
    predict_idx: slice,
    *,
    alpha: float,
    params: GBRParams,
    random_state: int,
) -> BlockPredictions:
    """Fit point + quantile GBRs on ``train_idx`` and predict ``predict_idx``.

    Quantile predictions are sorted elementwise so ``qlo <= qhi`` (gradient
    boosting quantile regressors can cross; the standard fix is to sort).
    """
    xt, yt = x[train_idx], y[train_idx]
    xp = x[predict_idx]

    point = GradientBoostingRegressor(random_state=random_state, **params.kwargs())
    point.fit(xt, yt)
    mu_hat = point.predict(xp)

    qlo_m = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2.0,
                                      random_state=random_state, **params.kwargs())
    qhi_m = GradientBoostingRegressor(loss="quantile", alpha=1.0 - alpha / 2.0,
                                      random_state=random_state, **params.kwargs())
    qlo_m.fit(xt, yt)
    qhi_m.fit(xt, yt)
    qlo = qlo_m.predict(xp)
    qhi = qhi_m.predict(xp)
    lo = np.minimum(qlo, qhi)
    hi = np.maximum(qlo, qhi)
    return BlockPredictions(mu_hat=mu_hat, qlo_hat=lo, qhi_hat=hi)


# --------------------------------------------------------------------------- #
# nonconformity scores
# --------------------------------------------------------------------------- #
def abs_scores(y: np.ndarray, mu_hat: np.ndarray) -> np.ndarray:
    """Absolute-residual score |y - mu_hat| (plain split conformal)."""
    return np.abs(y - mu_hat)


def norm_scores(y: np.ndarray, mu_hat: np.ndarray, sigma_hat: np.ndarray) -> np.ndarray:
    """Normalized score |y - mu_hat| / sigma_hat (the practical vol fix)."""
    return np.abs(y - mu_hat) / np.maximum(sigma_hat, 1e-12)


def cqr_scores(y: np.ndarray, qlo_hat: np.ndarray, qhi_hat: np.ndarray) -> np.ndarray:
    """CQR score max(qlo - y, y - qhi) (Romano, Patterson & Candes 2019)."""
    return np.maximum(qlo_hat - y, y - qhi_hat)
