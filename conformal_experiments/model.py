"""Data-generating processes with KNOWN conditional return distributions.

Every DGP emits returns of the form

    r_t = mu_t + sigma_t * z_t,

where ``z_t`` are i.i.d. *standardized* (zero-mean, unit-variance) Gaussian or
Student-t innovations, and ``(mu_t, sigma_t)`` are the true conditional mean and
standard deviation given the past. The true conditional quantile at level ``p``
is therefore exactly

    q_t(p) = mu_t + sigma_t * F_z^{-1}(p),

computable at every step. This is what makes coverage *exactly* measurable:
marginal coverage, regime-stratified conditional coverage, and post-break
behavior are all judged against a known ground truth.

Four DGPs:

* ``iid``    -- constant ``(mu, sigma)``; the exchangeable base case where the
                split-conformal theorem applies verbatim.
* ``ar1``    -- AR(1) conditional mean ``mu_t = mu + phi (r_{t-1} - mu)``,
                constant ``sigma`` (predictable mean, homoskedastic).
* ``garch``  -- GARCH(1,1) conditional variance
                ``sigma_t^2 = omega + a (r_{t-1}-mu)^2 + b sigma_{t-1}^2``
                with ``omega = sigma^2 (1-a-b)`` (vol clustering; stationary but
                *not* exchangeable). The true conditional sigma comes from the
                recursion itself.
* ``break``  -- iid base with an abrupt regime shift at a sampled step:
                ``sigma -> sigma * vol_mult`` and/or ``mu -> mu + mean_shift``.

All parameters are drawn by :func:`sample_config` from documented ranges and
recorded in every experiment record -- no hidden constants steer the results.
The learner never sees ``(mu_t, sigma_t)``; it only sees features built from
past returns (:func:`feature_matrix`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

DGPKind = Literal["iid", "ar1", "garch", "break"]
InnovKind = Literal["gaussian", "student_t"]

# Sampling ranges used by sample_config (documented, recorded per experiment).
SIGMA_RANGE = (0.005, 0.02)          # base per-step vol, log-uniform
MU_REL_RANGE = (-0.1, 0.1)           # |mu| <= 0.1 * sigma (small drift)
T_DOF_CHOICES = (4.0, 6.0, 10.0)     # Student-t dof when innovations are fat-tailed
P_STUDENT_T = 0.5                    # prob. of Student-t innovations
AR1_PHI_RANGE = (0.1, 0.5)           # AR(1) mean persistence
GARCH_ALPHA_RANGE = (0.05, 0.20)     # ARCH coefficient
GARCH_BETA_RANGE = (0.70, 0.92)      # GARCH coefficient (a+b capped below 1)
GARCH_PERSISTENCE_CAP = 0.97
VOL_MULT_CHOICES = (2.0, 4.0)        # post-break vol multiplier
MEAN_SHIFT_SIGMAS = (1.0, 2.0)       # post-break |mean shift| in units of sigma
BREAK_TYPE_PROBS = {"vol": 0.4, "mean": 0.2, "both": 0.4}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DGPConfig:
    """One fully-specified data-generating process (all fields are recorded)."""

    kind: DGPKind
    n_steps: int
    sigma: float                 # base per-step conditional vol
    mu: float = 0.0              # unconditional mean level
    phi: float = 0.0             # AR(1) mean coefficient (ar1 only)
    garch_alpha: float = 0.0     # ARCH coefficient (garch only)
    garch_beta: float = 0.0      # GARCH coefficient (garch only)
    break_step: int = -1         # absolute step of the regime break (break only)
    vol_mult: float = 1.0        # post-break vol multiplier (break only)
    mean_shift: float = 0.0      # post-break additive mean shift (break only)
    innov: InnovKind = "gaussian"
    t_dof: float = 0.0           # Student-t dof; 0.0 = not applicable (gaussian)
    label: str = "custom"


def innovation_ppf(cfg: DGPConfig, p: float | np.ndarray) -> float | np.ndarray:
    """Quantile of the *standardized* (unit-variance) innovation distribution."""
    if cfg.innov == "gaussian":
        return stats.norm.ppf(p)
    scale = np.sqrt((cfg.t_dof - 2.0) / cfg.t_dof)
    return stats.t.ppf(p, cfg.t_dof) * scale


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimulatedPath:
    """A simulated return path together with its true conditional moments."""

    returns: np.ndarray   # (T,)
    mu_t: np.ndarray      # (T,) true conditional mean given the past
    sigma_t: np.ndarray   # (T,) true conditional sd given the past
    cfg: DGPConfig

    def true_interval(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """The oracle central (1-alpha) conditional interval at every step."""
        lo_q = float(innovation_ppf(self.cfg, alpha / 2.0))
        hi_q = float(innovation_ppf(self.cfg, 1.0 - alpha / 2.0))
        return self.mu_t + self.sigma_t * lo_q, self.mu_t + self.sigma_t * hi_q


def _draw_innovations(cfg: DGPConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.innov == "gaussian":
        return rng.standard_normal(cfg.n_steps)
    scale = np.sqrt((cfg.t_dof - 2.0) / cfg.t_dof)
    return rng.standard_t(cfg.t_dof, size=cfg.n_steps) * scale


def simulate_path(cfg: DGPConfig, rng: np.random.Generator) -> SimulatedPath:
    """Simulate ``r_t = mu_t + sigma_t z_t`` and return the true moments."""
    t_len = cfg.n_steps
    z = _draw_innovations(cfg, rng)
    mu_t = np.full(t_len, cfg.mu)
    sigma_t = np.full(t_len, cfg.sigma)

    if cfg.kind == "iid":
        r = mu_t + sigma_t * z

    elif cfg.kind == "ar1":
        r = np.empty(t_len)
        prev = cfg.mu  # start at the unconditional mean
        for t in range(t_len):
            mu_t[t] = cfg.mu + cfg.phi * (prev - cfg.mu)
            r[t] = mu_t[t] + cfg.sigma * z[t]
            prev = r[t]

    elif cfg.kind == "garch":
        a, b = cfg.garch_alpha, cfg.garch_beta
        omega = cfg.sigma**2 * (1.0 - a - b)
        r = np.empty(t_len)
        sig2 = cfg.sigma**2  # initialize at the unconditional variance
        prev_eps2 = cfg.sigma**2
        for t in range(t_len):
            if t > 0:
                sig2 = omega + a * prev_eps2 + b * sig2
            sigma_t[t] = np.sqrt(sig2)
            r[t] = cfg.mu + sigma_t[t] * z[t]
            prev_eps2 = (r[t] - cfg.mu) ** 2

    elif cfg.kind == "break":
        if not (0 <= cfg.break_step < t_len):
            raise ValueError(f"break_step {cfg.break_step} outside [0, {t_len})")
        post = np.arange(t_len) >= cfg.break_step
        mu_t = np.where(post, cfg.mu + cfg.mean_shift, cfg.mu)
        sigma_t = np.where(post, cfg.sigma * cfg.vol_mult, cfg.sigma)
        r = mu_t + sigma_t * z

    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown DGP kind {cfg.kind!r}")

    return SimulatedPath(returns=r, mu_t=mu_t, sigma_t=sigma_t, cfg=cfg)


# --------------------------------------------------------------------------- #
# parameter sampling (everything recorded, nothing hidden)
# --------------------------------------------------------------------------- #
def sample_config(
    rng: np.random.Generator,
    kind: DGPKind,
    *,
    n_steps: int,
    break_range: tuple[int, int] | None = None,
) -> DGPConfig:
    """Draw a fully-specified DGP config from the documented ranges above.

    ``break_range`` (absolute step indices, required for ``kind='break'``) is
    where the regime break may land -- the caller places it inside the test
    segment so post-break behavior is observable.
    """
    sigma = float(np.exp(rng.uniform(*np.log(SIGMA_RANGE))))
    mu = float(rng.uniform(*MU_REL_RANGE) * sigma)
    if rng.uniform() < P_STUDENT_T:
        innov: InnovKind = "student_t"
        t_dof = float(rng.choice(T_DOF_CHOICES))
    else:
        innov, t_dof = "gaussian", 0.0
    base = dict(n_steps=n_steps, sigma=sigma, mu=mu, innov=innov, t_dof=t_dof,
                label=kind)

    if kind == "iid":
        return DGPConfig(kind="iid", **base)

    if kind == "ar1":
        phi = float(rng.uniform(*AR1_PHI_RANGE))
        return DGPConfig(kind="ar1", phi=phi, **base)

    if kind == "garch":
        a = float(rng.uniform(*GARCH_ALPHA_RANGE))
        b = float(rng.uniform(*GARCH_BETA_RANGE))
        b = min(b, GARCH_PERSISTENCE_CAP - a)  # keep a+b below the cap
        return DGPConfig(kind="garch", garch_alpha=a, garch_beta=b, **base)

    if kind == "break":
        if break_range is None:
            raise ValueError("break_range is required for kind='break'")
        break_step = int(rng.integers(break_range[0], break_range[1]))
        types = list(BREAK_TYPE_PROBS)
        btype = str(rng.choice(types, p=[BREAK_TYPE_PROBS[k] for k in types]))
        vol_mult = float(rng.choice(VOL_MULT_CHOICES)) if btype in ("vol", "both") else 1.0
        if btype in ("mean", "both"):
            mean_shift = float(rng.choice([-1.0, 1.0]) * rng.uniform(*MEAN_SHIFT_SIGMAS) * sigma)
        else:
            mean_shift = 0.0
        return DGPConfig(kind="break", break_step=break_step, vol_mult=vol_mult,
                         mean_shift=mean_shift, **base)

    raise ValueError(f"unknown DGP kind {kind!r}")


# --------------------------------------------------------------------------- #
# learner-visible features and volatility proxies
# --------------------------------------------------------------------------- #
def feature_burn_in(n_lags: int, vol_windows: tuple[int, ...]) -> int:
    """First index at which every feature is fully defined."""
    return max(n_lags, *vol_windows) + 1


def feature_matrix(
    returns: np.ndarray,
    *,
    n_lags: int = 5,
    vol_windows: tuple[int, ...] = (5, 20),
    ewma_lambda: float = 0.94,
) -> tuple[np.ndarray, list[str]]:
    """Causal feature matrix: row ``t`` uses returns up to ``t-1`` ONLY.

    Features: lagged returns ``r_{t-1}..r_{t-L}``, ``|r_{t-1}|``, rolling mean
    (shortest window), rolling stds over ``vol_windows``, and an EWMA vol proxy
    (RiskMetrics recursion with the given ``ewma_lambda``). Rows before the
    burn-in contain NaN and must not be used.
    """
    t_len = returns.size
    cols: list[np.ndarray] = []
    names: list[str] = []

    for lag in range(1, n_lags + 1):
        c = np.full(t_len, np.nan)
        c[lag:] = returns[:-lag]
        cols.append(c)
        names.append(f"ret_lag{lag}")

    abs1 = np.full(t_len, np.nan)
    abs1[1:] = np.abs(returns[:-1])
    cols.append(abs1)
    names.append("abs_ret_lag1")

    from numpy.lib.stride_tricks import sliding_window_view

    w0 = min(vol_windows)
    rmean = np.full(t_len, np.nan)
    rmean[w0:] = sliding_window_view(returns[:-1], w0).mean(axis=1)
    cols.append(rmean)
    names.append(f"roll_mean{w0}")

    for w in vol_windows:
        rstd = np.full(t_len, np.nan)
        rstd[w:] = sliding_window_view(returns[:-1], w).std(axis=1)
        cols.append(rstd)
        names.append(f"roll_std{w}")

    cols.append(ewma_sigma(returns, ewma_lambda))
    names.append("ewma_vol")
    return np.column_stack(cols), names


def ewma_sigma(returns: np.ndarray, lam: float, init_window: int = 20) -> np.ndarray:
    """EWMA (RiskMetrics-style) vol proxy; ``sigma_hat_t`` uses returns up to
    ``t-1`` only: ``sig2_t = lam sig2_{t-1} + (1-lam) r_{t-1}^2``. Initialized
    with the variance of the first ``init_window`` returns (this touches only
    the burn-in region; with ``lam <= 0.97`` the initialization is forgotten
    long before any prediction is made). Implemented as a linear filter for
    speed; identical to the scalar recursion."""
    from scipy.signal import lfilter

    t_len = returns.size
    v0 = max(float(np.var(returns[:init_window])), 1e-16)
    sig2 = np.empty(t_len)
    sig2[0] = v0
    if t_len > 1:
        u = returns[:-1] ** 2
        sig2[1:], _ = lfilter([1.0 - lam], [1.0, -lam], u, zi=np.array([lam * v0]))
    return np.sqrt(np.maximum(sig2, 1e-16))


def fit_ewma_lambda(
    train_returns: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Fit the EWMA decay by Gaussian quasi-maximum-likelihood on a grid.

    This is the 'fitted vol model' behind the parametric baseline: an
    IGARCH/RiskMetrics-style filter whose decay is chosen to maximize the
    Gaussian log-likelihood of the training returns.
    """
    if grid is None:
        grid = np.arange(0.80, 0.996, 0.01)
    best_lam, best_ll = float(grid[0]), -np.inf
    for lam in grid:
        sig2 = ewma_sigma(train_returns, float(lam)) ** 2
        ll = float(-0.5 * np.sum(np.log(sig2) + train_returns**2 / sig2))
        if ll > best_ll:
            best_ll, best_lam = ll, float(lam)
    return best_lam
