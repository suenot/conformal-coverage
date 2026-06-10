"""Sanity tests for the DGPs, the conformal machinery, and the online protocol.

Run: python -m pytest -q   (from the project root)

The tests check the *theorems* the paper leans on, against Monte-Carlo truth:
exact conditional quantiles of every DGP, split-conformal finite-sample
coverage on exchangeable data, the ACI long-run coverage guarantee under a
regime break, and the variance-reduction claim for the normalized score.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from conformal_experiments import (
    DGPConfig,
    Protocol,
    conformal_quantile,
    feature_matrix,
    innovation_ppf,
    method_names,
    run_experiment,
    sample_config,
    simulate_path,
)
from conformal_experiments.methods import aci_update

QUICK = Protocol(train_window=300, cal_window=150, test_steps=600, refit_every=200)


def _one_record(rows: list[dict], method: str) -> dict:
    matches = [r for r in rows if r["method"] == method]
    assert len(matches) == 1
    return matches[0]


# --------------------------------------------------------------------------- #
# DGP ground truth
# --------------------------------------------------------------------------- #
def test_iid_true_interval_coverage_monte_carlo():
    """The DGP's claimed true 90% interval covers 90% of iid draws (fat tails)."""
    cfg = DGPConfig(kind="iid", n_steps=200_000, sigma=0.01, mu=0.0005,
                    innov="student_t", t_dof=4.0)
    path = simulate_path(cfg, np.random.default_rng(0))
    lo, hi = path.true_interval(0.10)
    cov = np.mean((path.returns >= lo) & (path.returns <= hi))
    assert abs(cov - 0.90) < 0.005


def test_garch_recursion_exact():
    """sigma_t^2 satisfies the GARCH(1,1) recursion exactly, step by step."""
    cfg = DGPConfig(kind="garch", n_steps=500, sigma=0.01, mu=0.0002,
                    garch_alpha=0.12, garch_beta=0.80)
    path = simulate_path(cfg, np.random.default_rng(1))
    omega = cfg.sigma**2 * (1 - cfg.garch_alpha - cfg.garch_beta)
    sig2 = path.sigma_t**2
    for t in range(1, 500):
        expected = (omega + cfg.garch_alpha * (path.returns[t - 1] - cfg.mu) ** 2
                    + cfg.garch_beta * sig2[t - 1])
        assert sig2[t] == pytest.approx(expected, rel=1e-12)


def test_garch_true_interval_coverage_monte_carlo():
    """Conditional validity of the GARCH quantiles implies ~90% marginal hits."""
    cfg = DGPConfig(kind="garch", n_steps=200_000, sigma=0.01,
                    garch_alpha=0.10, garch_beta=0.85, innov="student_t", t_dof=6.0)
    path = simulate_path(cfg, np.random.default_rng(2))
    lo, hi = path.true_interval(0.10)
    cov = np.mean((path.returns >= lo) & (path.returns <= hi))
    assert abs(cov - 0.90) < 0.01


def test_ar1_conditional_mean_identity():
    cfg = DGPConfig(kind="ar1", n_steps=5_000, sigma=0.01, mu=0.001, phi=0.4)
    path = simulate_path(cfg, np.random.default_rng(3))
    expected = cfg.mu + cfg.phi * (path.returns[:-1] - cfg.mu)
    assert np.allclose(path.mu_t[1:], expected)
    assert np.allclose(path.sigma_t, cfg.sigma)


def test_break_parameters_applied():
    cfg = DGPConfig(kind="break", n_steps=1_000, sigma=0.01, mu=0.0,
                    break_step=600, vol_mult=4.0, mean_shift=0.02)
    path = simulate_path(cfg, np.random.default_rng(4))
    assert np.all(path.sigma_t[:600] == 0.01)
    assert np.all(path.sigma_t[600:] == 0.04)
    assert np.all(path.mu_t[:600] == 0.0)
    assert np.all(path.mu_t[600:] == 0.02)


def test_t_innovation_90pct_quantile_narrower_than_gaussian():
    """At the 90% level, *standardized* (unit-variance) Student-t innovations
    have a SMALLER central quantile than the Gaussian -- so a correctly-scaled
    Gaussian parametric interval over-covers (is too wide) at 90% under fat
    tails. The naive 'fat tails => Gaussian under-covers' intuition only kicks
    in at deeper tail levels."""
    cfg_t = DGPConfig(kind="iid", n_steps=10, sigma=1.0, innov="student_t", t_dof=4.0)
    q_t = innovation_ppf(cfg_t, 0.95)
    q_gauss = stats.norm.ppf(0.95)
    assert q_t < q_gauss - 0.05
    # ...while at the 0.5% tail the ordering flips (fat tails dominate)
    assert innovation_ppf(cfg_t, 0.995) > stats.norm.ppf(0.995)


def test_feature_matrix_is_causal():
    """Features at time t must not change when future returns change."""
    rng = np.random.default_rng(5)
    r = rng.standard_normal(300) * 0.01
    x1, _ = feature_matrix(r)
    r2 = r.copy()
    r2[200:] += 1.0  # corrupt the future
    x2, _ = feature_matrix(r2)
    assert np.allclose(x1[:200], x2[:200], equal_nan=True)


# --------------------------------------------------------------------------- #
# conformal machinery: the theorems
# --------------------------------------------------------------------------- #
def test_split_conformal_finite_sample_coverage():
    """The split-conformal theorem on exchangeable data: with n calibration
    scores, P(S_new <= q_hat) = ceil(0.9 (n+1))/(n+1) exactly. Monte Carlo."""
    rng = np.random.default_rng(6)
    n, reps, alpha = 80, 6_000, 0.10
    hits = 0
    for _ in range(reps):
        scores = rng.exponential(size=n)
        q = conformal_quantile(scores, alpha)
        hits += rng.exponential() <= q
    emp = hits / reps
    expected = math.ceil((1 - alpha) * (n + 1)) / (n + 1)  # = 73/81
    assert abs(emp - expected) < 0.013
    assert emp > 1 - alpha - 0.012          # the lower bound of the theorem
    assert emp < 1 - alpha + 1 / (n + 1) + 0.012  # ...and the upper bound


def test_conformal_quantile_infinite_when_rank_exceeds_n():
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.10) == np.inf
    assert np.isfinite(conformal_quantile(np.arange(100.0), 0.10))


def test_aci_update_direction():
    """A miss must LOWER alpha_t (wider next interval); a hit must raise it.
    (The source draft writes this update on q_hat with the opposite sign --
    a sign error this test pins down against the Gibbs-Candes recursion.)"""
    a = 0.10
    after_miss = aci_update(a, err=1.0, gamma=0.02, alpha_target=0.10)
    after_hit = aci_update(a, err=0.0, gamma=0.02, alpha_target=0.10)
    assert after_miss < a < after_hit


def test_online_split_conformal_iid_near_nominal():
    """On the exchangeable (iid) DGP the online split conformal pipeline sits
    near nominal coverage."""
    cfg = DGPConfig(kind="iid", n_steps=QUICK.total_steps(), sigma=0.01,
                    mu=0.0, innov="gaussian")
    rows, _, _ = run_experiment(cfg, QUICK, np.random.default_rng(7), ewma_lambda=0.94)
    cov = _one_record(rows, "split_abs")["coverage"]
    assert 0.86 <= cov <= 0.94


def test_aci_long_run_coverage_under_break():
    """The ACI theorem: |avg coverage - 0.9| <= (max(a1, 1-a1) + gamma)/(gamma T)
    for ANY data, including an abrupt regime break. gamma=0.05, T=600 gives a
    bound of 0.032; we assert it with a small numerical margin."""
    t0 = QUICK.test_start()
    cfg = DGPConfig(kind="break", n_steps=QUICK.total_steps(), sigma=0.01,
                    break_step=t0 + 250, vol_mult=4.0, mean_shift=0.01,
                    innov="student_t", t_dof=4.0)
    rows, _, _ = run_experiment(cfg, QUICK, np.random.default_rng(8), ewma_lambda=0.94)
    gamma, t_len = 0.05, QUICK.test_steps
    bound = (max(0.10, 0.90) + gamma) / (gamma * t_len)
    cov = _one_record(rows, "aci_abs_g0.05")["coverage"]
    assert abs(cov - 0.90) <= bound + 0.005
    # while plain split conformal on the same path is dented by the break
    cov_split = _one_record(rows, "split_abs")["coverage"]
    assert cov_split < cov


def test_normalized_score_reduces_vol_tercile_spread():
    """On a strongly heteroskedastic GARCH DGP, the normalized score must
    shrink the coverage spread across true-vol terciles vs the absolute score
    (averaged over seeds), and lift high-vol coverage."""
    spreads_abs, spreads_norm, hi_abs, hi_norm = [], [], [], []
    for seed in (10, 11, 12):
        cfg = DGPConfig(kind="garch", n_steps=QUICK.total_steps(), sigma=0.01,
                        garch_alpha=0.18, garch_beta=0.78, innov="gaussian")
        rows, _, _ = run_experiment(cfg, QUICK, np.random.default_rng(seed),
                                    ewma_lambda=0.94)
        spreads_abs.append(_one_record(rows, "split_abs")["vol_cov_spread"])
        spreads_norm.append(_one_record(rows, "split_norm")["vol_cov_spread"])
        hi_abs.append(_one_record(rows, "split_abs")["cov_vol_high"])
        hi_norm.append(_one_record(rows, "split_norm")["cov_vol_high"])
    assert np.mean(spreads_norm) < np.mean(spreads_abs)
    assert np.mean(hi_norm) > np.mean(hi_abs)


def test_oracle_coverage_all_kinds():
    """The oracle (true conditional quantiles) covers ~90% on every DGP."""
    rng = np.random.default_rng(13)
    t0 = QUICK.test_start()
    for kind in ("iid", "ar1", "garch", "break"):
        cfg = sample_config(rng, kind, n_steps=200_000,
                            break_range=(t0 + 180, t0 + 360))
        path = simulate_path(cfg, rng)
        lo, hi = path.true_interval(0.10)
        cov = np.mean((path.returns >= lo) & (path.returns <= hi))
        assert abs(cov - 0.90) < 0.01, kind


def test_intervals_well_formed_and_cqr_calibrated():
    """CQR/raw-QR widths are nonnegative; conformalizing the quantile regressor
    keeps iid coverage in a sane band."""
    cfg = DGPConfig(kind="iid", n_steps=QUICK.total_steps(), sigma=0.01,
                    innov="student_t", t_dof=6.0)
    rows, _, _ = run_experiment(cfg, QUICK, np.random.default_rng(14), ewma_lambda=0.94)
    for m in ("cqr", "raw_qr", "split_abs", "split_norm", "param_gauss"):
        rec = _one_record(rows, m)
        assert rec["mean_width"] >= 0.0
    assert 0.85 <= _one_record(rows, "cqr")["coverage"] <= 0.95


def test_run_experiment_deterministic():
    """Identical seeds -> bit-identical records (no hidden randomness)."""
    cfg = DGPConfig(kind="garch", n_steps=QUICK.total_steps(), sigma=0.01,
                    garch_alpha=0.10, garch_beta=0.85)
    a, _, _ = run_experiment(cfg, QUICK, np.random.default_rng(15), ewma_lambda=0.94)
    b, _, _ = run_experiment(cfg, QUICK, np.random.default_rng(15), ewma_lambda=0.94)
    for ra, rb in zip(a, b):
        assert ra["method"] == rb["method"]
        assert ra["coverage"] == rb["coverage"]
        assert ra["mean_width"] == rb["mean_width"] or (
            np.isnan(ra["mean_width"]) and np.isnan(rb["mean_width"]))


def test_records_have_expected_keys():
    cfg = DGPConfig(kind="iid", n_steps=QUICK.total_steps(), sigma=0.01)
    rows, traj, _ = run_experiment(cfg, QUICK, np.random.default_rng(16), ewma_lambda=0.94)
    assert {r["method"] for r in rows} == set(method_names(QUICK))
    assert traj is None  # trajectories only for break experiments
    rec = rows[0]
    for key in ("coverage", "mean_width", "width_vs_oracle", "cov_vol_high",
                "vol_cov_spread", "cfg_kind", "cfg_sigma", "cfg_innov",
                "ewma_lambda", "family", "frac_unbounded"):
        assert key in rec
    # iid: constant sigma -> tercile stratification is undefined by design
    assert np.isnan(rec["vol_cov_spread"])
