"""Online evaluation protocol: fit/calibrate on rolling windows, predict one
step ahead, score against the KNOWN truth.

For each experiment we (1) draw a fully-recorded DGP config, (2) simulate one
path, (3) walk forward through a test segment producing a one-step-ahead
interval from every method at nominal ``1 - alpha``, and (4) record per-method
coverage/width plus the regime state (true-vol tercile, pre/post break).

Protocol details (all in :class:`Protocol`, recorded in run metadata):

* The learners are refit every ``refit_every`` steps on the most recent
  ``train_window`` points; the calibration window is the most recent
  ``cal_window`` points, re-scored under the model currently in force and slid
  forward every step (newly observed outcomes enter the window immediately).
* ACI states update every step from the realized cover/miss of *their own*
  interval, as in Gibbs & Candes (2021).
* The point/quantile learners only ever see causal features
  (:func:`conformal_experiments.model.feature_matrix`); the true conditional
  moments are used exclusively for scoring and for the oracle benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

from .methods import (
    GBRParams,
    abs_scores,
    aci_update,
    conformal_quantile_sorted,
    cqr_scores,
    fit_predict_block,
    norm_scores,
)
from .model import (
    DGPConfig,
    DGPKind,
    ewma_sigma,
    feature_burn_in,
    feature_matrix,
    fit_ewma_lambda,
    sample_config,
    simulate_path,
)

# Documented batch-level constants (recorded in run metadata by run_all.py).
EWMA_LAMBDA_RANGE = (0.90, 0.97)   # sampled per experiment for the normalized score
BREAK_POS_RANGE = (0.30, 0.60)     # break lands in this fraction of the test segment
TRAJ_PRE, TRAJ_POST = 300, 600     # event-time range kept for break trajectories
POST_WINDOWS = ((0, 60), (60, 150), (150, 300), (300, 600))  # post-break windows


@dataclass(frozen=True)
class Protocol:
    """Everything about the online evaluation that is not the DGP."""

    train_window: int = 600
    cal_window: int = 250
    test_steps: int = 1500
    refit_every: int = 250
    alpha: float = 0.10
    n_lags: int = 5
    vol_windows: tuple[int, ...] = (5, 20)
    aci_gammas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    gbr: GBRParams = GBRParams()
    rolling_window: int = 60       # window for post-break rolling coverage
    recovery_tol: float = 0.03     # 'recovered' when rolling cov >= 1-alpha-tol

    def burn_in(self) -> int:
        return feature_burn_in(self.n_lags, self.vol_windows)

    def total_steps(self) -> int:
        return self.burn_in() + self.train_window + self.cal_window + self.test_steps

    def test_start(self) -> int:
        return self.burn_in() + self.train_window + self.cal_window


def method_names(proto: Protocol) -> list[str]:
    base = ["oracle", "split_abs", "split_norm", "cqr", "raw_qr", "param_gauss"]
    for g in proto.aci_gammas:
        base.append(f"aci_abs_g{g:g}")
        base.append(f"aci_norm_g{g:g}")
    return base


def _method_family(name: str) -> str:
    if name == "oracle":
        return "oracle"
    if name.startswith("aci_"):
        return "aci"
    if name in ("raw_qr", "param_gauss"):
        return "baseline"
    return "conformal"


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if x.size < w:
        return np.empty(0)
    return np.convolve(x.astype(float), np.ones(w) / w, mode="valid")


# --------------------------------------------------------------------------- #
# one experiment
# --------------------------------------------------------------------------- #
def run_experiment(
    cfg: DGPConfig,
    proto: Protocol,
    rng: np.random.Generator,
    *,
    ewma_lambda: float,
    exp_id: int = 0,
    return_steps: bool = False,
) -> tuple[list[dict], dict | None, dict | None]:
    """Run the full online protocol on one simulated path.

    Returns ``(rows, traj, steps)``: one record per method, an event-time
    trajectory dict for break experiments (else ``None``), and -- only when
    ``return_steps`` -- the per-step intervals for figure traces.
    """
    if cfg.n_steps != proto.total_steps():
        raise ValueError("cfg.n_steps must equal proto.total_steps()")
    path = simulate_path(cfg, rng)
    y = path.returns
    x, _ = feature_matrix(y, n_lags=proto.n_lags, vol_windows=proto.vol_windows,
                          ewma_lambda=ewma_lambda)
    sig_ewma = ewma_sigma(y, ewma_lambda)
    oracle_lo, oracle_hi = path.true_interval(proto.alpha)

    alpha = proto.alpha
    cal, train = proto.cal_window, proto.train_window
    t0, t1 = proto.test_start(), proto.total_steps()
    n_test = proto.test_steps
    z_param = float(stats.norm.ppf(1.0 - alpha / 2.0))

    names = method_names(proto)
    covered = {m: np.zeros(n_test, dtype=bool) for m in names}
    width = {m: np.full(n_test, np.nan) for m in names}
    steps: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    if return_steps:
        steps = {m: (np.full(n_test, np.nan), np.full(n_test, np.nan)) for m in names}

    aci_specs = [(f"aci_{stype}_g{g:g}", stype, g)
                 for g in proto.aci_gammas for stype in ("abs", "norm")]
    aci_alpha = {name: alpha for name, _, _ in aci_specs}

    for block_start in range(t0, t1, proto.refit_every):
        block_end = min(block_start + proto.refit_every, t1)
        fit_lo, fit_hi = block_start - cal - train, block_start - cal
        rs = int(rng.integers(0, 2**31 - 1))
        pred = fit_predict_block(x, y, slice(fit_lo, fit_hi), slice(fit_hi, block_end),
                                 alpha=alpha, params=proto.gbr, random_state=rs)
        lam_param = fit_ewma_lambda(y[fit_lo:fit_hi])
        sig_param = ewma_sigma(y, lam_param)

        # scores over [cal window + block] under the model currently in force;
        # window slicing below enforces causality (a point's score is only used
        # at steps strictly after it is observed).
        y_pr = y[fit_hi:block_end]
        s_abs = abs_scores(y_pr, pred.mu_hat)
        s_norm = norm_scores(y_pr, pred.mu_hat, sig_ewma[fit_hi:block_end])
        s_cqr = cqr_scores(y_pr, pred.qlo_hat, pred.qhi_hat)

        for t in range(block_start, block_end):
            j = t - fit_hi          # index into the block arrays (j >= cal)
            i = t - t0              # test index
            ya = y[t]
            mu, qlo, qhi = pred.mu_hat[j], pred.qlo_hat[j], pred.qhi_hat[j]
            se, sp = sig_ewma[t], sig_param[t]
            sorted_abs = np.sort(s_abs[j - cal:j])
            sorted_norm = np.sort(s_norm[j - cal:j])
            sorted_cqr = np.sort(s_cqr[j - cal:j])

            def rec(name: str, lo: float, hi: float) -> None:
                covered[name][i] = bool(lo <= ya <= hi)
                w = hi - lo
                width[name][i] = w if np.isfinite(w) else np.nan
                if steps is not None:
                    steps[name][0][i] = lo
                    steps[name][1][i] = hi

            q = conformal_quantile_sorted(sorted_abs, alpha)
            rec("split_abs", mu - q, mu + q)
            q = conformal_quantile_sorted(sorted_norm, alpha)
            rec("split_norm", mu - q * se, mu + q * se)
            q = conformal_quantile_sorted(sorted_cqr, alpha)
            rec("cqr", qlo - q, qhi + q)
            rec("raw_qr", qlo, qhi)
            rec("param_gauss", mu - z_param * sp, mu + z_param * sp)
            rec("oracle", oracle_lo[t], oracle_hi[t])

            for name, stype, g in aci_specs:
                a_t = aci_alpha[name]
                sorted_w = sorted_abs if stype == "abs" else sorted_norm
                scale = 1.0 if stype == "abs" else se
                q = conformal_quantile_sorted(sorted_w, a_t)
                if q == -np.inf:                # alpha_t >= 1: empty interval
                    lo, hi = np.nan, np.nan
                    cov = False
                    width[name][i] = 0.0
                elif not np.isfinite(q):        # always-cover interval
                    lo, hi = -np.inf, np.inf
                    cov = True
                    width[name][i] = np.nan
                else:
                    lo, hi = mu - q * scale, mu + q * scale
                    cov = bool(lo <= ya <= hi)
                    width[name][i] = hi - lo
                covered[name][i] = cov
                if steps is not None:
                    steps[name][0][i] = lo
                    steps[name][1][i] = hi
                aci_alpha[name] = aci_update(a_t, 0.0 if cov else 1.0, g, alpha)

    # ---------------- aggregate per-method records ------------------------- #
    sigma_test = path.sigma_t[t0:t1]
    oracle_w = oracle_hi[t0:t1] - oracle_lo[t0:t1]
    break_rel = cfg.break_step - t0 if cfg.kind == "break" else -1

    base = {f"cfg_{k}": v for k, v in asdict(cfg).items()}
    base.update(exp_id=exp_id, ewma_lambda=ewma_lambda, n_test=n_test)

    rows = []
    for name in names:
        g = None
        for aname, _, gamma in aci_specs:
            if aname == name:
                g = gamma
        rows.append({
            **base,
            "method": name,
            "family": _method_family(name),
            "aci_gamma": np.nan if g is None else g,
            **_method_metrics(covered[name], width[name], oracle_w, sigma_test,
                              cfg.kind, break_rel, proto),
        })

    traj = None
    if cfg.kind == "break":
        traj = {"break_rel": break_rel, "covered": {m: covered[m] for m in names}}
    return rows, traj, steps


def _method_metrics(
    cov: np.ndarray,
    wid: np.ndarray,
    oracle_w: np.ndarray,
    sigma_test: np.ndarray,
    kind: DGPKind,
    break_rel: int,
    proto: Protocol,
) -> dict:
    """Marginal + regime-stratified coverage/width metrics for one method."""
    out: dict = {
        "coverage": float(cov.mean()),
        "mean_width": float(np.nanmean(wid)),
        "median_width": float(np.nanmedian(wid)),
        "width_vs_oracle": float(np.nanmean(wid / oracle_w)),
        "frac_unbounded": float(np.mean(np.isnan(wid))),
    }

    # conditional coverage by TRUE-vol tercile (only when sigma_t varies)
    keys = ("low", "mid", "high")
    if float(sigma_test.std()) > 1e-15:
        edges = np.quantile(sigma_test, [1.0 / 3.0, 2.0 / 3.0])
        terc = np.digitize(sigma_test, edges)  # 0 / 1 / 2
        covs = []
        for k, lbl in enumerate(keys):
            m = terc == k
            c = float(cov[m].mean()) if m.any() else np.nan
            out[f"cov_vol_{lbl}"] = c
            out[f"width_ratio_vol_{lbl}"] = float(np.nanmean(wid[m] / oracle_w[m])) if m.any() else np.nan
            covs.append(c)
        out["vol_cov_spread"] = float(np.nanmax(covs) - np.nanmin(covs))
    else:
        for lbl in keys:
            out[f"cov_vol_{lbl}"] = np.nan
            out[f"width_ratio_vol_{lbl}"] = np.nan
        out["vol_cov_spread"] = np.nan

    # post-break behavior
    if kind == "break" and 0 <= break_rel < cov.size:
        pre, post = cov[:break_rel], cov[break_rel:]
        out["cov_pre_break"] = float(pre.mean()) if pre.size else np.nan
        out["cov_post_break"] = float(post.mean()) if post.size else np.nan
        for a, b in POST_WINDOWS:
            seg = post[a:b]
            out[f"cov_post_{a}_{b}"] = float(seg.mean()) if seg.size else np.nan
        roll = _rolling_mean(post, proto.rolling_window)
        if roll.size:
            out["hole_depth"] = float(roll.min())
            ok = roll >= (1.0 - proto.alpha - proto.recovery_tol)
            if ok.any():
                out["recovery_steps"] = float(int(np.argmax(ok)) + proto.rolling_window)
                out["recovered"] = 1.0
            else:
                out["recovery_steps"] = np.nan
                out["recovered"] = 0.0
        else:
            out["hole_depth"] = np.nan
            out["recovery_steps"] = np.nan
            out["recovered"] = np.nan
    else:
        out.update(cov_pre_break=np.nan, cov_post_break=np.nan, hole_depth=np.nan,
                   recovery_steps=np.nan, recovered=np.nan)
        for a, b in POST_WINDOWS:
            out[f"cov_post_{a}_{b}"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# event-time trajectory accumulator (break experiments)
# --------------------------------------------------------------------------- #
class TrajectoryAccumulator:
    """Aggregates covered/missed flags in event time around the break."""

    def __init__(self, proto: Protocol) -> None:
        self.rel = np.arange(-TRAJ_PRE, TRAJ_POST)
        self.names = method_names(proto)
        self.sums = {m: np.zeros(self.rel.size) for m in self.names}
        self.counts = np.zeros(self.rel.size)
        self.n_experiments = 0
        self.test_steps = proto.test_steps

    def add(self, traj: dict) -> None:
        br = traj["break_rel"]
        lo_rel = max(-TRAJ_PRE, -br)
        hi_rel = min(TRAJ_POST, self.test_steps - br)
        if hi_rel <= lo_rel:
            return
        rel_idx = np.arange(lo_rel, hi_rel)
        arr_idx = rel_idx + TRAJ_PRE
        t_idx = rel_idx + br
        for m in self.names:
            self.sums[m][arr_idx] += traj["covered"][m][t_idx]
        self.counts[arr_idx] += 1
        self.n_experiments += 1

    def result(self) -> dict:
        cov = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            for m in self.names:
                c = self.sums[m] / self.counts
                cov[m] = [None if not np.isfinite(v) else float(v) for v in c]
        return {
            "rel_time": [int(v) for v in self.rel],
            "n_experiments": self.n_experiments,
            "coverage": cov,
        }


# --------------------------------------------------------------------------- #
# Monte-Carlo batches
# --------------------------------------------------------------------------- #
def run_batch(
    counts: dict[str, int],
    proto: Protocol,
    *,
    seed: int = 0,
    progress_every: int = 0,
) -> tuple[list[dict], dict]:
    """Run ``counts[kind]`` experiments per DGP kind with independently-seeded
    child RNGs (fully reproducible given ``seed`` and the protocol)."""
    order: tuple[DGPKind, ...] = ("iid", "ar1", "garch", "break")
    total = sum(counts.get(k, 0) for k in order)
    children = np.random.SeedSequence(seed).spawn(total)
    t0 = proto.test_start()
    break_range = (t0 + int(BREAK_POS_RANGE[0] * proto.test_steps),
                   t0 + int(BREAK_POS_RANGE[1] * proto.test_steps))

    records: list[dict] = []
    traj = TrajectoryAccumulator(proto)
    i = 0
    for kind in order:
        for _ in range(counts.get(kind, 0)):
            rng = np.random.default_rng(children[i])
            lam = float(rng.uniform(*EWMA_LAMBDA_RANGE))
            cfg = sample_config(rng, kind, n_steps=proto.total_steps(),
                                break_range=break_range if kind == "break" else None)
            rows, tr, _ = run_experiment(cfg, proto, rng, ewma_lambda=lam, exp_id=i)
            records.extend(rows)
            if tr is not None:
                traj.add(tr)
            i += 1
            if progress_every and i % progress_every == 0:
                print(f"  {i}/{total} experiments", flush=True)
    return records, traj.result()
