"""Controlled validation of conformal prediction for financial returns:
where distribution-free coverage survives (marginally, under stationary
vol clustering) and where it breaks (conditionally in high-vol regimes,
and after abrupt regime shifts) -- measured against DGPs whose true
conditional quantiles are known exactly."""

from .methods import (
    GBRParams,
    abs_scores,
    aci_update,
    conformal_quantile,
    conformal_quantile_sorted,
    cqr_scores,
    fit_predict_block,
    norm_scores,
)
from .model import (
    DGPConfig,
    SimulatedPath,
    ewma_sigma,
    feature_burn_in,
    feature_matrix,
    fit_ewma_lambda,
    innovation_ppf,
    sample_config,
    simulate_path,
)
from .simulate import (
    Protocol,
    TrajectoryAccumulator,
    method_names,
    run_batch,
    run_experiment,
)

__all__ = [
    # model
    "DGPConfig",
    "SimulatedPath",
    "simulate_path",
    "sample_config",
    "innovation_ppf",
    "feature_matrix",
    "feature_burn_in",
    "ewma_sigma",
    "fit_ewma_lambda",
    # methods
    "GBRParams",
    "conformal_quantile",
    "conformal_quantile_sorted",
    "aci_update",
    "abs_scores",
    "norm_scores",
    "cqr_scores",
    "fit_predict_block",
    # experiments
    "Protocol",
    "method_names",
    "run_experiment",
    "run_batch",
    "TrajectoryAccumulator",
]
__version__ = "0.1.0"
