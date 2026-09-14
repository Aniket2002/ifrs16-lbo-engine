"""Run the frozen v3 Bayesian audit and synthetic validation protocol."""

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from analysis.calibration.bayes_calibrate import (
    PARAMETERS,
    BayesianCalibrator,
    FirmData,
    PriorSpecification,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/calibration/hotel_operators.csv"
PROTOCOL = ROOT / "docs/V3_BAYESIAN_VALIDATION_PROTOCOL.md"
OUTPUT = ROOT / "results/v3/bayesian_validation"
SEEDS = (1101, 1102, 1103)
MCMC = {"draws": 600, "tune": 600, "chains": 2, "target_accept": 0.95, "nuts_sampler": "nutpie"}

CENTRAL = {
    "mu_g": 0.05,
    "sigma_g": 0.012,
    "mu_m": 0.26,
    "sigma_m": 0.025,
    "mu_L": 1.2,
    "sigma_L": 0.18,
    "mu_r_sen": 0.06,
    "sigma_r_sen": 0.007,
    "mu_r_mezz": 0.09,
    "sigma_r_mezz": 0.009,
}
DESIGNS = {
    "central": (30, CENTRAL),
    "low_variance": (
        30,
        {
            **CENTRAL,
            "sigma_g": 0.004,
            "sigma_m": 0.008,
            "sigma_L": 0.05,
            "sigma_r_sen": 0.0025,
            "sigma_r_mezz": 0.003,
        },
    ),
    "high_variance": (
        30,
        {
            **CENTRAL,
            "sigma_g": 0.025,
            "sigma_m": 0.06,
            "sigma_L": 0.4,
            "sigma_r_sen": 0.016,
            "sigma_r_mezz": 0.016,
        },
    ),
    "near_boundary": (
        30,
        {
            **CENTRAL,
            "mu_g": 0.008,
            "mu_m": 0.12,
            "mu_L": 0.35,
            "mu_r_sen": 0.027,
            "mu_r_mezz": 0.112,
        },
    ),
    "small_sample": (10, CENTRAL),
    "larger_sample": (100, CENTRAL),
}
DIMENSIONS = {
    "revenue_growth": ("mu_g", "sigma_g", (0.0, 0.15), False),
    "terminal_margin": ("mu_m", "sigma_m", (0.1, 0.4), False),
    "lease_multiple": ("mu_L", "sigma_L", (1.0, 8.0), True),
    "senior_rate": ("mu_r_sen", "sigma_r_sen", (0.02, 0.12), False),
    "mezz_rate": ("mu_r_mezz", "sigma_r_mezz", (0.02, 0.12), False),
}


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def truncated(rng, mu, sigma, bounds, size=None):
    return truncnorm.rvs(
        (bounds[0] - mu) / sigma,
        (bounds[1] - mu) / sigma,
        loc=mu,
        scale=sigma,
        size=size,
        random_state=rng,
    )


def simulated_firms(truth, n, seed):
    rng = np.random.default_rng(seed)
    values = {}
    for dimension, (mu, sigma, bounds, logged) in DIMENSIONS.items():
        if logged:
            values[dimension] = np.exp(rng.normal(truth[mu], truth[sigma], n))
        else:
            values[dimension] = truncated(rng, truth[mu], truth[sigma], bounds, n)
    return [
        FirmData(
            f"synthetic-{seed}-{i}",
            values["revenue_growth"][i],
            values["terminal_margin"][i],
            values["lease_multiple"][i],
            values["senior_rate"][i],
            values["mezz_rate"][i],
        )
        for i in range(n)
    ]


def fit(firms, seed, priors=None):
    model = BayesianCalibrator(seed=seed, priors=priors)
    for firm in firms:
        model.add_firm(firm)
    model.fit_hierarchical_model("mcmc", **MCMC)
    return model


def posterior_summary(model, design, seed, truth):
    posterior = model.trace.posterior
    rows = []
    for parameter in PARAMETERS:
        values = posterior[parameter].values.reshape(-1)
        true = truth[parameter]
        mean = float(values.mean())
        rows.append(
            {
                "design": design,
                "seed": seed,
                "parameter": parameter,
                "true_value": true,
                "posterior_mean": mean,
                "posterior_median": float(np.median(values)),
                "posterior_sd": float(values.std(ddof=1)),
                "interval_50_low": float(np.quantile(values, 0.25)),
                "interval_50_high": float(np.quantile(values, 0.75)),
                "interval_90_low": float(np.quantile(values, 0.05)),
                "interval_90_high": float(np.quantile(values, 0.95)),
                "covered_90": bool(np.quantile(values, 0.05) <= true <= np.quantile(values, 0.95)),
                "absolute_error": abs(mean - true),
                "relative_error": abs(mean - true) / abs(true),
            }
        )
    return rows


def diagnostics(model, design, seed):
    posterior = model.trace
    rhat = az.rhat(posterior, var_names=list(PARAMETERS))
    bulk = az.ess(posterior, var_names=list(PARAMETERS), method="bulk")
    tail = az.ess(posterior, var_names=list(PARAMETERS), method="tail")
    divergences = int(posterior.sample_stats["diverging"].sum())
    reached = posterior.sample_stats.get("reached_max_treedepth")
    tree_warnings = int(reached.sum()) if reached is not None else 0
    return [
        {
            "design": design,
            "seed": seed,
            "parameter": parameter,
            "r_hat": float(rhat[parameter]),
            "bulk_ess": float(bulk[parameter]),
            "tail_ess": float(tail[parameter]),
            "run_divergences": divergences,
            "run_tree_depth_warnings": tree_warnings,
            "sampler": model.actual_sampler,
            "chains": MCMC["chains"],
            "draws_per_chain": MCMC["draws"],
        }
        for parameter in PARAMETERS
    ]


def predictive_summary(samples, design, seed, truth=None, observed=None):
    rows = []
    for dimension, (mu, _sigma, bounds, _logged) in DIMENSIONS.items():
        values = samples[dimension].to_numpy()
        low, high = np.quantile(values, [0.05, 0.95])
        row = {
            "dataset": design,
            "seed": seed,
            "dimension": dimension,
            "predictive_mean": float(values.mean()),
            "predictive_sd": float(values.std(ddof=1)),
            "predictive_q05": float(low),
            "predictive_median": float(np.median(values)),
            "predictive_q95": float(high),
            "plausible_range_frequency": float(
                ((values >= bounds[0]) & (values <= bounds[1])).mean()
            ),
            "exact_boundary_frequency": float(
                ((values == bounds[0]) | (values == bounds[1])).mean()
            ),
        }
        if truth is not None:
            target = np.exp(truth[mu]) if dimension == "lease_multiple" else truth[mu]
            row.update(
                {
                    "known_location": target,
                    "known_location_in_predictive_90": bool(low <= target <= high),
                }
            )
        if observed is not None:
            raw = observed[dimension].to_numpy()
            row.update({"observed_mean": float(raw.mean()), "observed_sd": float(raw.std(ddof=1))})
        rows.append(row)
    return rows


def prior_predictive(prior, n=20_000, seed=701):
    rng = np.random.default_rng(seed)
    hyper = {
        "mu_g": rng.normal(*prior.mu_g_prior, n),
        "sigma_g": abs(rng.normal(0, prior.sigma_g_prior[1], n)),
        "mu_m": rng.normal(*prior.mu_m_prior, n),
        "sigma_m": abs(rng.normal(0, prior.sigma_m_prior[1], n)),
        "mu_L": rng.normal(*prior.mu_L_prior, n),
        "sigma_L": abs(rng.normal(0, prior.sigma_L_prior[1], n)),
        "mu_r_sen": rng.normal(*prior.mu_r_prior, n),
        "sigma_r_sen": abs(rng.normal(0, prior.sigma_r_prior[1], n)),
        "mu_r_mezz": rng.normal(*prior.mu_r_prior, n),
        "sigma_r_mezz": abs(rng.normal(0, prior.sigma_r_prior[1], n)),
    }
    values = {}
    for dimension, (mu, sigma, bounds, logged) in DIMENSIONS.items():
        if logged:
            values[dimension] = np.exp(rng.normal(hyper[mu], hyper[sigma]))
        else:
            values[dimension] = np.array(
                [
                    truncated(rng, m, max(s, np.finfo(float).eps), bounds)
                    for m, s in zip(hyper[mu], hyper[sigma], strict=True)
                ]
            )
    frame = pd.DataFrame(values)
    return predictive_summary(frame, "prior", seed)


def recovery_aggregate(recovery, diagnostic):
    data = pd.DataFrame(recovery)
    data["signed_relative_error"] = (data.posterior_mean - data.true_value) / data.true_value.abs()
    data["parameter_type"] = np.where(data.parameter.str.startswith("mu_"), "mean", "scale")
    by_design = {name: float(group.covered_90.mean()) for name, group in data.groupby("design")}
    by_type = {}
    for name, group in data.groupby("parameter_type"):
        relative_rmse = float(
            np.sqrt(np.mean(((group.posterior_mean - group.true_value) / group.true_value) ** 2))
        )
        by_type[name] = {"relative_rmse": relative_rmse}
    bias = data.groupby("parameter").signed_relative_error.mean().to_dict()
    diagnostic = pd.DataFrame(diagnostic)
    criteria = {
        "zero_divergences": int(diagnostic.run_divergences.max()) == 0,
        "r_hat_at_most_1_01": bool((diagnostic.r_hat <= 1.01).all()),
        "bulk_ess_at_least_200": bool((diagnostic.bulk_ess >= 200).all()),
        "tail_ess_at_least_200": bool((diagnostic.tail_ess >= 200).all()),
        "overall_90_coverage_at_least_0_80": float(data.covered_90.mean()) >= 0.80,
        "each_design_coverage_at_least_0_70": min(by_design.values()) >= 0.70,
        "mean_relative_rmse_at_most_0_35": by_type["mean"]["relative_rmse"] <= 0.35,
        "scale_relative_rmse_at_most_0_60": by_type["scale"]["relative_rmse"] <= 0.60,
        "max_absolute_relative_bias_at_most_0_75": max(abs(x) for x in bias.values()) <= 0.75,
    }
    return {
        "n_runs": len(DESIGNS) * len(SEEDS),
        "n_parameter_evaluations": len(data),
        "overall_90_interval_coverage": float(data.covered_90.mean()),
        "coverage_by_design": by_design,
        "error_by_parameter_type": by_type,
        "mean_signed_relative_error_by_parameter": bias,
        "diagnostic_extrema": {
            "max_r_hat": float(diagnostic.r_hat.max()),
            "min_bulk_ess": float(diagnostic.bulk_ess.min()),
            "min_tail_ess": float(diagnostic.tail_ess.min()),
            "total_divergences": int(
                diagnostic.drop_duplicates(["design", "seed"]).run_divergences.sum()
            ),
            "total_tree_depth_warnings": int(
                diagnostic.drop_duplicates(["design", "seed"]).run_tree_depth_warnings.sum()
            ),
        },
        "acceptance_criteria": criteria,
        "technical_recovery_gate_passed": all(criteria.values()),
    }


def widened(prior):
    return replace(
        prior,
        mu_g_prior=(prior.mu_g_prior[0], prior.mu_g_prior[1] * 2),
        sigma_g_prior=(0.0, prior.sigma_g_prior[1] * 2),
        mu_m_prior=(prior.mu_m_prior[0], prior.mu_m_prior[1] * 2),
        sigma_m_prior=(0.0, prior.sigma_m_prior[1] * 2),
        mu_L_prior=(prior.mu_L_prior[0], prior.mu_L_prior[1] * 2),
        sigma_L_prior=(0.0, prior.sigma_L_prior[1] * 2),
        mu_r_prior=(prior.mu_r_prior[0], prior.mu_r_prior[1] * 2),
        sigma_r_prior=(0.0, prior.sigma_r_prior[1] * 2),
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    prior = PriorSpecification()
    implementation_audit = {
        "source_commit": source_commit,
        "old_defects": {
            "map_laplace": "sample means/population SDs; no optimization, posterior, Hessian or Laplace covariance",
            "posterior_predictive": "Normal draws conditional on posterior-mean hyperparameters; discarded hyperparameter uncertainty",
            "silent_mcmc_fallback": True,
            "fit_method_provenance": "reported PyMC availability rather than method run",
            "advertised_covariates": "loaded but never entered model",
            "clipping": "deterministic clipping caused flat likelihood regions and predictive boundary point mass",
            "observation_noise": "fixed 0.01 for unlike scales without measurement-error evidence",
        },
        "corrections": {
            "empirical_path": "renamed empirical_moments; no Bayesian or Laplace claim",
            "mcmc": "requested MCMC raises ImportError if unavailable and records actual method/sampler",
            "likelihood": "direct population likelihood; truncated Normal for bounded values, Normal on log lease",
            "predictive": "separate trace-integrated population predictive and empirical parameter draws",
            "covariates": "explicitly metadata only",
            "validation": "finite/domain/schema checks added",
        },
        "major_redesign_required": False,
    }
    provenance = {
        "classification": "UNVERIFIED / STIPULATED CALIBRATION INPUTS",
        "n_rows": 10,
        "repository_search": "no variable-level citations or reconstruction metadata found",
        "variables": {
            field: {"classification": "unverified/stipulated", "source_documented": False}
            for field in (
                "revenue_growth",
                "terminal_margin",
                "lease_multiple",
                "senior_rate",
                "mezz_rate",
            )
        },
        "metadata": {
            field: "unused categorical metadata; provenance unverified"
            for field in ("name", "region", "rating", "brand_tier")
        },
        "sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    }
    dump(OUTPUT / "implementation_audit.json", implementation_audit)
    dump(OUTPUT / "data_provenance.json", provenance)
    dump(
        OUTPUT / "prior_predictive_summary.json",
        {
            "seed": 701,
            "n_draws": 20_000,
            "priors": asdict(prior),
            "dimensions": prior_predictive(prior),
        },
    )

    recovery, diagnostic, predictive = [], [], []
    for design, (n, truth) in DESIGNS.items():
        for seed in SEEDS:
            model = fit(simulated_firms(truth, n, seed), seed)
            recovery.extend(posterior_summary(model, design, seed, truth))
            diagnostic.extend(diagnostics(model, design, seed))
            predictive.extend(
                predictive_summary(
                    model.generate_population_predictive_from_trace(2000, seed=seed + 50_000),
                    design,
                    seed,
                    truth=truth,
                )
            )
    pd.DataFrame(recovery).to_csv(OUTPUT / "recovery_runs.csv", index=False)
    pd.DataFrame(diagnostic).to_csv(OUTPUT / "mcmc_diagnostics.csv", index=False)
    summary = recovery_aggregate(recovery, diagnostic)
    dump(OUTPUT / "recovery_summary.json", summary)

    observed = pd.read_csv(INPUT)
    calibration_firms = []
    loader = BayesianCalibrator()
    loader.load_from_csv(INPUT)
    calibration_firms = loader.firms
    sensitivity_models = {
        "baseline": fit(calibration_firms, 2101, prior),
        "wider": fit(calibration_firms, 2102, widened(prior)),
    }
    sensitivity_rows = []
    for parameter in PARAMETERS:
        baseline_values = (
            sensitivity_models["baseline"].trace.posterior[parameter].values.reshape(-1)
        )
        wider_values = sensitivity_models["wider"].trace.posterior[parameter].values.reshape(-1)
        shift = abs(wider_values.mean() - baseline_values.mean()) / baseline_values.std(ddof=1)
        sensitivity_rows.append(
            {
                "parameter": parameter,
                "baseline_mean": float(baseline_values.mean()),
                "wider_mean": float(wider_values.mean()),
                "baseline_sd": float(baseline_values.std(ddof=1)),
                "absolute_shift_in_baseline_sd": float(shift),
            }
        )
    max_shift = max(row["absolute_shift_in_baseline_sd"] for row in sensitivity_rows)
    sensitivity = {
        "fits": {"baseline_seed": 2101, "wider_seed": 2102},
        "wider_prior": asdict(widened(prior)),
        "parameter_comparison": sensitivity_rows,
        "maximum_shift_in_baseline_posterior_sd": max_shift,
        "material_threshold": 0.5,
        "material_prior_sensitivity": max_shift > 0.5,
    }
    dump(OUTPUT / "prior_sensitivity.json", sensitivity)
    predictive.extend(
        predictive_summary(
            sensitivity_models["baseline"].generate_population_predictive_from_trace(
                5000, seed=31_001
            ),
            "stipulated_calibration_inputs",
            2101,
            observed=observed,
        )
    )
    dump(
        OUTPUT / "posterior_predictive_summary.json",
        {
            "method": "new population observations integrated over actual posterior draws",
            "results": predictive,
        },
    )
    admitted = (
        summary["technical_recovery_gate_passed"] and not sensitivity["material_prior_sensitivity"]
    )
    decision = {
        "classification": "B" if admitted else "C",
        "label": "RETAIN ONLY AS SYNTHETIC METHODOLOGICAL EXPERIMENT"
        if admitted
        else "EXCLUDE FROM V3",
        "technical_validation": "PASS" if admitted else "FAIL",
        "empirical_validation": "FAIL",
        "reason": (
            "Technical MCMC/recovery gates pass, but all calibration variables lack repository provenance and ten cross-sectional rows cannot support real-world claims."
            if admitted
            else "The corrected model fails a frozen technical recovery/diagnostic or prior-sensitivity gate; unverified inputs independently preclude main-results admission."
        ),
        "not_admitted_to_main_results": True,
        "no_lbo_integration_performed": True,
    }
    dump(OUTPUT / "admission_decision.json", decision)
    dump(
        OUTPUT / "run_metadata.json",
        {
            "source_commit": source_commit,
            "command": "python -m analysis.run_v3_bayesian_validation",
            "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "pymc", "arviz", "nutpie")
            },
            "mcmc": MCMC,
            "recovery_seeds": SEEDS,
        },
    )
    print(
        json.dumps(
            {"recovery": summary, "prior_sensitivity": sensitivity, "decision": decision}, indent=2
        )
    )


if __name__ == "__main__":
    main()
