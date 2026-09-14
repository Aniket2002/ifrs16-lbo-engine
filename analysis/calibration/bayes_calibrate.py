"""Audited population calibration for five independent LBO input dimensions."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

try:
    import arviz as az  # type: ignore
    import pymc as pm  # type: ignore

    HAS_PYMC = True
except ImportError:
    az = None
    pm = None
    HAS_PYMC = False


@dataclass(frozen=True)
class FirmData:
    name: str
    revenue_growth: float
    terminal_margin: float
    lease_multiple: float
    senior_rate: float
    mezz_rate: float
    region: str | None = None
    rating: str | None = None
    brand_tier: str | None = None


@dataclass(frozen=True)
class PriorSpecification:
    mu_g_prior: tuple[float, float] = (0.04, 0.02)
    sigma_g_prior: tuple[float, float] = (0.0, 0.015)
    g_bounds: tuple[float, float] = (0.0, 0.15)
    mu_m_prior: tuple[float, float] = (0.25, 0.05)
    sigma_m_prior: tuple[float, float] = (0.0, 0.03)
    m_bounds: tuple[float, float] = (0.1, 0.4)
    mu_L_prior: tuple[float, float] = (1.2, 0.3)
    sigma_L_prior: tuple[float, float] = (0.0, 0.2)
    mu_r_prior: tuple[float, float] = (0.06, 0.01)
    sigma_r_prior: tuple[float, float] = (0.0, 0.008)
    r_bounds: tuple[float, float] = (0.02, 0.12)


PARAMETERS = (
    "mu_g",
    "sigma_g",
    "mu_m",
    "sigma_m",
    "mu_L",
    "sigma_L",
    "mu_r_sen",
    "sigma_r_sen",
    "mu_r_mezz",
    "sigma_r_mezz",
)


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


class BayesianCalibrator:
    """Independent Bayesian population models; covariates are metadata only."""

    def __init__(self, seed: int = 42, priors: PriorSpecification | None = None):
        self.seed = seed
        self.priors = priors or PriorSpecification()
        self.firms: list[FirmData] = []
        self.hyperparameters: dict[str, float] | None = None
        self.population_samples: pd.DataFrame | None = None
        self.trace = None
        self.actual_fit_method: str | None = None
        self.actual_sampler: str | None = None

    def add_firm(self, firm: FirmData) -> None:
        if not isinstance(firm.name, str) or not firm.name.strip():
            raise ValueError("name must be a nonempty string")
        fields = (
            "revenue_growth",
            "terminal_margin",
            "lease_multiple",
            "senior_rate",
            "mezz_rate",
        )
        values = {field: _finite_number(getattr(firm, field), field) for field in fields}
        if values["lease_multiple"] <= 0:
            raise ValueError("lease_multiple must be positive before log transformation")
        for field, bounds in (
            ("revenue_growth", self.priors.g_bounds),
            ("terminal_margin", self.priors.m_bounds),
            ("senior_rate", self.priors.r_bounds),
            ("mezz_rate", self.priors.r_bounds),
        ):
            if not bounds[0] <= values[field] <= bounds[1]:
                raise ValueError(f"{field} must be within {bounds}")
        self.firms.append(firm)

    def load_from_csv(self, csv_path: str | Path) -> None:
        data = pd.read_csv(csv_path)
        required = {
            "name",
            "revenue_growth",
            "terminal_margin",
            "lease_multiple",
            "senior_rate",
            "mezz_rate",
        }
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        if data.empty:
            raise ValueError("Calibration data must contain at least two firms")
        dataclass_fields = FirmData.__dataclass_fields__
        for row in data.to_dict("records"):
            self.add_firm(FirmData(**{key: row.get(key) for key in dataclass_fields}))

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "g": np.asarray([f.revenue_growth for f in self.firms]),
            "m": np.asarray([f.terminal_margin for f in self.firms]),
            "L_log": np.log([f.lease_multiple for f in self.firms]),
            "r_sen": np.asarray([f.senior_rate for f in self.firms]),
            "r_mezz": np.asarray([f.mezz_rate for f in self.firms]),
        }

    def fit_hierarchical_model(
        self,
        method: str = "empirical_moments",
        *,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        nuts_sampler: str = "pymc",
    ) -> dict[str, float]:
        if len(self.firms) < 2:
            raise ValueError("Need at least 2 firms for population modeling")
        if method == "mcmc":
            if not HAS_PYMC:
                raise ImportError("method='mcmc' requires PyMC; no fallback was run")
            return self._fit_mcmc(draws, tune, chains, target_accept, nuts_sampler)
        if method == "empirical_moments":
            return self._fit_empirical_moments()
        raise ValueError(
            "method must be 'mcmc' or 'empirical_moments'; 'map' was never MAP/Laplace"
        )

    def _fit_mcmc(
        self, draws: int, tune: int, chains: int, target_accept: float, nuts_sampler: str
    ) -> dict[str, float]:
        data, prior = self._arrays(), self.priors
        with pm.Model():
            mu_g = pm.Normal("mu_g", *prior.mu_g_prior)
            sigma_g = pm.HalfNormal("sigma_g", prior.sigma_g_prior[1])
            mu_m = pm.Normal("mu_m", *prior.mu_m_prior)
            sigma_m = pm.HalfNormal("sigma_m", prior.sigma_m_prior[1])
            mu_L = pm.Normal("mu_L", *prior.mu_L_prior)
            sigma_L = pm.HalfNormal("sigma_L", prior.sigma_L_prior[1])
            mu_r_sen = pm.Normal("mu_r_sen", *prior.mu_r_prior)
            sigma_r_sen = pm.HalfNormal("sigma_r_sen", prior.sigma_r_prior[1])
            mu_r_mezz = pm.Normal("mu_r_mezz", *prior.mu_r_prior)
            sigma_r_mezz = pm.HalfNormal("sigma_r_mezz", prior.sigma_r_prior[1])
            pm.TruncatedNormal(
                "growth_obs",
                mu=mu_g,
                sigma=sigma_g,
                lower=prior.g_bounds[0],
                upper=prior.g_bounds[1],
                observed=data["g"],
            )
            pm.TruncatedNormal(
                "margin_obs",
                mu=mu_m,
                sigma=sigma_m,
                lower=prior.m_bounds[0],
                upper=prior.m_bounds[1],
                observed=data["m"],
            )
            pm.Normal("lease_obs", mu_L, sigma_L, observed=data["L_log"])
            pm.TruncatedNormal(
                "senior_rate_obs",
                mu=mu_r_sen,
                sigma=sigma_r_sen,
                lower=prior.r_bounds[0],
                upper=prior.r_bounds[1],
                observed=data["r_sen"],
            )
            pm.TruncatedNormal(
                "mezz_rate_obs",
                mu=mu_r_mezz,
                sigma=sigma_r_mezz,
                lower=prior.r_bounds[0],
                upper=prior.r_bounds[1],
                observed=data["r_mezz"],
            )
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=min(chains, 2),
                random_seed=self.seed,
                target_accept=target_accept,
                return_inferencedata=True,
                progressbar=False,
                nuts_sampler=nuts_sampler,
            )
        self.hyperparameters = {
            name: float(self.trace.posterior[name].mean()) for name in PARAMETERS
        }
        self.actual_fit_method = "mcmc"
        self.actual_sampler = nuts_sampler
        return self.hyperparameters

    def _fit_empirical_moments(self) -> dict[str, float]:
        data = self._arrays()
        self.hyperparameters = {
            "mu_g": float(data["g"].mean()),
            "sigma_g": float(data["g"].std()),
            "mu_m": float(data["m"].mean()),
            "sigma_m": float(data["m"].std()),
            "mu_L": float(data["L_log"].mean()),
            "sigma_L": float(data["L_log"].std()),
            "mu_r_sen": float(data["r_sen"].mean()),
            "sigma_r_sen": float(data["r_sen"].std()),
            "mu_r_mezz": float(data["r_mezz"].mean()),
            "sigma_r_mezz": float(data["r_mezz"].std()),
        }
        self.actual_fit_method = "empirical_moments"
        return self.hyperparameters

    @staticmethod
    def _truncated(rng, mu, sigma, bounds):
        sigma = max(float(sigma), np.finfo(float).eps)
        return truncnorm.rvs(
            (bounds[0] - mu) / sigma,
            (bounds[1] - mu) / sigma,
            loc=mu,
            scale=sigma,
            random_state=rng,
        )

    def generate_population_predictive_from_trace(
        self, n_samples: int = 1000, *, seed: int | None = None
    ) -> pd.DataFrame:
        if self.actual_fit_method != "mcmc" or self.trace is None:
            raise ValueError("An actual MCMC trace is required")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        posterior = self.trace.posterior.stack(sample=("chain", "draw"))
        indices = rng.integers(0, posterior.sizes["sample"], n_samples)
        rows = []
        for sample_id, index in enumerate(indices):
            hp = {name: float(posterior[name].isel(sample=index)) for name in PARAMETERS}
            rows.append(self._draw_population_row(rng, hp, sample_id))
        self.population_samples = pd.DataFrame(rows)
        return self.population_samples

    def generate_empirical_parameter_draws(
        self, n_samples: int = 1000, *, seed: int | None = None
    ) -> pd.DataFrame:
        if self.actual_fit_method != "empirical_moments" or self.hyperparameters is None:
            raise ValueError("An empirical-moments fit is required")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        rows = [self._draw_population_row(rng, self.hyperparameters, i) for i in range(n_samples)]
        self.population_samples = pd.DataFrame(rows)
        return self.population_samples

    def _draw_population_row(self, rng, hp, sample_id):
        return {
            "revenue_growth": self._truncated(rng, hp["mu_g"], hp["sigma_g"], self.priors.g_bounds),
            "terminal_margin": self._truncated(
                rng, hp["mu_m"], hp["sigma_m"], self.priors.m_bounds
            ),
            "lease_multiple": float(np.exp(rng.normal(hp["mu_L"], hp["sigma_L"]))),
            "senior_rate": self._truncated(
                rng, hp["mu_r_sen"], hp["sigma_r_sen"], self.priors.r_bounds
            ),
            "mezz_rate": self._truncated(
                rng, hp["mu_r_mezz"], hp["sigma_r_mezz"], self.priors.r_bounds
            ),
            "sample_id": sample_id,
        }

    def generate_posterior_predictive(self, n_samples: int = 1000) -> pd.DataFrame:
        """Compatibility dispatcher based on the method actually fitted."""
        if self.actual_fit_method == "mcmc":
            return self.generate_population_predictive_from_trace(n_samples)
        if self.actual_fit_method == "empirical_moments":
            return self.generate_empirical_parameter_draws(n_samples)
        raise ValueError("Must fit a model before generating samples")

    def diagnostic_summary(self) -> pd.DataFrame:
        if self.actual_fit_method != "mcmc" or self.trace is None or az is None:
            raise ValueError("An actual MCMC trace is required")
        result = az.summary(self.trace, var_names=list(PARAMETERS), kind="diagnostics")
        return result.reset_index(names="parameter").rename(
            columns={"ess_bulk": "bulk_ess", "ess_tail": "tail_ess"}
        )

    def export_priors(self, output_path: str | Path, *, data_classification: str) -> None:
        if self.hyperparameters is None or self.actual_fit_method is None:
            raise ValueError("Must fit model before exporting")
        output = {
            "hyperparameters": self.hyperparameters,
            "model_info": {
                "n_firms": len(self.firms),
                "firm_names": [f.name for f in self.firms],
                "seed": self.seed,
                "actual_fit_method": self.actual_fit_method,
                "actual_sampler": self.actual_sampler,
                "data_classification": data_classification,
                "covariates_used_in_model": [],
            },
            "priors": asdict(self.priors),
        }
        Path(output_path).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    def export_samples(self, output_path: str | Path) -> None:
        if self.population_samples is None:
            raise ValueError("Must generate samples before exporting")
        self.population_samples.to_parquet(output_path, index=False)

    def get_firm_summary(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(firm) for firm in self.firms)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audited LBO population calibration")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="analysis/calibration/output")
    parser.add_argument(
        "--method",
        choices=["mcmc", "empirical_moments"],
        default="empirical_moments",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibrator = BayesianCalibrator(seed=args.seed)
    calibrator.load_from_csv(args.input)
    calibrator.fit_hierarchical_model(method=args.method)
    calibrator.generate_posterior_predictive(args.n_samples)
    calibrator.export_priors(output / "priors.json", data_classification="user_supplied")
    calibrator.export_samples(output / "population_samples.parquet")
    calibrator.get_firm_summary().to_csv(output / "firm_summary.csv", index=False)


if __name__ == "__main__":
    main()
