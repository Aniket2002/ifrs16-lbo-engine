import json

import numpy as np
import pandas as pd
import pytest

from analysis.calibration import bayes_calibrate as bayes


def firms(n=6):
    return [
        bayes.FirmData(
            name=f"firm-{i}",
            revenue_growth=0.025 + i * 0.002,
            terminal_margin=0.22 + i * 0.01,
            lease_multiple=2.5 + i * 0.2,
            senior_rate=0.045 + i * 0.002,
            mezz_rate=0.075 + i * 0.003,
            region="metadata",
        )
        for i in range(n)
    ]


def calibrator():
    result = bayes.BayesianCalibrator(seed=19)
    for firm in firms():
        result.add_firm(firm)
    return result


def test_requested_mcmc_never_silently_falls_back(monkeypatch):
    model = calibrator()
    monkeypatch.setattr(bayes, "HAS_PYMC", False)
    with pytest.raises(ImportError, match="no fallback"):
        model.fit_hierarchical_model("mcmc")
    assert model.actual_fit_method is None


def test_empirical_fit_is_named_accurately_and_map_is_rejected():
    model = calibrator()
    estimates = model.fit_hierarchical_model("empirical_moments")
    assert model.actual_fit_method == "empirical_moments"
    assert estimates["mu_g"] == pytest.approx(np.mean([f.revenue_growth for f in firms()]))
    with pytest.raises(ValueError, match="never MAP/Laplace"):
        model.fit_hierarchical_model("map")


def test_empirical_draws_are_seed_deterministic_and_bounded():
    model = calibrator()
    model.fit_hierarchical_model()
    first = model.generate_empirical_parameter_draws(50, seed=23)
    second = model.generate_empirical_parameter_draws(50, seed=23)
    pd.testing.assert_frame_equal(first, second)
    assert first.revenue_growth.between(*model.priors.g_bounds).all()
    assert first.terminal_margin.between(*model.priors.m_bounds).all()
    assert first.senior_rate.between(*model.priors.r_bounds).all()
    assert first.mezz_rate.between(*model.priors.r_bounds).all()
    assert (first.lease_multiple > 0).all()


class FakeValue:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)


class FakeVariable:
    def __init__(self, values):
        self.values = values

    def isel(self, *, sample):
        return FakeValue(self.values[sample])


class FakePosterior:
    def __init__(self):
        self.sizes = {"sample": 2}
        self.variables = {
            "mu_g": [0.02, 0.12],
            "sigma_g": [1e-9, 1e-9],
            "mu_m": [0.15, 0.35],
            "sigma_m": [1e-9, 1e-9],
            "mu_L": [0.5, 2.0],
            "sigma_L": [1e-9, 1e-9],
            "mu_r_sen": [0.03, 0.11],
            "sigma_r_sen": [1e-9, 1e-9],
            "mu_r_mezz": [0.04, 0.10],
            "sigma_r_mezz": [1e-9, 1e-9],
        }

    def stack(self, **_kwargs):
        return self

    def __getitem__(self, name):
        return FakeVariable(self.variables[name])


class FakeTrace:
    posterior = FakePosterior()


def test_trace_predictive_uses_multiple_posterior_draws():
    model = calibrator()
    model.trace = FakeTrace()
    model.actual_fit_method = "mcmc"
    sample = model.generate_population_predictive_from_trace(100, seed=2)
    assert sample.revenue_growth.min() < 0.03
    assert sample.revenue_growth.max() > 0.11
    assert sample.lease_multiple.min() < 2
    assert sample.lease_multiple.max() > 7


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("name", "", "nonempty"),
        ("revenue_growth", np.nan, "finite"),
        ("terminal_margin", 0.8, "within"),
        ("lease_multiple", 0, "positive"),
        ("senior_rate", np.inf, "finite"),
        ("mezz_rate", 0.5, "within"),
    ],
)
def test_invalid_firm_inputs_are_rejected(field, value, match):
    values = firms(1)[0].__dict__.copy()
    values[field] = value
    with pytest.raises(ValueError, match=match):
        bayes.BayesianCalibrator().add_firm(bayes.FirmData(**values))


def test_missing_csv_variables_fail_clearly(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"name": ["x"], "lease_multiple": [3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        bayes.BayesianCalibrator().load_from_csv(path)


def test_export_records_actual_method_data_class_and_unused_covariates(tmp_path):
    model = calibrator()
    model.fit_hierarchical_model("empirical_moments")
    path = tmp_path / "fit.json"
    model.export_priors(path, data_classification="unverified_stipulated")
    exported = json.loads(path.read_text())
    assert exported["model_info"]["actual_fit_method"] == "empirical_moments"
    assert exported["model_info"]["data_classification"] == "unverified_stipulated"
    assert exported["model_info"]["covariates_used_in_model"] == []


def test_recovery_smoke_for_empirical_population_moments():
    rng = np.random.default_rng(31)
    model = bayes.BayesianCalibrator()
    for i in range(500):
        model.add_firm(
            bayes.FirmData(
                f"synthetic-{i}",
                rng.normal(0.05, 0.01),
                rng.normal(0.25, 0.02),
                np.exp(rng.normal(1.2, 0.15)),
                rng.normal(0.06, 0.005),
                rng.normal(0.09, 0.007),
            )
        )
    fit = model.fit_hierarchical_model("empirical_moments")
    assert fit["mu_g"] == pytest.approx(0.05, abs=0.002)
    assert fit["sigma_g"] == pytest.approx(0.01, abs=0.002)
    assert fit["mu_L"] == pytest.approx(1.2, abs=0.03)


def test_fit_and_predict_preconditions():
    model = bayes.BayesianCalibrator()
    with pytest.raises(ValueError, match="at least 2"):
        model.fit_hierarchical_model()
    with pytest.raises(ValueError, match="fit"):
        model.generate_posterior_predictive()
    fitted = calibrator()
    fitted.fit_hierarchical_model()
    with pytest.raises(ValueError, match="positive"):
        fitted.generate_empirical_parameter_draws(0)
