import numpy as np

from lbo import AnalyticAssumptions, AnalyticBoundsModel, AnalyticLBOModel


def test_calculate_assumption_bounds():
    b = AnalyticBoundsModel().calculate_assumption_bounds()
    assert np.isfinite(b.icr_error_bound)
    assert np.isfinite(b.leverage_error_bound)
    assert 0 <= b.classification_accuracy_estimate <= 1


def test_analytic_model_supports_two_lease_treatments():
    run_off = AnalyticLBOModel(AnalyticAssumptions(lease_treatment="run_off")).solve_paths()
    steady = AnalyticLBOModel(AnalyticAssumptions(lease_treatment="steady_state")).solve_paths()
    assert run_off.lease_treatment == "run_off"
    assert steady.lease_treatment == "steady_state"
    assert run_off.lease_liability[-1] != steady.lease_liability[-1]
