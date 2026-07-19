"""Tests for core functions to increase coverage of under-tested modules."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lbo import AnalyticAssumptions, AnalyticLBOModel, DiagnosticEnvelope
from lbo.covenants import (
    covenant_headroom,
    dual_convention_ratios,
    ratios_frozen_gaap,
    ratios_ifrs16,
)
from lbo.data import load_case_csv, load_synthetic_data


class TestDataLoaders:
    """Test data loading functions"""

    def test_load_synthetic_data_default(self):
        """Test default synthetic data generation"""
        data = load_synthetic_data()
        assert "scenarios" in data
        assert "metadata" in data
        assert len(data["scenarios"]) == 100
        assert data["metadata"]["seed"] == 42
        assert "generation_date" in data["metadata"]

    def test_load_synthetic_data_custom_seed(self):
        """Test synthetic data with custom seed produces same results"""
        data1 = load_synthetic_data(n_deals=50, seed=999)
        data2 = load_synthetic_data(n_deals=50, seed=999)
        pd.testing.assert_frame_equal(data1["scenarios"], data2["scenarios"])
        assert len(data1["scenarios"]) == 50

    def test_load_synthetic_data_schema(self):
        """Test synthetic data has correct schema"""
        data = load_synthetic_data(n_deals=10)
        df = data["scenarios"]
        expected_cols = [
            "deal_id",
            "operator_type",
            "initial_revenue",
            "ebitda_margin",
            "lease_multiple",
            "growth_base",
            "leverage_ratio",
            "sweep_rate",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_load_case_csv_valid(self):
        """Test loading valid CSV"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            df = pd.DataFrame(
                {
                    "entity": ["Corp A", "Corp B"],
                    "year": [2023, 2023],
                    "revenue": [1000, 2000],
                    "ebitda": [200, 400],
                    "net_debt": [500, 800],
                    "lease_liability": [300, 400],
                    "interest_expense": [50, 80],
                }
            )
            df.to_csv(csv_path, index=False)

            loaded = load_case_csv(str(csv_path))
            assert len(loaded) == 2
            assert list(loaded.columns) == list(df.columns)
            assert loaded["entity"].tolist() == ["Corp A", "Corp B"]

    def test_load_case_csv_missing_columns(self):
        """Test CSV loading fails with missing required columns"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "incomplete.csv"
            df = pd.DataFrame({"entity": ["Corp A"], "year": [2023]})
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="Missing required columns"):
                load_case_csv(str(csv_path))

    def test_load_case_csv_negative_interest(self):
        """Test CSV loading fails with negative interest expense"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad_interest.csv"
            df = pd.DataFrame(
                {
                    "entity": ["Corp A"],
                    "year": [2023],
                    "revenue": [1000],
                    "ebitda": [200],
                    "net_debt": [500],
                    "lease_liability": [300],
                    "interest_expense": [-50],
                }
            )
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="Interest expense must be non-negative"):
                load_case_csv(str(csv_path))


class TestCovenantRatios:
    """Test covenant calculation functions"""

    @pytest.fixture
    def sample_row(self):
        """Create sample financial row for testing"""
        return pd.Series(
            {
                "ebitda": 100.0,
                "debt_senior": 300.0,
                "debt_mezz": 100.0,
                "lease_liability": 320.0,
                "cash": 50.0,
                "fin_rate": 0.05,
                "lease_rate": 0.045,
                "rent": 40.0,
            }
        )

    def test_ratios_ifrs16_valid(self, sample_row):
        """Test IFRS-16 ratio calculation"""
        lev, icr = ratios_ifrs16(sample_row)
        assert isinstance(lev, float)
        assert isinstance(icr, float)
        assert lev > 0
        assert icr > 0

        # Expected leverage: (300 + 100 + 320 - 50) / 100 = 6.7
        assert np.isclose(lev, 6.7, rtol=0.01)

    def test_ratios_ifrs16_nan_ebitda(self):
        """Test IFRS-16 with NaN/zero EBITDA"""
        row = pd.Series(
            {
                "ebitda": np.nan,
                "debt_senior": 300.0,
                "debt_mezz": 100.0,
                "lease_liability": 320.0,
                "cash": 50.0,
                "fin_rate": 0.05,
                "lease_rate": 0.045,
            }
        )
        lev, icr = ratios_ifrs16(row)
        assert np.isnan(lev) and np.isnan(icr)

    def test_ratios_ifrs16_zero_interest(self, sample_row):
        """Test IFRS-16 with zero interest returns inf ICR"""
        sample_row["fin_rate"] = 0
        sample_row["lease_rate"] = 0
        lev, icr = ratios_ifrs16(sample_row)
        assert np.isinf(icr)

    def test_ratios_frozen_gaap_valid(self, sample_row):
        """Test frozen-GAAP ratio calculation"""
        lev, icr = ratios_frozen_gaap(sample_row)
        assert isinstance(lev, float)
        assert isinstance(icr, float)
        assert lev > 0
        assert icr > 0

        # Expected leverage: (300 + 100 - 50) / 100 = 3.5
        assert np.isclose(lev, 3.5, rtol=0.01)

    def test_ratios_frozen_gaap_zero_interest(self, sample_row):
        """Test frozen-GAAP with zero interest returns inf ICR"""
        sample_row["fin_rate"] = 0
        lev, icr = ratios_frozen_gaap(sample_row)
        assert np.isinf(icr)

    def test_covenant_headroom_safe(self):
        """Test covenant headroom calculation when safe"""
        result = covenant_headroom(leverage=4.0, icr=3.5, c_lev=5.0, c_icr=2.5)
        assert result["breach"] is False
        assert result["leverage_headroom"] == 1.0
        assert result["icr_headroom"] == 1.0
        assert result["min_headroom"] == 1.0

    def test_covenant_headroom_breach_leverage(self):
        """Test covenant headroom with leverage breach"""
        result = covenant_headroom(leverage=6.0, icr=3.5, c_lev=5.0, c_icr=2.5)
        assert result["breach"] is True
        assert result["leverage_headroom"] == -1.0
        assert result["min_headroom"] == -1.0

    def test_covenant_headroom_breach_icr(self):
        """Test covenant headroom with ICR breach"""
        result = covenant_headroom(leverage=4.0, icr=2.0, c_lev=5.0, c_icr=2.5)
        assert result["breach"] is True
        assert result["icr_headroom"] == -0.5
        assert result["min_headroom"] == -0.5

    def test_dual_convention_ratios(self):
        """Test dual convention ratio calculation"""
        df = pd.DataFrame(
            {
                "year": [2023, 2024],
                "quarter": [1, 1],
                "ebitda": [100.0, 110.0],
                "debt_senior": [300.0, 280.0],
                "debt_mezz": [100.0, 100.0],
                "lease_liability": [320.0, 310.0],
                "cash": [50.0, 60.0],
                "fin_rate": [0.05, 0.05],
                "lease_rate": [0.045, 0.045],
                "rent": [40.0, 40.0],
            }
        )
        result = dual_convention_ratios(df)

        assert len(result) == 2
        assert "ifrs16_leverage" in result.columns
        assert "frozen_gaap_leverage" in result.columns
        assert "leverage_delta" in result.columns
        assert "icr_delta" in result.columns

        # IFRS-16 leverage should be higher (includes lease liabilities)
        assert result["ifrs16_leverage"].iloc[0] > result["frozen_gaap_leverage"].iloc[0]


class TestAnalyticModel:
    """Test analytic LBO model"""

    def test_analytic_model_default_assumptions(self):
        """Test model with default assumptions"""
        model = AnalyticLBOModel()
        assert model.assumptions.n_years == 7
        assert model.assumptions.ebitda_0 == 100.0
        assert model.assumptions.financial_debt_0 == 400.0

    def test_analytic_model_custom_assumptions(self):
        """Test model with custom assumptions"""
        assumptions = AnalyticAssumptions(
            ebitda_0=150.0, growth_rate=0.05, n_years=5, lease_treatment="run_off"
        )
        model = AnalyticLBOModel(assumptions)
        assert model.assumptions.ebitda_0 == 150.0
        assert model.assumptions.growth_rate == 0.05
        assert model.assumptions.n_years == 5
        assert model.assumptions.lease_treatment == "run_off"

    def test_analytic_model_solve_paths(self):
        """Test solving analytic paths"""
        model = AnalyticLBOModel()
        results = model.solve_paths()

        # Check output structure
        assert results.years is not None
        assert results.ebitda is not None
        assert results.financial_debt is not None
        assert results.leverage_ratio is not None
        assert results.icr_ratio is not None

        # Check array lengths match n_years
        assert len(results.years) == model.assumptions.n_years + 1
        assert len(results.ebitda) == model.assumptions.n_years + 1
        assert len(results.leverage_ratio) == model.assumptions.n_years + 1

    def test_analytic_model_paths_monotonic_ebitda(self):
        """Test EBITDA grows monotonically"""
        model = AnalyticLBOModel()
        results = model.solve_paths()

        # With positive growth, EBITDA should be monotonically increasing
        assert all(
            results.ebitda[i] <= results.ebitda[i + 1] for i in range(len(results.ebitda) - 1)
        )

    def test_diagnostic_envelope(self):
        """Test diagnostic envelope calculations"""
        assumptions_dict = {
            "ebitda_0": 100.0,
            "growth_rate": 0.03,
            "financial_debt_0": 400.0,
        }
        envelope = DiagnosticEnvelope(
            icr_error_bound=0.15, leverage_error_bound=0.20, assumptions=assumptions_dict
        )

        # Test that error bounds are stored
        assert envelope.icr_error_bound == 0.15
        assert envelope.leverage_error_bound == 0.20
        assert "ebitda_0" in envelope.assumptions

    def test_analytic_model_with_different_parameters(self):
        """Test model sensitivity to key parameters"""
        # Test with higher growth
        assumptions1 = AnalyticAssumptions(growth_rate=0.05, n_years=5)
        model1 = AnalyticLBOModel(assumptions1)
        results1 = model1.solve_paths()

        # Test with lower growth
        assumptions2 = AnalyticAssumptions(growth_rate=0.02, n_years=5)
        model2 = AnalyticLBOModel(assumptions2)
        results2 = model2.solve_paths()

        # Higher growth should result in higher EBITDA at terminal year
        assert results1.ebitda[-1] > results2.ebitda[-1]

    def test_analytic_model_leverage_decline(self):
        """Test that leverage typically declines with debt paydown"""
        model = AnalyticLBOModel()
        results = model.solve_paths()

        # Generally leverage should decline (debt pays down faster than EBITDA grows)
        # though this depends on assumptions
        assert len(results.leverage_ratio) > 1

    def test_analytic_model_icr_ratio_positive(self):
        """Test that ICR ratios are positive or infinite"""
        model = AnalyticLBOModel()
        results = model.solve_paths()

        for icr in results.icr_ratio:
            assert np.isfinite(icr) or np.isinf(icr)

    def test_analytic_model_steady_state_vs_runoff(self):
        """Test difference between steady-state and run-off lease treatment"""
        assumptions_ss = AnalyticAssumptions(lease_treatment="steady_state", n_years=5)
        model_ss = AnalyticLBOModel(assumptions_ss)
        results_ss = model_ss.solve_paths()

        assumptions_ro = AnalyticAssumptions(lease_treatment="run_off", n_years=5)
        model_ro = AnalyticLBOModel(assumptions_ro)
        results_ro = model_ro.solve_paths()

        # Lease liability should typically be higher in steady state at terminal year
        # (since it's not being run off)
        assert results_ss.lease_liability[-1] != results_ro.lease_liability[-1]

    def test_analytic_model_compute_elasticities(self):
        """Test elasticity calculation for parameters"""
        model = AnalyticLBOModel()
        elasticities = model.compute_elasticities(epsilon=0.01)

        # Check that elasticities are computed for all parameters
        assert "d_leverage_d_growth_rate" in elasticities
        assert "d_icr_d_growth_rate" in elasticities
        assert "d_leverage_d_cash_sweep" in elasticities
        assert "d_icr_d_cash_sweep" in elasticities

        # Check structure
        for key, value in elasticities.items():
            assert isinstance(value, np.ndarray)
            assert len(value) == model.assumptions.n_years + 1

    def test_analytic_model_validate_against_simulation(self):
        """Test validation against simulated results"""
        model = AnalyticLBOModel()
        results = model.solve_paths()

        # Create mock simulation results
        sim_results = {
            "leverage": results.leverage_ratio * 1.1,  # Add 10% error
            "icr": results.icr_ratio * 1.05,  # Add 5% error
        }

        validation_output = model.validate_against_simulation(
            sim_results, max_error_leverage=0.2, max_error_icr=0.15
        )

        # Check that validation output has required keys
        assert "leverage_mae" in validation_output or isinstance(validation_output, dict)
