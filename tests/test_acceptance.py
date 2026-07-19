import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy_financial as npf
import pandas as pd

from analysis.run_benchmark import ensure_synthetic_data
from lbo import (
    AnalyticAssumptions,
    AnalyticLBOModel,
    FullSimulationAssumptions,
    FullSimulationModel,
)
from lbo.covenants import ratios_frozen_gaap, ratios_ifrs16
from lbo.full_simulation import (
    entry_sources_and_uses,
    equity_cash_flow_vector,
    equity_return_metrics,
)


def test_cash_flow_reconciliation():
    sim = FullSimulationModel(FullSimulationAssumptions(years=3)).simulate()
    df = pd.DataFrame(sim)
    lhs = (
        df["opening_cash"]
        + df["ebitda"]
        - df["delta_working_capital"]
        - df["cash_taxes"]
        - df["cash_interest"]
        - df["lease_interest_cash_payment"]
        - df["capex"]
        - df["lease_principal_cash_payment"]
    )
    rhs = (
        df["ending_cash"]
        + df["actual_mandatory_amortisation"]
        + df["cash_sweep"]
        + df["revolver_repayment"]
        - df["revolver_draw"]
    )
    np.testing.assert_allclose(lhs.to_numpy(), rhs.to_numpy(), rtol=0, atol=1e-6)


def test_debt_roll_forward():
    sim = FullSimulationModel(FullSimulationAssumptions(years=4)).simulate()
    rows = pd.DataFrame(sim)
    opening = 450.0
    for _, r in rows.iterrows():
        expected = max(
            0.0,
            opening - r["actual_mandatory_amortisation"] - r["cash_sweep"],
        )
        assert abs(expected - r["debt_balance"]) <= 1e-6
        opening = r["debt_balance"]


def test_lease_roll_forward():
    a = FullSimulationAssumptions(years=4)
    sim = FullSimulationModel(a).simulate()
    rows = pd.DataFrame(sim)
    opening = a.lease_opening
    for _, r in rows.iterrows():
        expected = max(
            0.0,
            opening + r["lease_additions"] - r["lease_principal_cash_payment"],
        )
        assert abs(expected - r["lease_liability"]) <= 1e-6
        opening = r["lease_liability"]


def test_revolver_limit_and_insolvency_are_recorded():
    a = FullSimulationAssumptions(
        years=1,
        revenue_0=100.0,
        revenue_growth=0.0,
        ebitda_margin=0.0,
        capex_pct_revenue=1.0,
        tax_rate=0.0,
        debt_opening=0.0,
        lease_opening=0.0,
        initial_cash=0.0,
        revolver_limit=0.0,
        scheduled_debt_amort=0.0,
        cash_sweep=0.0,
        lease_interest_rate=0.0,
        lease_additions_pct_revenue=0.0,
        lease_principal_pct_opening=0.0,
    )
    rows = pd.DataFrame(FullSimulationModel(a).simulate())
    assert bool(rows.iloc[0]["insolvency_flag"])
    assert rows.iloc[0]["funding_deficit"] > 0
    assert rows.iloc[0]["ending_cash"] < a.min_cash


def test_ifrs_vs_frozen_gaap_definitions():
    row = pd.Series(
        {
            "ebitda": 100.0,
            "debt_senior": 200.0,
            "debt_mezz": 100.0,
            "lease_liability": 150.0,
            "cash": 20.0,
            "fin_rate": 0.06,
            "lease_rate": 0.05,
            "rent": 10.0,
        }
    )
    lev_ifrs, icr_ifrs = ratios_ifrs16(row)
    lev_gaap, icr_gaap = ratios_frozen_gaap(row)
    assert lev_ifrs > lev_gaap
    assert icr_ifrs < icr_gaap


def test_analytical_vs_simulation_error_is_bounded_in_practice():
    sim = pd.DataFrame(FullSimulationModel(FullSimulationAssumptions(years=5)).simulate())
    analytic = AnalyticLBOModel(
        AnalyticAssumptions(
            n_years=5,
            ebitda_0=sim.iloc[0]["ebitda"],
            financial_debt_0=450.0,
            lease_liability_0=350.0,
            lease_treatment="run_off",
        )
    ).solve_paths()
    sim_leverage = (
        sim["debt_balance"] + sim["revolver_balance"] + sim["lease_liability"] - sim["cash"]
    ) / sim["ebitda"]
    analytic_leverage = np.array(analytic.leverage_ratio[1:])
    assert np.percentile(np.abs(analytic_leverage - sim_leverage.to_numpy()), 95) < 3.0


def test_seed_reproducibility():
    a = FullSimulationAssumptions(years=3)
    x1 = pd.DataFrame(FullSimulationModel(a).simulate())
    x2 = pd.DataFrame(FullSimulationModel(a).simulate())
    pd.testing.assert_frame_equal(x1, x2)


def test_no_nan_or_inf_outputs():
    sim = pd.DataFrame(FullSimulationModel(FullSimulationAssumptions(years=5)).simulate())
    assert np.isfinite(sim.select_dtypes(include=[np.number]).to_numpy()).all()


def test_monotonicity_under_controlled_shock():
    low = pd.DataFrame(
        FullSimulationModel(FullSimulationAssumptions(revenue_growth=0.01, years=5)).simulate()
    )
    high = pd.DataFrame(
        FullSimulationModel(FullSimulationAssumptions(revenue_growth=0.06, years=5)).simulate()
    )
    assert high.iloc[-1]["ebitda"] > low.iloc[-1]["ebitda"]


def test_exact_benchmark_checksums():
    ensure_synthetic_data()
    root = Path(__file__).resolve().parents[1]
    checksums = json.loads((root / "data/synthetic/checksums.json").read_text(encoding="utf-8-sig"))

    import hashlib

    ops = hashlib.sha256((root / "data/synthetic/operators.csv").read_bytes()).hexdigest()
    scn = hashlib.sha256((root / "data/synthetic/scenario_parameters.csv").read_bytes()).hexdigest()

    assert ops == checksums["operators_csv_sha256"]
    assert scn == checksums["scenario_parameters_csv_sha256"]


def test_exit_bridge_and_equity_cash_flow_vector():
    a = FullSimulationAssumptions(years=5)
    rows = FullSimulationModel(a).simulate()
    df = pd.DataFrame(rows)
    metrics = equity_return_metrics(rows, a)
    vector = equity_cash_flow_vector(rows, a)
    sources_and_uses = entry_sources_and_uses(a)

    assert len(vector) == a.years + 1
    assert np.isfinite(npf.irr(vector))
    assert abs(metrics["irr"] - npf.irr(vector)) < 1e-9
    assert abs(metrics["moic"] - (metrics["exit_equity"] / metrics["initial_equity"])) < 1e-9
    assert abs(metrics["initial_equity"] - sources_and_uses["sponsor_equity"]) < 1e-9
    assert abs(vector[0] + sources_and_uses["sponsor_equity"]) < 1e-9

    final = df.iloc[-1]
    sale_costs = final["exit_enterprise_value"] * a.sale_cost_pct
    expected_exit_equity = (
        final["exit_enterprise_value"]
        - final["debt_balance"]
        - final["revolver_balance"]
        - final["lease_liability"]
        + final["ending_cash"]
        - sale_costs
    )
    assert abs(final["exit_equity"] - expected_exit_equity) <= 1e-6
