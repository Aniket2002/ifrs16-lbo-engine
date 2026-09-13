import json
from pathlib import Path

import numpy as np
import numpy_financial as npf
import pandas as pd
import pytest

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


def waterfall_assumptions(**overrides):
    values = dict(
        years=1,
        revenue_0=100.0,
        revenue_growth=0.0,
        ebitda_margin=0.0,
        da_pct_revenue=0.0,
        tax_rate=0.0,
        capex_pct_revenue=0.0,
        wc_pct_revenue=0.0,
        cash_interest_rate=0.0,
        lease_opening=0.0,
        lease_additions_pct_revenue=0.0,
        debt_opening=100.0,
        scheduled_debt_amort=30.0,
        initial_cash=0.0,
        min_cash=25.0,
        cash_sweep=0.0,
        revolver_limit=100.0,
    )
    values.update(overrides)
    return FullSimulationAssumptions(**values)


@pytest.mark.parametrize(
    "opening,margin,limit,minimum,paid,draw,ending,unpaid",
    [
        (10, 0.8, 100, 25, 30, 0, 60, 0),  # sufficient operating cash
        (10, 0.0, 100, 0, 30, 20, 0, 0),  # partial refinancing
        (0, 0.0, 100, 0, 30, 30, 0, 0),  # full refinancing
        (40, 0.0, 100, 25, 30, 15, 25, 0),  # liquidity only
        (10, 0.0, 10, 25, 20, 10, 0, 10),  # partial payment/default
        (10, 0.0, 100, 25, 30, 45, 25, 0),  # refinancing plus liquidity
        (0, -0.2, 100, 25, 30, 75, 25, 0),  # preserve operating deficit
        (0, -0.2, 10, 25, 10, 10, -20, 20),  # unfunded cash deficit
        (0, 0.0, 30, 25, 30, 30, 0, 0),  # no liquidity capacity left
    ],
)
def test_waterfall_cash_and_debt_conservation(
    opening, margin, limit, minimum, paid, draw, ending, unpaid
):
    a = waterfall_assumptions(
        initial_cash=opening, ebitda_margin=margin, revolver_limit=limit, min_cash=minimum
    )
    r = FullSimulationModel(a).simulate()[0]
    assert r["actual_mandatory_amortisation"] == pytest.approx(paid)
    assert r["revolver_draw"] == pytest.approx(draw)
    assert r["ending_cash"] == pytest.approx(ending)
    assert r["unpaid_amortisation"] == pytest.approx(unpaid)
    assert r["payment_default_flag"] == (unpaid > 0)
    assert r["funding_deficit"] == pytest.approx(max(0, minimum - ending))
    assert r["insolvency_flag"] == (ending < minimum)
    assert r["ending_cash"] == pytest.approx(
        opening
        + r["operating_cash_generation"]
        - r["lease_principal_cash_payment"]
        - paid
        - r["cash_sweep"]
        + draw
        - r["revolver_repayment"]
    )
    assert r["debt_balance"] + r["revolver_balance"] == pytest.approx(
        a.debt_opening - paid - r["cash_sweep"] + draw - r["revolver_repayment"]
    )
    assert 0 <= r["revolver_balance"] <= limit
    assert paid + unpaid == pytest.approx(r["scheduled_debt_amortisation"])
    assert r["revolver_draw_for_amortisation"] == pytest.approx(
        paid - min(paid, max(0, r["cash_before_financing"]))
    )
    assert r["revolver_draw_for_amortisation"] + r["revolver_draw_for_liquidity"] == draw
    assert r["cash_after_mandatory_amortisation"] == pytest.approx(
        r["cash_before_financing"] - paid + r["revolver_draw_for_amortisation"]
    )


def test_cash_sweep_cannot_exceed_remaining_term_debt():
    a = waterfall_assumptions(initial_cash=100, debt_opening=40, cash_sweep=1)
    r = FullSimulationModel(a).simulate()[0]
    assert r["cash_sweep"] == 10
    assert r["ending_cash"] == 60
    assert r["debt_balance"] == 0


@pytest.mark.parametrize("opening_cash", [0.0, 40.0, 100.0])
def test_opening_cash_is_funded_as_a_use(opening_cash):
    a = FullSimulationAssumptions(initial_cash=opening_cash)
    su = entry_sources_and_uses(a)
    assert su["cash_sources"] == 0
    assert su["total_uses"] == pytest.approx(1030 + opening_cash)
    assert su["sponsor_equity"] == pytest.approx(580 + opening_cash)
    assert su["opening_cash_use"] == opening_cash
    assert su["total_sources"] == su["total_uses"]
    assert su["debt_sources"] + su["cash_sources"] + su["sponsor_equity"] == su["total_uses"]
    rows = FullSimulationModel(a).simulate()
    assert rows[0]["opening_cash"] == opening_cash
    assert equity_cash_flow_vector(rows, a)[0] == -su["sponsor_equity"]


def test_zero_sponsor_equity_is_not_replaced_with_epsilon():
    a = FullSimulationAssumptions(debt_opening=1070)
    rows = FullSimulationModel(a).simulate()
    assert entry_sources_and_uses(a)["sponsor_equity"] == 0
    assert equity_cash_flow_vector(rows, a)[0] == 0
    metrics = equity_return_metrics(rows, a)
    assert metrics["initial_equity"] == 0
    assert np.isnan(metrics["moic"])
    assert np.isnan(metrics["irr"])


def test_overfunded_entry_is_rejected_instead_of_unbalanced():
    with pytest.raises(ValueError, match="debt.*uses"):
        entry_sources_and_uses(FullSimulationAssumptions(debt_opening=1100))


def test_revolver_is_repaid_before_sweeping_term_debt_across_years():
    a = waterfall_assumptions(years=3, revenue_growth=1, ebitda_margin=0.2, cash_sweep=1)
    rows = FullSimulationModel(a).simulate()
    assert [r["revolver_balance"] for r in rows] == [35, 25, 0]
    assert [r["cash_sweep"] for r in rows] == [0, 0, 10]
    assert [r["ending_cash"] for r in rows] == [25, 25, 40]
    for previous, current in zip(rows, rows[1:]):
        assert current["opening_cash"] == previous["ending_cash"]
        assert current["opening_financial_debt"] == previous["debt_balance"]
        assert current["opening_revolver"] == previous["revolver_balance"]
        assert current["ending_cash"] == pytest.approx(
            current["cash_before_financing"]
            - current["actual_mandatory_amortisation"]
            + current["revolver_draw"]
            - current["revolver_repayment"]
            - current["cash_sweep"]
        )


def test_empty_simulation_has_no_invented_returns():
    a = FullSimulationAssumptions(years=0)
    rows = FullSimulationModel(a).simulate()
    assert equity_cash_flow_vector(rows, a) == []
    metrics = equity_return_metrics(rows, a)
    assert metrics["entry_sources_and_uses"] == {}
    assert np.isnan(metrics["irr"])
    assert np.isnan(metrics["moic"])


def test_unpaid_amortisation_remains_in_debt_after_default():
    a = waterfall_assumptions(years=2, revolver_limit=10, min_cash=0)
    first, second = FullSimulationModel(a).simulate()
    assert first["actual_mandatory_amortisation"] == 10
    assert first["unpaid_amortisation"] == 20
    assert first["debt_balance"] == second["opening_financial_debt"] == 90
    assert second["actual_mandatory_amortisation"] == 0
    assert second["unpaid_amortisation"] == 30
    assert second["debt_balance"] == 90
    assert first["payment_default_flag"] and second["payment_default_flag"]
    assert first["revolver_balance"] == second["revolver_balance"] == 10
