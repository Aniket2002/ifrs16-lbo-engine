import math

import pytest

from lbo.full_simulation import (
    FullSimulationAssumptions,
    FullSimulationModel,
    equity_cash_flow_vector,
    equity_return_metrics,
)


def test_closed_form_five_year_irr_and_moic_are_independent():
    assumptions = FullSimulationAssumptions(
        years=5,
        entry_enterprise_value=100.0,
        transaction_fees_pct=0.0,
        debt_opening=0.0,
        initial_cash=0.0,
    )
    rows = [{"exit_equity": 0.0} for _ in range(4)] + [{"exit_equity": 200.0}]

    result = equity_return_metrics(rows, assumptions)
    expected_irr = (200.0 / 100.0) ** (1.0 / 5.0) - 1.0
    expected_moic = 200.0 / 100.0

    assert result["equity_cash_flow_vector"] == [-100.0, 0.0, 0.0, 0.0, 0.0, 200.0]
    assert result["irr"] == pytest.approx(expected_irr, rel=0.0, abs=1e-12)
    assert result["moic"] == pytest.approx(expected_moic, rel=0.0, abs=1e-12)


def test_negative_exit_equity_is_retained_as_diagnostic_and_has_no_irr_root():
    assumptions = FullSimulationAssumptions(
        years=5,
        entry_enterprise_value=100.0,
        transaction_fees_pct=0.0,
        debt_opening=0.0,
        initial_cash=0.0,
    )
    rows = [{"exit_equity": 0.0} for _ in range(4)] + [{"exit_equity": -25.0}]

    result = equity_return_metrics(rows, assumptions)

    assert result["exit_equity"] == -25.0
    assert result["moic"] == -25.0 / 100.0
    assert math.isnan(result["irr"])
    assert not math.isinf(result["irr"])


def test_zero_exit_proceeds_has_no_valid_economic_irr_root():
    assumptions = FullSimulationAssumptions(
        years=5,
        entry_enterprise_value=100.0,
        transaction_fees_pct=0.0,
        debt_opening=0.0,
        initial_cash=0.0,
    )
    rows = [{"exit_equity": 0.0} for _ in range(5)]

    result = equity_return_metrics(rows, assumptions)

    assert result["equity_cash_flow_vector"] == [-100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert math.isnan(result["irr"])
    assert not math.isinf(result["irr"])
    assert result["moic"] == 0.0


def test_full_simulation_can_emit_negative_exit_equity_without_flooring():
    assumptions = FullSimulationAssumptions(
        years=1,
        entry_enterprise_value=100.0,
        transaction_fees_pct=0.0,
        revenue_0=0.0,
        ebitda_margin=0.0,
        tax_rate=0.0,
        capex_pct_revenue=0.0,
        wc_pct_revenue=0.0,
        cash_interest_rate=0.0,
        lease_interest_rate=0.0,
        lease_opening=50.0,
        lease_additions_pct_revenue=0.0,
        lease_principal_pct_opening=0.0,
        debt_opening=50.0,
        scheduled_debt_amort=0.0,
        cash_sweep=0.0,
        revolver_limit=0.0,
        initial_cash=0.0,
        min_cash=0.0,
        exit_multiple=0.0,
        sale_cost_pct=0.0,
    )
    rows = FullSimulationModel(assumptions).simulate()
    result = equity_return_metrics(rows, assumptions)

    assert rows[-1]["exit_equity"] == -100.0
    assert equity_cash_flow_vector(rows, assumptions) == [-50.0, -100.0]
    assert math.isnan(result["irr"])
    assert result["moic"] == -2.0


def test_sponsor_cash_flow_shape_cannot_have_multiple_sign_changes():
    assumptions = FullSimulationAssumptions(years=5)
    vector = equity_cash_flow_vector(FullSimulationModel(assumptions).simulate(), assumptions)
    nonzero_signs = [math.copysign(1.0, value) for value in vector if value != 0.0]

    assert len(nonzero_signs) <= 2
    assert sum(left != right for left, right in zip(nonzero_signs, nonzero_signs[1:])) <= 1


def test_hand_derived_fixed_debt_and_interest_schedule():
    assumptions = FullSimulationAssumptions(
        years=5,
        revenue_0=0.0,
        revenue_growth=0.0,
        ebitda_margin=0.0,
        da_pct_revenue=0.0,
        tax_rate=0.0,
        capex_pct_revenue=0.0,
        wc_pct_revenue=0.0,
        cash_interest_rate=0.10,
        lease_interest_rate=0.0,
        lease_opening=0.0,
        lease_additions_pct_revenue=0.0,
        lease_principal_pct_opening=0.0,
        debt_opening=100.0,
        scheduled_debt_amort=20.0,
        cash_sweep=0.0,
        revolver_limit=0.0,
        initial_cash=200.0,
        min_cash=0.0,
    )
    rows = FullSimulationModel(assumptions).simulate()
    expected = [
        (100.0, 10.0, 20.0, 80.0),
        (80.0, 8.0, 20.0, 60.0),
        (60.0, 6.0, 20.0, 40.0),
        (40.0, 4.0, 20.0, 20.0),
        (20.0, 2.0, 20.0, 0.0),
    ]

    actual = [
        (
            row["opening_financial_debt"],
            row["cash_interest"],
            row["actual_mandatory_amortisation"],
            row["debt_balance"],
        )
        for row in rows
    ]
    assert actual == expected
    assert all(row["revolver_balance"] == 0.0 for row in rows)
    assert all(row["unpaid_amortisation"] == 0.0 for row in rows)
