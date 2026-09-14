import copy
import json

import pytest

from lbo.full_simulation import FullSimulationAssumptions, FullSimulationModel
from lbo.validation import SimulationInvariantError, validate_simulation


def _good_path():
    assumptions = FullSimulationAssumptions(years=3)
    return assumptions, FullSimulationModel(assumptions).simulate()


def _assert_corruption(field, value, invariant, *, row_index=0):
    assumptions, rows = _good_path()
    rows = copy.deepcopy(rows)
    rows[row_index][field] = value(rows[row_index]) if callable(value) else value

    with pytest.raises(SimulationInvariantError) as caught:
        validate_simulation(
            rows,
            assumptions,
            scenario_id="adversarial-001",
            source_commit="test-source-sha",
        )

    error = caught.value
    assert error.invariant == invariant
    assert error.year == row_index + 1
    assert error.scenario_id == "adversarial-001"
    record = error.to_record()
    assert record["source_commit"] == "test-source-sha"
    assert record["row"][field] == rows[row_index][field]
    json.dumps(record)


def test_known_good_simulation_validates_every_year():
    assumptions, rows = _good_path()
    report = validate_simulation(rows, assumptions, scenario_id="good-001")

    assert report["validated_years"] == 3
    assert report["scenario_id"] == "good-001"


def test_zero_ratio_denominators_are_explicitly_recorded():
    assumptions = FullSimulationAssumptions(
        years=1,
        revenue_0=0.0,
        ebitda_margin=0.0,
        debt_opening=0.0,
        lease_opening=0.0,
        cash_interest_rate=0.0,
        lease_interest_rate=0.0,
        scheduled_debt_amort=0.0,
        initial_cash=0.0,
        min_cash=0.0,
    )
    report = validate_simulation(FullSimulationModel(assumptions).simulate(), assumptions)

    assert report["zero_ebitda_years"] == [1]
    assert report["zero_interest_denominator_years"] == [1]


def test_detects_corrupted_ending_cash():
    _assert_corruption("ending_cash", lambda row: row["ending_cash"] + 1.0, "cash_reconciliation")


def test_detects_corrupted_closing_term_debt():
    _assert_corruption(
        "debt_balance", lambda row: row["debt_balance"] + 1.0, "term_debt_reconciliation"
    )


def test_detects_revolver_above_capacity():
    assumptions = FullSimulationAssumptions(years=1, revolver_limit=200.0)
    rows = FullSimulationModel(assumptions).simulate()
    rows[0]["revolver_balance"] = 201.0
    rows[0]["revolver_draw"] = 201.0 + rows[0]["revolver_repayment"]
    rows[0]["revolver_draw_for_liquidity"] = (
        rows[0]["revolver_draw"] - rows[0]["revolver_draw_for_amortisation"]
    )
    cash_funded = (
        rows[0]["actual_mandatory_amortisation"] - rows[0]["revolver_draw_for_amortisation"]
    )
    rows[0]["ending_cash"] = (
        rows[0]["opening_cash"]
        + rows[0]["operating_cash_generation"]
        - rows[0]["lease_principal_cash_payment"]
        - cash_funded
        + rows[0]["revolver_draw_for_liquidity"]
        - rows[0]["revolver_repayment"]
        - rows[0]["cash_sweep"]
    )
    rows[0]["cash"] = rows[0]["cash_after_financing"] = rows[0]["ending_cash"]

    with pytest.raises(SimulationInvariantError) as caught:
        validate_simulation(rows, assumptions, scenario_id="capacity-corruption")
    # The excessive draw is caught before it can produce an above-limit closing balance.
    assert caught.value.invariant == "revolver_draw_capacity"
    assert caught.value.year == 1
    assert caught.value.scenario_id == "capacity-corruption"


def test_detects_corrupted_unpaid_amortisation():
    _assert_corruption(
        "unpaid_amortisation",
        lambda row: row["unpaid_amortisation"] + 1.0,
        "amortisation_consistency",
    )


def test_detects_sweep_above_remaining_term_debt():
    assumptions = FullSimulationAssumptions(years=1)
    rows = FullSimulationModel(assumptions).simulate()
    row = rows[0]
    row["cash_sweep"] = row["opening_financial_debt"] - row["actual_mandatory_amortisation"] + 1.0
    row["debt_balance"] = 0.0
    cash_funded = row["actual_mandatory_amortisation"] - row["revolver_draw_for_amortisation"]
    row["ending_cash"] = (
        row["opening_cash"]
        + row["operating_cash_generation"]
        - row["lease_principal_cash_payment"]
        - cash_funded
        + row["revolver_draw_for_liquidity"]
        - row["revolver_repayment"]
        - row["cash_sweep"]
    )
    row["cash"] = row["cash_after_financing"] = row["ending_cash"]
    row["funding_deficit"] = max(0.0, assumptions.min_cash - row["ending_cash"])
    row["insolvency_flag"] = row["funding_deficit"] > 0.0

    with pytest.raises(SimulationInvariantError) as caught:
        validate_simulation(rows, assumptions, scenario_id="sweep-corruption")
    assert caught.value.invariant == "sweep_remaining_term_debt"
    assert caught.value.year == 1
    assert caught.value.scenario_id == "sweep-corruption"


def test_detects_corrupted_year_two_opening_balance():
    assumptions, rows = _good_path()
    rows = copy.deepcopy(rows)
    rows[1]["opening_cash"] += 1.0

    with pytest.raises(SimulationInvariantError) as caught:
        validate_simulation(
            rows,
            assumptions,
            scenario_id="roll-forward-corruption",
            source_commit="test-source-sha",
        )

    error = caught.value
    assert error.invariant == "opening_cash_roll_forward"
    assert error.year == 2
    assert error.scenario_id == "roll-forward-corruption"
    assert error.to_record()["previous_row"] == rows[0]
