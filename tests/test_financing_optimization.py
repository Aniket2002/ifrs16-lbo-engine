from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis.optimization.financing_policy import (
    FinancingPolicy,
    candidate_grid,
    limited_liability_returns,
    select_candidate,
    solve_known_optimum_toy,
)
from analysis.run_v3_optimization import (
    assumptions_for,
    build_operating_scenarios,
    evaluate_one,
    leave_one_template_out,
)


def metric_row(policy, objective, broad=0.1, payment=0.05):
    return {
        **policy.to_record(),
        "median_annualized_return": objective,
        "broad_failure_rate": broad,
        "payment_default_rate": payment,
    }


def test_single_feasible_candidate_is_selected():
    policy = FinancingPolicy(2.0, 0.1, 0.5)
    selected = select_candidate(pd.DataFrame([metric_row(policy, 0.2)]), 0.1, 0.05)
    assert selected.policy_id == policy.policy_id


def test_no_feasible_candidates_is_explicit():
    policy = FinancingPolicy(2.0, 0.1, 0.5)
    assert select_candidate(pd.DataFrame([metric_row(policy, 0.2, 0.2)]), 0.1, 0.05) is None


def test_conservative_tie_rule_is_applied_in_documented_order():
    policies = [
        FinancingPolicy(2.5, 0.05, 0.7),
        FinancingPolicy(2.0, 0.05, 0.7),
        FinancingPolicy(2.0, 0.05, 0.4),
        FinancingPolicy(2.0, 0.125, 0.4),
    ]
    rows = [metric_row(policy, 0.2 + i * 1e-14) for i, policy in enumerate(policies)]
    selected = select_candidate(pd.DataFrame(rows), 0.1, 0.05)
    assert selected.policy_id == policies[-1].policy_id


def test_known_optimum_toy_returns_independently_derived_boundary():
    result = solve_known_optimum_toy([0, 15, 30, 45, 60])
    # d[(120-D)/(100-D)]/dD = 20/(100-D)^2 > 0 independently.
    assert all(20 / (100 - debt) ** 2 > 0 for debt in [0, 15, 30, 45, 60])
    assert result["selected_debt"] == 60
    assert result["selected_moic"] == pytest.approx(1.5)


def test_stricter_risk_constraint_removes_unconstrained_optimum():
    safe = FinancingPolicy(2.0, 0.1, 0.5)
    risky = FinancingPolicy(3.0, 0.05, 0.7)
    metrics = pd.DataFrame([metric_row(safe, 0.15), metric_row(risky, 0.30, 0.2, 0.1)])
    assert select_candidate(metrics, 0.3, 0.2).policy_id == risky.policy_id
    assert select_candidate(metrics, 0.1, 0.05).policy_id == safe.policy_id


def operator(operator_id="A"):
    return SimpleNamespace(
        operator_id=operator_id,
        revenue_0=1000.0,
        ebitda_0=220.0,
        revenue_growth_mean=0.03,
        revenue_growth_std=0.04,
        ebitda_margin_mean=0.22,
        financial_debt_0=450.0,
        lease_liability_0=350.0,
        cash_0=40.0,
        cash_sweep=0.5,
        lease_principal_rate=0.12,
        lease_additions_rate=0.01,
    )


def scenario(operator_id="A"):
    return SimpleNamespace(
        scenario_id=f"{operator_id}:000",
        operator_id=operator_id,
        scenario_type="base",
        revenue_growth=0.03,
        ebitda_margin=0.22,
    )


def test_identical_policy_evaluation_is_deterministic_and_validated():
    policy = FinancingPolicy(2.0, 0.1, 0.5)
    first = evaluate_one(operator(), scenario(), policy, 0.06, "test")
    second = evaluate_one(operator(), scenario(), policy, 0.06, "test")
    assert first == second
    assert first["sponsor_exit_proceeds"] == max(0, first["raw_exit_equity"])


def complete_outcome(operator_id, policy, annualized):
    return {
        "scenario_id": f"{operator_id}:0",
        "operator_id": operator_id,
        "policy_id": policy.policy_id,
        "sponsor_annualized_return": annualized,
        "sponsor_moic": (1 + annualized) ** 5,
        "total_equity_loss": 0,
        "broad_failure": 0,
        "payment_default": 0,
        "insolvency": 0,
        "covenant_breach": 0,
        "initial_sponsor_equity": 100,
        "opening_debt": policy.debt_multiple * 20,
        "ending_term_debt": 0,
        "max_revolver": 0,
    }


def reference_outcome(operator_id):
    row = complete_outcome(operator_id, FinancingPolicy(2.0, 0.1, 0.5), 0.1)
    row["policy_id"] = f"reference_{operator_id}"
    return row


def test_heldout_outcomes_cannot_change_training_selected_policy():
    low = FinancingPolicy(1.5, 0.1, 0.5)
    high = FinancingPolicy(3.0, 0.1, 0.5)
    outcomes = pd.DataFrame(
        [
            complete_outcome("A", low, 0.1),
            complete_outcome("A", high, 0.9),
            complete_outcome("B", low, 0.2),
            complete_outcome("B", high, 0.3),
            complete_outcome("C", low, 0.3),
            complete_outcome("C", high, 0.2),
        ]
    )
    references = pd.DataFrame([reference_outcome(x) for x in "ABC"])
    first = leave_one_template_out(outcomes, references, [low, high])[1]
    changed = outcomes.copy()
    changed.loc[changed.operator_id == "A", "sponsor_annualized_return"] = [-10, 10]
    second = leave_one_template_out(changed, references, [low, high])[1]
    first_a = first.set_index("held_out_template").loc["A", "selected_policy_id"]
    second_a = second.set_index("held_out_template").loc["A", "selected_policy_id"]
    assert first_a == second_a


def test_scenario_and_candidate_order_do_not_change_results():
    policies = candidate_grid()[:3]
    frame = pd.DataFrame([metric_row(policy, 0.1 + i / 100) for i, policy in enumerate(policies)])
    expected = select_candidate(frame, 0.1, 0.05).policy_id
    assert select_candidate(frame.sample(frac=1, random_state=4), 0.1, 0.05).policy_id == expected
    operators = pd.DataFrame([operator("A").__dict__, operator("B").__dict__])
    first = build_operating_scenarios(operators)
    second = build_operating_scenarios(operators.iloc[::-1])
    pd.testing.assert_frame_equal(first, second)


def test_zero_equity_and_zero_recovery_are_handled_explicitly():
    with pytest.raises(ValueError, match="positive"):
        limited_liability_returns(100, 0, 5)
    result = limited_liability_returns(-20, 100, 5)
    assert result["raw_exit_equity"] == -20
    assert result["sponsor_exit_proceeds"] == 0
    assert result["sponsor_moic"] == 0
    assert result["sponsor_annualized_return"] == -1


def test_candidate_debt_cannot_reach_or_exceed_entry_uses():
    excessive = FinancingPolicy(40.0, 0.1, 0.5)
    with pytest.raises(ValueError, match="below entry uses"):
        assumptions_for(operator(), scenario(), excessive, 0.06)


def test_grid_is_frozen_unique_and_brackets_existing_synthetic_policies():
    policies = candidate_grid()
    assert len(policies) == len({policy.policy_id for policy in policies}) == 112
    assert {policy.debt_multiple for policy in policies} == set(np.arange(1.5, 3.01, 0.25))
    assert {policy.amortisation_rate for policy in policies} == {0.05, 0.075, 0.1, 0.125}
    assert {policy.cash_sweep for policy in policies} == {0.4, 0.5, 0.6, 0.7}
    reference_debt_multiples = [520 / 260, 470 / 205, 610 / 330, 390 / 150, 650 / 390]
    reference_sweeps = [0.55, 0.50, 0.58, 0.48, 0.60]
    assert min(reference_debt_multiples) >= 1.5 and max(reference_debt_multiples) <= 3.0
    assert min(reference_sweeps) >= 0.4 and max(reference_sweeps) <= 0.7


@pytest.mark.parametrize("bad", [[], [np.nan], [-1], [61]])
def test_invalid_toy_grid_rejected(bad):
    with pytest.raises(ValueError):
        solve_known_optimum_toy(bad)
