"""Run the frozen v3 synthetic financing-design validation."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.optimization.financing_policy import (
    candidate_grid,
    limited_liability_returns,
    select_candidate,
    solve_known_optimum_toy,
    summarize_outcomes,
)
from lbo.full_simulation import FullSimulationAssumptions, FullSimulationModel
from lbo.validation import validate_simulation

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/synthetic/operators.csv"
PROTOCOL = ROOT / "docs/V3_OPTIMIZATION_VALIDATION_PROTOCOL.md"
OUTPUT = ROOT / "results/v3/optimization_validation"
SEED = 314159
SCENARIOS_PER_TEMPLATE = 100
PRIMARY_RATE = 0.06
HIGHER_RATE = 0.075
ENTRY_MULTIPLE = 7.5
EXIT_MULTIPLE = 8.0
YEARS = 5


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_operating_scenarios(operators: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for operator in operators.sort_values("operator_id").to_dict("records"):
        for index in range(SCENARIOS_PER_TEMPLATE):
            regime = str(rng.choice(["base", "downside", "distressed"], p=[0.5, 0.3, 0.2]))
            growth = rng.normal(operator["revenue_growth_mean"], operator["revenue_growth_std"])
            margin = rng.normal(operator["ebitda_margin_mean"], 0.02)
            if regime == "downside":
                growth -= 0.04
                margin -= 0.03
            elif regime == "distressed":
                growth -= 0.08
                margin -= 0.06
            rows.append(
                {
                    "scenario_id": f"{operator['operator_id']}:{index:03d}",
                    "operator_id": operator["operator_id"],
                    "scenario_type": regime,
                    "revenue_growth": float(np.clip(growth, -0.12, 0.18)),
                    "ebitda_margin": float(np.clip(margin, 0.08, 0.35)),
                }
            )
    result = pd.DataFrame(rows)
    if (
        result.scenario_id.duplicated().any()
        or len(result) != len(operators) * SCENARIOS_PER_TEMPLATE
    ):
        raise ValueError("Operating scenario IDs are not exhaustive and unique")
    return result


def reference_policies(operators: pd.DataFrame) -> pd.DataFrame:
    result = operators[
        ["operator_id", "ebitda_0", "financial_debt_0", "cash_sweep", "cash_0", "lease_liability_0"]
    ].copy()
    result["debt_multiple"] = result.financial_debt_0 / result.ebitda_0
    result["annual_scheduled_amortisation"] = 30.0
    result["amortisation_rate"] = 30.0 / result.financial_debt_0
    return result.rename(
        columns={
            "financial_debt_0": "opening_debt",
            "cash_0": "opening_cash",
            "lease_liability_0": "opening_lease_liability",
        }
    )


def assumptions_for(operator, scenario, policy, term_rate, *, reference=False):
    entry_ebitda = float(operator.ebitda_0)
    if reference:
        debt = float(operator.financial_debt_0)
        amortisation = 30.0
        sweep = float(operator.cash_sweep)
    else:
        debt = policy.debt_multiple * entry_ebitda
        amortisation = policy.amortisation_rate * debt
        sweep = policy.cash_sweep
    assumptions = FullSimulationAssumptions(
        years=YEARS,
        entry_enterprise_value=entry_ebitda * ENTRY_MULTIPLE,
        transaction_fees_pct=0.03,
        revenue_0=float(operator.revenue_0),
        revenue_growth=float(scenario.revenue_growth),
        ebitda_margin=float(scenario.ebitda_margin),
        debt_opening=debt,
        scheduled_debt_amort=amortisation,
        cash_sweep=sweep,
        cash_interest_rate=term_rate,
        lease_interest_rate=0.05,
        lease_opening=float(operator.lease_liability_0),
        lease_additions_pct_revenue=float(operator.lease_additions_rate),
        lease_principal_pct_opening=float(operator.lease_principal_rate),
        initial_cash=float(operator.cash_0),
        min_cash=float(operator.cash_0) * 0.5,
        revolver_limit=entry_ebitda * 0.75,
        exit_multiple=EXIT_MULTIPLE,
    )
    total_uses = assumptions.entry_enterprise_value * 1.03 + assumptions.initial_cash
    if debt >= total_uses:
        raise ValueError("Candidate debt must be below entry uses")
    return assumptions


def evaluate_one(operator, scenario, policy, term_rate, source_commit, *, reference=False):
    assumptions = assumptions_for(operator, scenario, policy, term_rate, reference=reference)
    rows = FullSimulationModel(assumptions).simulate()
    validate_simulation(
        rows, assumptions, scenario_id=scenario.scenario_id, source_commit=source_commit
    )
    frame = pd.DataFrame(rows)
    leverage = (
        frame.debt_balance + frame.revolver_balance + frame.lease_liability - frame.ending_cash
    ) / frame.ebitda
    interest = frame.cash_interest + frame.lease_interest
    coverage = frame.ebitda / interest.replace(0, np.nan)
    negative_ebitda = bool((frame.ebitda <= 0).any())
    payment_default = bool(frame.payment_default_flag.any())
    insolvency = bool(frame.insolvency_flag.any())
    covenant_breach = bool((leverage > 6.0).any() or (coverage < 1.8).any())
    broad_failure = negative_ebitda or payment_default or insolvency or covenant_breach
    initial_equity = float(frame.sponsor_equity.iloc[0])
    returns = limited_liability_returns(float(frame.exit_equity.iloc[-1]), initial_equity, YEARS)
    return {
        "scenario_id": scenario.scenario_id,
        "operator_id": operator.operator_id,
        "scenario_type": scenario.scenario_type,
        "term_rate": term_rate,
        "policy_id": f"reference_{operator.operator_id}" if reference else policy.policy_id,
        "debt_multiple": float(operator.financial_debt_0 / operator.ebitda_0)
        if reference
        else policy.debt_multiple,
        "amortisation_rate": float(30.0 / operator.financial_debt_0)
        if reference
        else policy.amortisation_rate,
        "cash_sweep_fraction": float(operator.cash_sweep) if reference else policy.cash_sweep,
        "opening_debt": assumptions.debt_opening,
        "initial_sponsor_equity": initial_equity,
        "ending_term_debt": float(frame.debt_balance.iloc[-1]),
        "max_revolver": float(frame.revolver_balance.max()),
        "negative_ebitda": int(negative_ebitda),
        "payment_default": int(payment_default),
        "insolvency": int(insolvency),
        "covenant_breach": int(covenant_breach),
        "broad_failure": int(broad_failure),
        "total_equity_loss": int(returns["sponsor_exit_proceeds"] == 0),
        **returns,
    }


def evaluate_all(operators, scenarios, policies, term_rate, source_commit):
    operator_map = {row.operator_id: row for row in operators.itertuples(index=False)}
    policy_rows, reference_rows = [], []
    for operator_id, group in scenarios.groupby("operator_id", sort=True):
        operator = operator_map[operator_id]
        scenario_rows = list(group.itertuples(index=False))
        for scenario in scenario_rows:
            reference_rows.append(
                evaluate_one(operator, scenario, None, term_rate, source_commit, reference=True)
            )
        for policy in policies:
            for scenario in scenario_rows:
                policy_rows.append(
                    evaluate_one(operator, scenario, policy, term_rate, source_commit)
                )
    return pd.DataFrame(policy_rows), pd.DataFrame(reference_rows)


def prefixed(summary, prefix):
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def leave_one_template_out(policy_outcomes, reference_outcomes, policies):
    candidate_rows, fold_rows, heldout_rows = [], [], []
    templates = sorted(reference_outcomes.operator_id.unique())
    policy_map = {policy.policy_id: policy for policy in policies}
    for heldout in templates:
        train_reference = reference_outcomes.loc[reference_outcomes.operator_id != heldout]
        reference_train_summary = summarize_outcomes(train_reference)
        metrics = []
        for policy_id, records in policy_outcomes.loc[
            policy_outcomes.operator_id != heldout
        ].groupby("policy_id", sort=True):
            row = {"held_out_template": heldout, **policy_map[policy_id].to_record()}
            row.update(summarize_outcomes(records))
            row["reference_broad_failure_rate"] = reference_train_summary["broad_failure_rate"]
            row["reference_payment_default_rate"] = reference_train_summary["payment_default_rate"]
            row["feasible"] = bool(
                row["broad_failure_rate"] <= row["reference_broad_failure_rate"] + 1e-12
                and row["payment_default_rate"] <= row["reference_payment_default_rate"] + 1e-12
            )
            metrics.append(row)
            candidate_rows.append(row)
        selected = select_candidate(
            pd.DataFrame(metrics),
            reference_train_summary["broad_failure_rate"],
            reference_train_summary["payment_default_rate"],
        )
        if selected is None:
            fold_rows.append(
                {
                    "held_out_template": heldout,
                    "feasible": False,
                    **prefixed(reference_train_summary, "train_reference"),
                }
            )
            continue
        selected_id = selected.policy_id
        test_candidate = policy_outcomes.loc[
            (policy_outcomes.operator_id == heldout) & (policy_outcomes.policy_id == selected_id)
        ].copy()
        test_reference = reference_outcomes.loc[reference_outcomes.operator_id == heldout].copy()
        candidate_summary = summarize_outcomes(test_candidate)
        reference_summary = summarize_outcomes(test_reference)
        fold_rows.append(
            {
                "held_out_template": heldout,
                "feasible": True,
                "selected_policy_id": selected_id,
                "selected_debt_multiple": selected.debt_multiple,
                "selected_amortisation_rate": selected.amortisation_rate,
                "selected_cash_sweep": selected.cash_sweep,
                "train_objective": selected.median_annualized_return,
                **prefixed(reference_train_summary, "train_reference"),
                **prefixed(candidate_summary, "test_optimized"),
                **prefixed(reference_summary, "test_reference"),
            }
        )
        test_candidate["evaluation_policy"] = "optimized"
        test_reference["evaluation_policy"] = "reference"
        test_candidate["held_out_template"] = heldout
        test_reference["held_out_template"] = heldout
        heldout_rows.extend([test_candidate, test_reference])
    heldout = pd.concat(heldout_rows, ignore_index=True) if heldout_rows else pd.DataFrame()
    if not heldout.empty:
        for kind in ("optimized", "reference"):
            subset = heldout.loc[heldout.evaluation_policy == kind]
            if len(subset) != len(reference_outcomes) or subset.scenario_id.duplicated().any():
                raise ValueError(f"{kind} held-out coverage is not exactly once")
    return pd.DataFrame(candidate_rows), pd.DataFrame(fold_rows), heldout


def experiment_at_rate(operators, scenarios, policies, rate, source_commit):
    policy, reference = evaluate_all(operators, scenarios, policies, rate, source_commit)
    candidates, folds, heldout = leave_one_template_out(policy, reference, policies)
    aggregate = (
        {
            kind: summarize_outcomes(heldout.loc[heldout.evaluation_policy == kind])
            for kind in ("optimized", "reference")
        }
        if not heldout.empty
        else {}
    )
    return candidates, folds, heldout, aggregate


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    operators = pd.read_csv(INPUT)
    scenarios = build_operating_scenarios(operators)
    policies = candidate_grid()
    references = reference_policies(operators)
    pd.DataFrame([policy.to_record() for policy in policies]).to_csv(
        OUTPUT / "candidate_grid.csv", index=False
    )
    references.to_csv(OUTPUT / "reference_policy.csv", index=False)
    toy = solve_known_optimum_toy([0, 15, 30, 45, 60])
    toy.update(
        {
            "independent_formula": "MOIC=(120-D)/(100-D)",
            "independent_derivative": "20/(100-D)^2 > 0 on [0,60]",
            "expected_debt": 60.0,
            "passed": toy["selected_debt"] == 60.0,
        }
    )
    dump(OUTPUT / "toy_validation.json", toy)
    dump(
        OUTPUT / "economic_audit.json",
        {
            "source_commit": source_commit,
            "existing_v2_return_optimization_suitable": False,
            "finding": "EXISTING V2 BENCHMARK IS NOT A VALID RETURN-OPTIMIZATION DATASET.",
            "v2_entry_basis": "7.5 * revenue_0",
            "v2_exit_basis": "8.0 * final EBITDA",
            "intentional_mismatch_documented": False,
            "v2_effect": "none on ranking/covenant results; v2 disclaims return optimization",
            "separate_experiment": "entry and exit EV both use EBITDA multiples",
            "remaining_economic_limitations": [
                "fixed borrowing rate has no leverage-dependent spread",
                "engine continues mechanically after default and has no recovery model",
                "entry sources/uses do not adjust purchase use for assumed lease while exit equity subtracts lease",
                "synthetic operating paths and multiples have no market-practice claim",
            ],
        },
    )
    frozen = {
        "seed": SEED,
        "scenarios_per_template": SCENARIOS_PER_TEMPLATE,
        "entry_multiple_on_ebitda": ENTRY_MULTIPLE,
        "exit_multiple_on_ebitda": EXIT_MULTIPLE,
        "years": YEARS,
        "primary_term_rate": PRIMARY_RATE,
        "higher_term_rate": HIGHER_RATE,
        "grid": [policy.to_record() for policy in policies],
        "objective": "maximize training median limited-liability annualized sponsor return",
        "constraints": [
            "candidate training broad failure rate <= same-path reference rate",
            "candidate training payment-default rate <= same-path reference rate",
        ],
        "tie_rule": [
            "lower broad failure",
            "lower payment default",
            "lower debt multiple",
            "lower cash sweep",
            "higher amortisation",
        ],
        "limited_liability": "proceeds=max(0,raw_exit_equity); zero proceeds return=-100%",
    }
    dump(OUTPUT / "frozen_protocol.json", frozen)

    primary = experiment_at_rate(operators, scenarios, policies, PRIMARY_RATE, source_commit)
    primary[0].to_csv(OUTPUT / "candidate_results.csv", index=False)
    primary[1].to_csv(OUTPUT / "fold_selection.csv", index=False)
    primary[2].to_csv(OUTPUT / "heldout_results.csv", index=False)
    higher = experiment_at_rate(operators, scenarios, policies, HIGHER_RATE, source_commit)
    sensitivity_rows = []
    for rate, experiment in ((PRIMARY_RATE, primary), (HIGHER_RATE, higher)):
        for row in experiment[1].to_dict("records"):
            sensitivity_rows.append(
                {
                    "term_rate": rate,
                    "held_out_template": row["held_out_template"],
                    "feasible": row["feasible"],
                    "selected_policy_id": row.get("selected_policy_id"),
                    "selected_debt_multiple": row.get("selected_debt_multiple"),
                    "selected_amortisation_rate": row.get("selected_amortisation_rate"),
                    "selected_cash_sweep": row.get("selected_cash_sweep"),
                    "heldout_median_annualized_return": row.get(
                        "test_optimized_median_annualized_return"
                    ),
                    "heldout_broad_failure_rate": row.get("test_optimized_broad_failure_rate"),
                    "heldout_payment_default_rate": row.get("test_optimized_payment_default_rate"),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUTPUT / "rate_sensitivity.csv", index=False)
    primary_policies = primary[1].set_index("held_out_template").selected_policy_id
    higher_policies = higher[1].set_index("held_out_template").selected_policy_id
    changed_folds = int((primary_policies != higher_policies).sum())
    median_change = abs(
        higher[3]["optimized"]["median_annualized_return"]
        - primary[3]["optimized"]["median_annualized_return"]
    )
    rate_material = changed_folds >= 2 or median_change > 0.02
    selected = primary[1]
    all_feasible = bool(selected.feasible.all())
    boundary_count = (
        int(
            (
                selected.selected_debt_multiple.isin([1.5, 3.0])
                | selected.selected_amortisation_rate.isin([0.05, 0.125])
                | selected.selected_cash_sweep.isin([0.4, 0.7])
            ).sum()
        )
        if all_feasible
        else 0
    )
    aggregate = primary[3]
    risk_deterioration = (
        max(
            aggregate["optimized"][metric] - aggregate["reference"][metric]
            for metric in ("broad_failure_rate", "payment_default_rate")
        )
        if aggregate
        else float("inf")
    )
    qualifies_a = (
        all_feasible and boundary_count <= 2 and risk_deterioration <= 0.05 and not rate_material
    )
    classification = "A" if qualifies_a else ("B" if all_feasible else "C")
    labels = {
        "A": "ADMIT AS SYNTHETIC FINANCING-DESIGN RESULT",
        "B": "RETAIN ONLY AS TOY / METHODOLOGICAL DEMONSTRATION",
        "C": "EXCLUDE OPTIMIZATION FROM V3",
    }
    decision = {
        "classification": classification,
        "label": labels[classification],
        "all_folds_feasible": all_feasible,
        "folds_with_any_selected_grid_boundary": boundary_count,
        "heldout_risk_rate_deterioration_max": risk_deterioration,
        "rate_sensitivity_changed_policy_folds": changed_folds,
        "rate_sensitivity_aggregate_median_return_absolute_change": median_change,
        "material_rate_sensitivity": rate_material,
        "reason": (
            "Mechanics validate, but boundary behavior, fixed-rate sensitivity or held-out risk prevents a substantive synthetic financing claim."
            if classification == "B"
            else "Frozen substantive and mechanical admission conditions determine this classification."
        ),
        "real_world_optimality_claim": False,
    }
    dump(OUTPUT / "admission_decision.json", decision)
    summary = {
        "primary_rate": PRIMARY_RATE,
        "n_templates": operators.operator_id.nunique(),
        "n_operating_scenarios": len(scenarios),
        "n_candidates": len(policies),
        "validated_simulation_paths": 2 * (len(policies) + 1) * len(scenarios),
        "primary_aggregate_heldout": aggregate,
        "higher_rate_aggregate_heldout": higher[3],
        "decision": decision,
        "sanity_checks": {
            "unique_operating_scenarios": True,
            "ex_ante_common_candidate_grid": True,
            "each_primary_heldout_scenario_once_per_policy_kind": True,
            "training_only_selection": True,
            "runtime_validation_all_fresh_paths": True,
            "toy_known_optimum_passed": toy["passed"],
        },
    }
    dump(OUTPUT / "summary.json", summary)
    dump(
        OUTPUT / "run_metadata.json",
        {
            "source_commit": source_commit,
            "command": "python -m analysis.run_v3_optimization",
            "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
