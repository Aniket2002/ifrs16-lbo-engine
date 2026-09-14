"""Run the v3 foundation checks without changing reviewed-v2 methodology."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.run_benchmark import DATA_DIR, _draw_scenario, ensure_synthetic_data
from lbo.full_simulation import (
    FullSimulationAssumptions,
    FullSimulationModel,
    equity_return_metrics,
)
from lbo.validation import SimulationInvariantError, validate_simulation

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "v3"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _return_check() -> dict[str, float]:
    assumptions = FullSimulationAssumptions(
        years=5,
        entry_enterprise_value=100.0,
        transaction_fees_pct=0.0,
        debt_opening=0.0,
        initial_cash=0.0,
    )
    rows = [{"exit_equity": 0.0} for _ in range(4)] + [{"exit_equity": 200.0}]
    actual = equity_return_metrics(rows, assumptions)
    expected_irr = (200.0 / 100.0) ** (1.0 / 5.0) - 1.0
    expected_moic = 200.0 / 100.0
    if not math.isclose(actual["irr"], expected_irr, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("production IRR differs from independent closed form")
    if not math.isclose(actual["moic"], expected_moic, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("production MOIC differs from independent ratio")
    return {
        "expected_irr": expected_irr,
        "actual_irr": actual["irr"],
        "absolute_irr_difference": abs(actual["irr"] - expected_irr),
        "expected_moic": expected_moic,
        "actual_moic": actual["moic"],
        "absolute_moic_difference": abs(actual["moic"] - expected_moic),
        "tolerance": 1e-12,
    }


def _debt_check() -> dict[str, object]:
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
        {
            "year": year,
            "opening_debt": opening,
            "cash_interest": opening * 0.10,
            "mandatory_amortisation": 20.0,
            "closing_debt": max(0.0, opening - 20.0),
        }
        for year, opening in enumerate((100.0, 80.0, 60.0, 40.0, 20.0), start=1)
    ]
    actual = [
        {
            "year": row["year"],
            "opening_debt": row["opening_financial_debt"],
            "cash_interest": row["cash_interest"],
            "mandatory_amortisation": row["actual_mandatory_amortisation"],
            "closing_debt": row["debt_balance"],
        }
        for row in rows
    ]
    if actual != expected:
        raise AssertionError("production debt schedule differs from hand-derived schedule")
    return {"expected": expected, "actual": actual, "exact_match": True}


def _validate_seed42(source_commit: str) -> dict[str, object]:
    ensure_synthetic_data()
    operators = pd.read_csv(DATA_DIR / "operators.csv")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(operators), size=200)
    validated_years = 0
    failure_dir = OUTPUT / "invariant_failures"
    for index, operator_index in enumerate(indices):
        scenario = _draw_scenario(operators.iloc[int(operator_index)], index, rng)
        rows = FullSimulationModel(scenario.simulation).simulate()
        try:
            report = validate_simulation(
                rows,
                scenario.simulation,
                scenario_id=scenario.label,
                source_commit=source_commit,
            )
        except SimulationInvariantError as error:
            failure_dir.mkdir(parents=True, exist_ok=True)
            path = failure_dir / f"{scenario.scenario_id}.json"
            path.write_text(json.dumps(error.to_record(), indent=2) + "\n")
            raise
        validated_years += int(report["validated_years"])
    return {
        "seed": 42,
        "scenarios_validated": 200,
        "years_validated": validated_years,
        "failures": 0,
        "failure_record_directory": "results/v3/invariant_failures",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_commit = _git_sha()
    report = {
        "stage": "v3 foundation validation",
        "source_commit": source_commit,
        "reviewed_v2_head": "a990dd2166f47917781112a7e12bbc16f33605fd",
        "baseline_checkpoint": "655953ffbce56b812f526a6a019b9c71782dc644",
        "independent_return_check": _return_check(),
        "independent_debt_schedule": _debt_check(),
        "runtime_validation": _validate_seed42(source_commit),
        "production_financial_logic_changed": False,
        "later_v3_stages_run": [],
    }
    (OUTPUT / "foundation_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
