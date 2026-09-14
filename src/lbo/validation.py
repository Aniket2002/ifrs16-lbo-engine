"""Side-effect-free runtime checks for full-simulation financial paths."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence


def _json_value(value: Any) -> Any:
    """Convert scalar/container state to JSON-compatible built-in values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass
class SimulationInvariantError(AssertionError):
    """Structured invariant failure with enough context to reproduce the row."""

    invariant: str
    scenario_id: str | None
    year: int | None
    expected: Any
    actual: Any
    tolerance: float
    row: Mapping[str, Any]
    previous_row: Mapping[str, Any] | None
    assumptions: Mapping[str, Any]
    source_commit: str

    def __post_init__(self) -> None:
        message = (
            f"{self.invariant} failed for scenario={self.scenario_id!r}, year={self.year}: "
            f"expected {self.expected!r}, actual {self.actual!r}, tolerance={self.tolerance}"
        )
        AssertionError.__init__(self, message)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable failure record."""
        return _json_value(
            {
                "scenario_id": self.scenario_id,
                "year": self.year,
                "invariant": self.invariant,
                "assumptions": self.assumptions,
                "row": self.row,
                "previous_row": self.previous_row,
                "expected": self.expected,
                "actual": self.actual,
                "tolerance": self.tolerance,
                "source_commit": self.source_commit,
            }
        )


def _assumption_state(assumptions: Any) -> dict[str, Any]:
    if is_dataclass(assumptions):
        return _json_value(asdict(assumptions))
    if isinstance(assumptions, Mapping):
        return _json_value(assumptions)
    return _json_value(vars(assumptions))


def validate_simulation(
    rows: Sequence[Mapping[str, Any]],
    assumptions: Any,
    *,
    scenario_id: str | None = None,
    tolerance: float = 1e-8,
    source_commit: str = "unknown",
) -> dict[str, Any]:
    """Validate every annual row without changing the simulation or its outputs.

    Zero EBITDA or zero interest is recorded as an explicitly undefined ratio
    denominator state; it is not itself an accounting-invariant failure.
    """
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")

    assumption_state = _assumption_state(assumptions)
    zero_ebitda_years: list[int] = []
    zero_interest_years: list[int] = []

    def fail(
        invariant: str,
        row: Mapping[str, Any],
        previous: Mapping[str, Any] | None,
        expected: Any,
        actual: Any,
    ) -> None:
        raise SimulationInvariantError(
            invariant=invariant,
            scenario_id=scenario_id,
            year=int(row["year"]) if "year" in row else None,
            expected=_json_value(expected),
            actual=_json_value(actual),
            tolerance=tolerance,
            row=_json_value(row),
            previous_row=_json_value(previous) if previous is not None else None,
            assumptions=assumption_state,
            source_commit=source_commit,
        )

    def close(
        invariant: str,
        row: Mapping[str, Any],
        previous: Mapping[str, Any] | None,
        expected: float,
        actual: float,
    ) -> None:
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
            fail(invariant, row, previous, expected, actual)

    def at_most(
        invariant: str,
        row: Mapping[str, Any],
        previous: Mapping[str, Any] | None,
        upper: float,
        actual: float,
    ) -> None:
        if float(actual) > float(upper) + tolerance:
            fail(invariant, row, previous, f"<= {upper}", actual)

    previous: Mapping[str, Any] | None = None
    for row in rows:
        year = int(row["year"])
        numeric_values = {
            key: value
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        invalid = {key: value for key, value in numeric_values.items() if not math.isfinite(value)}
        if invalid:
            fail("finite_financial_values", row, previous, "all numeric row values finite", invalid)

        close("entry_sources_equal_uses", row, previous, row["total_uses"], row["total_sources"])
        component_sources = row["debt_sources"] + row["sponsor_equity"] + row["cash_sources"]
        component_uses = row["purchase_price"] + row["transaction_fees"] + row["opening_cash_use"]
        close("entry_component_reconciliation", row, previous, component_uses, component_sources)

        # Check opening state before within-year identities so a corrupted roll-forward
        # is attributed to the boundary where it occurred.
        if previous is None:
            close("initial_cash", row, previous, assumptions.initial_cash, row["opening_cash"])
            close(
                "initial_term_debt",
                row,
                previous,
                assumptions.debt_opening,
                row["opening_financial_debt"],
            )
            close("initial_revolver", row, previous, 0.0, row["opening_revolver"])
            close(
                "initial_lease",
                row,
                previous,
                assumptions.lease_opening,
                row["opening_lease_liability"],
            )
        else:
            close(
                "opening_cash_roll_forward",
                row,
                previous,
                previous["ending_cash"],
                row["opening_cash"],
            )
            close(
                "opening_term_debt_roll_forward",
                row,
                previous,
                previous["debt_balance"],
                row["opening_financial_debt"],
            )
            close(
                "opening_revolver_roll_forward",
                row,
                previous,
                previous["revolver_balance"],
                row["opening_revolver"],
            )
            close(
                "opening_lease_roll_forward",
                row,
                previous,
                previous["lease_liability"],
                row["opening_lease_liability"],
            )

        close(
            "revolver_draw_components",
            row,
            previous,
            row["revolver_draw_for_amortisation"] + row["revolver_draw_for_liquidity"],
            row["revolver_draw"],
        )
        for field in (
            "revolver_draw_for_amortisation",
            "revolver_draw_for_liquidity",
            "revolver_draw",
            "revolver_repayment",
        ):
            if row[field] < -tolerance:
                fail(f"{field}_nonnegative", row, previous, ">= 0", row[field])
        at_most(
            "revolver_draw_capacity",
            row,
            previous,
            max(0.0, assumptions.revolver_limit - row["opening_revolver"]),
            row["revolver_draw"],
        )
        cash_funded_amortisation = (
            row["actual_mandatory_amortisation"] - row["revolver_draw_for_amortisation"]
        )
        close(
            "cash_before_financing_reconciliation",
            row,
            previous,
            row["opening_cash"]
            + row["operating_cash_generation"]
            - row["lease_principal_cash_payment"],
            row["cash_before_financing"],
        )
        close(
            "cash_after_mandatory_reconciliation",
            row,
            previous,
            row["cash_before_financing"] - cash_funded_amortisation,
            row["cash_after_mandatory_amortisation"],
        )
        if cash_funded_amortisation < -tolerance:
            fail(
                "cash_funded_amortisation_nonnegative",
                row,
                previous,
                ">= 0",
                cash_funded_amortisation,
            )
        expected_cash = (
            row["opening_cash"]
            + row["operating_cash_generation"]
            - row["lease_principal_cash_payment"]
            - cash_funded_amortisation
            + row["revolver_draw_for_liquidity"]
            - row["revolver_repayment"]
            - row["cash_sweep"]
        )
        close("cash_reconciliation", row, previous, expected_cash, row["ending_cash"])
        close("cash_alias_consistency", row, previous, row["ending_cash"], row["cash"])
        close(
            "cash_after_financing_consistency",
            row,
            previous,
            row["ending_cash"],
            row["cash_after_financing"],
        )

        expected_debt = max(
            0.0,
            row["opening_financial_debt"]
            - row["actual_mandatory_amortisation"]
            - row["cash_sweep"],
        )
        close("term_debt_reconciliation", row, previous, expected_debt, row["debt_balance"])
        if row["debt_balance"] < -tolerance:
            fail("term_debt_floor", row, previous, ">= 0", row["debt_balance"])

        expected_revolver = max(
            0.0, row["opening_revolver"] + row["revolver_draw"] - row["revolver_repayment"]
        )
        close("revolver_reconciliation", row, previous, expected_revolver, row["revolver_balance"])
        if row["revolver_balance"] < -tolerance:
            fail("revolver_floor", row, previous, ">= 0", row["revolver_balance"])
        at_most(
            "revolver_capacity",
            row,
            previous,
            assumptions.revolver_limit,
            row["revolver_balance"],
        )
        at_most(
            "revolver_repayment_limit",
            row,
            previous,
            row["opening_revolver"] + row["revolver_draw"],
            row["revolver_repayment"],
        )

        for field in ("actual_mandatory_amortisation", "unpaid_amortisation"):
            if row[field] < -tolerance:
                fail(f"{field}_nonnegative", row, previous, ">= 0", row[field])
        close(
            "scheduled_amortisation_policy",
            row,
            previous,
            min(assumptions.scheduled_debt_amort, max(0.0, row["opening_financial_debt"])),
            row["scheduled_debt_amortisation"],
        )
        close(
            "amortisation_consistency",
            row,
            previous,
            row["scheduled_debt_amortisation"],
            row["actual_mandatory_amortisation"] + row["unpaid_amortisation"],
        )
        expected_default = row["unpaid_amortisation"] > tolerance
        if bool(row["payment_default_flag"]) != expected_default:
            fail(
                "payment_default_consistency",
                row,
                previous,
                expected_default,
                bool(row["payment_default_flag"]),
            )

        expected_funding_deficit = max(0.0, assumptions.min_cash - row["ending_cash"])
        close(
            "funding_deficit_consistency",
            row,
            previous,
            expected_funding_deficit,
            row["funding_deficit"],
        )
        # Production deliberately uses the exact stored positive-deficit rule.
        # Do not reinterpret a small recorded shortfall as no shortfall here.
        expected_insolvency = row["funding_deficit"] > 0.0
        if bool(row["insolvency_flag"]) != expected_insolvency:
            fail(
                "insolvency_flag_consistency",
                row,
                previous,
                expected_insolvency,
                bool(row["insolvency_flag"]),
            )

        if row["cash_sweep"] < -tolerance:
            fail("sweep_nonnegative", row, previous, ">= 0", row["cash_sweep"])
        remaining_debt = max(
            0.0, row["opening_financial_debt"] - row["actual_mandatory_amortisation"]
        )
        at_most("sweep_remaining_term_debt", row, previous, remaining_debt, row["cash_sweep"])
        cash_after_draw_before_repayment = (
            row["cash_after_mandatory_amortisation"] + row["revolver_draw_for_liquidity"]
        )
        eligible_excess_cash = max(
            0.0,
            cash_after_draw_before_repayment - row["revolver_repayment"] - assumptions.min_cash,
        )
        at_most(
            "sweep_eligible_excess_cash", row, previous, eligible_excess_cash, row["cash_sweep"]
        )
        sweep_rate_limit = max(0.0, row["cash_before_financing"]) * assumptions.cash_sweep
        at_most("sweep_rate_limit", row, previous, sweep_rate_limit, row["cash_sweep"])

        expected_lease = max(
            0.0,
            row["opening_lease_liability"]
            + row["lease_additions"]
            - row["lease_principal_cash_payment"],
        )
        close("lease_reconciliation", row, previous, expected_lease, row["lease_liability"])
        if row["lease_liability"] < -tolerance:
            fail("lease_liability_floor", row, previous, ">= 0", row["lease_liability"])

        interest_denominator = row["cash_interest"] + row["lease_interest"]
        if abs(row["ebitda"]) <= tolerance:
            zero_ebitda_years.append(year)
        if abs(interest_denominator) <= tolerance:
            zero_interest_years.append(year)
        previous = row

    return {
        "scenario_id": scenario_id,
        "validated_years": len(rows),
        "zero_ebitda_years": zero_ebitda_years,
        "zero_interest_denominator_years": zero_interest_years,
        "ratio_policy": "zero denominators recorded; ratios must be handled explicitly downstream",
    }
