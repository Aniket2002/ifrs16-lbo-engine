"""Frozen synthetic financing policies and exhaustive grid selection."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, order=True)
class FinancingPolicy:
    debt_multiple: float
    amortisation_rate: float
    cash_sweep: float

    @property
    def policy_id(self) -> str:
        return (
            f"debt_{self.debt_multiple:.2f}x__amort_{self.amortisation_rate:.3f}"
            f"__sweep_{self.cash_sweep:.2f}"
        )

    def to_record(self) -> dict[str, float | str]:
        return {"policy_id": self.policy_id, **asdict(self)}


def candidate_grid() -> list[FinancingPolicy]:
    return [
        FinancingPolicy(debt, amortisation, sweep)
        for debt in (1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00)
        for amortisation in (0.05, 0.075, 0.10, 0.125)
        for sweep in (0.40, 0.50, 0.60, 0.70)
    ]


def limited_liability_returns(
    raw_exit_equity: float, initial_sponsor_equity: float, holding_period: int
) -> dict[str, float]:
    if not np.isfinite([raw_exit_equity, initial_sponsor_equity]).all():
        raise ValueError("Return inputs must be finite")
    if initial_sponsor_equity <= 0:
        raise ValueError("Initial sponsor equity must be positive")
    if holding_period <= 0:
        raise ValueError("Holding period must be positive")
    proceeds = max(0.0, raw_exit_equity)
    moic = proceeds / initial_sponsor_equity
    annualized = moic ** (1.0 / holding_period) - 1.0 if moic > 0 else -1.0
    return {
        "raw_exit_equity": float(raw_exit_equity),
        "sponsor_exit_proceeds": float(proceeds),
        "sponsor_moic": float(moic),
        "sponsor_annualized_return": float(annualized),
    }


def summarize_outcomes(records: pd.DataFrame) -> dict[str, float | int]:
    if records.empty:
        raise ValueError("Cannot summarize empty outcomes")
    required = {
        "sponsor_annualized_return",
        "sponsor_moic",
        "total_equity_loss",
        "broad_failure",
        "payment_default",
        "insolvency",
        "covenant_breach",
        "initial_sponsor_equity",
        "opening_debt",
        "ending_term_debt",
        "max_revolver",
    }
    if not required.issubset(records):
        raise ValueError("Outcome records are missing required fields")
    return {
        "n_scenarios": len(records),
        "median_annualized_return": float(records.sponsor_annualized_return.median()),
        "mean_annualized_return": float(records.sponsor_annualized_return.mean()),
        "median_moic": float(records.sponsor_moic.median()),
        "annualized_return_q10": float(records.sponsor_annualized_return.quantile(0.10)),
        "total_equity_loss_rate": float(records.total_equity_loss.mean()),
        "broad_failure_rate": float(records.broad_failure.mean()),
        "payment_default_rate": float(records.payment_default.mean()),
        "insolvency_rate": float(records.insolvency.mean()),
        "covenant_breach_rate": float(records.covenant_breach.mean()),
        "median_initial_sponsor_equity": float(records.initial_sponsor_equity.median()),
        "median_opening_debt": float(records.opening_debt.median()),
        "median_ending_term_debt": float(records.ending_term_debt.median()),
        "mean_max_revolver": float(records.max_revolver.mean()),
    }


def select_candidate(
    candidate_metrics: pd.DataFrame,
    reference_failure_rate: float,
    reference_payment_default_rate: float,
    *,
    tolerance: float = TIE_TOLERANCE,
) -> pd.Series | None:
    required = {
        "policy_id",
        "debt_multiple",
        "amortisation_rate",
        "cash_sweep",
        "median_annualized_return",
        "broad_failure_rate",
        "payment_default_rate",
    }
    if candidate_metrics.empty or not required.issubset(candidate_metrics):
        raise ValueError("Candidate metrics are empty or incomplete")
    if not np.isfinite(candidate_metrics[list(required - {"policy_id"})].to_numpy()).all():
        raise ValueError("Candidate metrics must be finite")
    feasible = candidate_metrics.loc[
        (candidate_metrics.broad_failure_rate <= reference_failure_rate + tolerance)
        & (candidate_metrics.payment_default_rate <= reference_payment_default_rate + tolerance)
    ].copy()
    if feasible.empty:
        return None
    maximum = feasible.median_annualized_return.max()
    tied = feasible.loc[maximum - feasible.median_annualized_return <= tolerance]
    selected = tied.sort_values(
        [
            "broad_failure_rate",
            "payment_default_rate",
            "debt_multiple",
            "cash_sweep",
            "amortisation_rate",
            "policy_id",
        ],
        ascending=[True, True, True, True, False, True],
        kind="stable",
    )
    return selected.iloc[0]


def solve_known_optimum_toy(debt_candidates) -> dict[str, float | list[dict[str, float]]]:
    candidates = np.asarray(list(debt_candidates), dtype=float)
    if candidates.size == 0 or not np.isfinite(candidates).all():
        raise ValueError("Finite debt candidates are required")
    if (candidates < 0).any() or (candidates > 60).any():
        raise ValueError("Toy debt must be within [0,60]")
    values = (120.0 - candidates) / (100.0 - candidates)
    index = int(np.argmax(values))
    return {
        "selected_debt": float(candidates[index]),
        "selected_moic": float(values[index]),
        "analytic_derivative_numerator": 20.0,
        "evaluations": [
            {"debt": float(debt), "moic": float(moic)}
            for debt, moic in zip(candidates, values, strict=True)
        ],
    }
