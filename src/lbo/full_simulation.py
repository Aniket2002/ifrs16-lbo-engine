from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy_financial as npf


@dataclass
class FullSimulationAssumptions:
    years: int = 5
    entry_enterprise_value: float = 1000.0
    transaction_fees_pct: float = 0.03
    revenue_0: float = 1000.0
    revenue_growth: float = 0.03
    ebitda_margin: float = 0.22
    da_pct_revenue: float = 0.03
    tax_rate: float = 0.25
    capex_pct_revenue: float = 0.04
    wc_pct_revenue: float = 0.02
    cash_interest_rate: float = 0.06
    lease_interest_rate: float = 0.05
    lease_opening: float = 350.0
    lease_additions_pct_revenue: float = 0.01
    lease_principal_pct_opening: float = 0.12
    debt_opening: float = 450.0
    scheduled_debt_amort: float = 30.0
    cash_sweep: float = 0.5
    revolver_limit: float = 200.0
    initial_cash: float = 40.0
    min_cash: float = 25.0
    sale_cost_pct: float = 0.015
    exit_multiple: float = 8.0


class FullSimulationModel:
    """Full simulation with separate operating, financing, and lease line items."""

    def __init__(self, assumptions: FullSimulationAssumptions | None = None) -> None:
        self.a = assumptions or FullSimulationAssumptions()

    def simulate(self) -> list[dict[str, Any]]:
        a = self.a
        rows: list[dict[str, Any]] = []
        sources_and_uses = entry_sources_and_uses(a)

        revenue = a.revenue_0
        debt = a.debt_opening
        lease = a.lease_opening
        cash = a.initial_cash
        revolver = 0.0
        prev_wc = revenue * a.wc_pct_revenue

        for year in range(1, a.years + 1):
            if year > 1:
                revenue *= 1 + a.revenue_growth

            opening_cash = cash
            opening_debt = debt
            opening_revolver = revolver
            opening_lease = lease

            ebitda = revenue * a.ebitda_margin
            da = revenue * a.da_pct_revenue
            ebit = ebitda - da
            negative_ebitda_flag = ebitda <= 0

            wc = revenue * a.wc_pct_revenue
            delta_wc = wc - prev_wc
            prev_wc = wc

            capex = revenue * a.capex_pct_revenue
            cash_interest = (debt + revolver) * a.cash_interest_rate
            lease_interest_cash_payment = lease * a.lease_interest_rate
            lease_additions = revenue * a.lease_additions_pct_revenue
            lease_principal_cash_payment = lease * a.lease_principal_pct_opening

            taxable_income = ebit - cash_interest - lease_interest_cash_payment
            cash_taxes = max(0.0, taxable_income * a.tax_rate)

            operating_cash_generation = (
                ebitda - delta_wc - cash_taxes - cash_interest - lease_interest_cash_payment - capex
            )
            cash_before_financing = (
                opening_cash + operating_cash_generation - lease_principal_cash_payment
            )

            # Separate scheduled vs actual paid amortisation
            scheduled_amortisation = min(a.scheduled_debt_amort, max(0.0, opening_debt))
            revolver_draw = 0.0
            revolver_repayment = 0.0
            funding_deficit = 0.0
            insolvency_flag = False
            unpaid_amortisation = 0.0
            payment_default_flag = False

            # Check if we can pay scheduled amortisation
            cash_available_for_amort = max(0.0, cash_before_financing)
            if cash_available_for_amort >= scheduled_amortisation:
                # Can pay amortisation from operating cash
                actual_mandatory_amortisation = scheduled_amortisation
                cash_after_mandatory = cash_before_financing - actual_mandatory_amortisation
            else:
                # Need to assess if revolver can cover shortfall
                amortisation_shortfall = scheduled_amortisation - cash_available_for_amort
                available_revolver = max(0.0, a.revolver_limit - opening_revolver)

                if available_revolver >= amortisation_shortfall:
                    # Revolver can cover the gap
                    actual_mandatory_amortisation = scheduled_amortisation
                    revolver_draw += amortisation_shortfall
                    cash_after_mandatory = 0.0
                else:
                    # Revolver insufficient; skip amortisation, track as unpaid
                    actual_mandatory_amortisation = max(0.0, cash_before_financing)
                    unpaid_amortisation = scheduled_amortisation - actual_mandatory_amortisation
                    payment_default_flag = True
                    cash_after_mandatory = 0.0
                    revolver_draw = available_revolver

            # Minimum cash check after amortisation attempt
            if cash_after_mandatory < a.min_cash:
                required_cash = a.min_cash - cash_after_mandatory
                additional_revolver = min(
                    max(0.0, a.revolver_limit - opening_revolver - revolver_draw), required_cash
                )
                revolver_draw += additional_revolver
                cash_after_draw = cash_after_mandatory + revolver_draw
                funding_deficit = max(0.0, a.min_cash - cash_after_draw)
                insolvency_flag = funding_deficit > 0
            else:
                cash_after_draw = cash_after_mandatory

            if (
                not insolvency_flag
                and opening_revolver + revolver_draw > 0
                and cash_after_draw > a.min_cash
            ):
                revolver_repayment = min(
                    opening_revolver + revolver_draw, cash_after_draw - a.min_cash
                )
                cash_after_draw -= revolver_repayment

            optional_sweep_base = max(0.0, cash_after_draw - a.min_cash)
            proposed_sweep = max(0.0, cash_before_financing) * a.cash_sweep
            cash_sweep = min(optional_sweep_base, proposed_sweep)

            ending_cash = cash_after_draw - cash_sweep
            debt = max(0.0, opening_debt - actual_mandatory_amortisation - cash_sweep)
            revolver = max(0.0, opening_revolver + revolver_draw - revolver_repayment)
            lease = max(0.0, opening_lease + lease_additions - lease_principal_cash_payment)
            cash = ending_cash

            exit_enterprise_value = 0.0
            exit_equity = 0.0
            if year == a.years:
                exit_enterprise_value = ebitda * a.exit_multiple
                sale_costs = exit_enterprise_value * a.sale_cost_pct
                exit_equity = (
                    exit_enterprise_value - debt - revolver - lease + ending_cash - sale_costs
                )

            rows.append(
                {
                    "year": year,
                    "opening_cash": opening_cash,
                    "opening_financial_debt": opening_debt,
                    "opening_revolver": opening_revolver,
                    "opening_lease_liability": opening_lease,
                    "entry_enterprise_value": sources_and_uses["entry_enterprise_value"],
                    "transaction_fees": sources_and_uses["transaction_fees"],
                    "purchase_price": sources_and_uses["purchase_price"],
                    "total_uses": sources_and_uses["total_uses"],
                    "debt_sources": sources_and_uses["debt_sources"],
                    "cash_sources": sources_and_uses["cash_sources"],
                    "sponsor_equity": sources_and_uses["sponsor_equity"],
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "da": da,
                    "ebit": ebit,
                    "negative_ebitda_flag": negative_ebitda_flag,
                    "operating_cash_generation": operating_cash_generation,
                    "cash_before_financing": cash_before_financing,
                    "cash_taxes": cash_taxes,
                    "capex": capex,
                    "working_capital": wc,
                    "delta_working_capital": delta_wc,
                    "cash_interest": cash_interest,
                    "lease_interest_cash_payment": lease_interest_cash_payment,
                    "lease_interest": lease_interest_cash_payment,
                    "lease_principal_cash_payment": lease_principal_cash_payment,
                    "lease_additions": lease_additions,
                    "scheduled_debt_amortisation": scheduled_amortisation,
                    "actual_mandatory_amortisation": actual_mandatory_amortisation,
                    "unpaid_amortisation": unpaid_amortisation,
                    "payment_default_flag": payment_default_flag,
                    "cash_after_mandatory_amortisation": cash_after_mandatory,
                    "revolver_draw": revolver_draw,
                    "revolver_repayment": revolver_repayment,
                    "cash_sweep": cash_sweep,
                    "funding_deficit": funding_deficit,
                    "insolvency_flag": insolvency_flag,
                    "cash_after_financing": ending_cash,
                    "debt_balance": debt,
                    "revolver_balance": revolver,
                    "lease_liability": lease,
                    "cash": cash,
                    "ending_cash": ending_cash,
                    "exit_enterprise_value": exit_enterprise_value,
                    "exit_equity": exit_equity,
                }
            )

        return rows


def entry_sources_and_uses(assumptions: FullSimulationAssumptions) -> dict[str, float]:
    purchase_price = float(assumptions.entry_enterprise_value)
    transaction_fees = purchase_price * float(assumptions.transaction_fees_pct)
    debt_sources = float(assumptions.debt_opening)
    cash_sources = float(assumptions.initial_cash)
    total_uses = purchase_price + transaction_fees
    sponsor_equity = max(0.0, total_uses - debt_sources - cash_sources)

    return {
        "entry_enterprise_value": purchase_price,
        "purchase_price": purchase_price,
        "transaction_fees": transaction_fees,
        "total_uses": total_uses,
        "debt_sources": debt_sources,
        "cash_sources": cash_sources,
        "sponsor_equity": sponsor_equity,
    }


def equity_cash_flow_vector(
    rows: list[dict[str, Any]], assumptions: FullSimulationAssumptions
) -> list[float]:
    if not rows:
        return []

    initial_equity = max(1e-9, entry_sources_and_uses(assumptions)["sponsor_equity"])
    return [-initial_equity] + [0.0] * (len(rows) - 1) + [float(rows[-1]["exit_equity"])]


def equity_return_metrics(
    rows: list[dict[str, Any]], assumptions: FullSimulationAssumptions
) -> dict[str, Any]:
    cash_flows = equity_cash_flow_vector(rows, assumptions)
    if len(cash_flows) < 2:
        return {
            "equity_cash_flow_vector": cash_flows,
            "irr": float("nan"),
            "moic": float("nan"),
            "initial_equity": float("nan"),
            "exit_equity": float("nan"),
            "sponsor_equity": float("nan"),
            "entry_sources_and_uses": {},
        }

    initial_equity = abs(cash_flows[0])
    exit_equity = cash_flows[-1]
    irr = npf.irr(cash_flows)
    moic = exit_equity / initial_equity if initial_equity > 0 else float("nan")
    sources_and_uses = entry_sources_and_uses(assumptions)

    return {
        "equity_cash_flow_vector": cash_flows,
        "irr": float(irr) if irr is not None and np.isfinite(irr) else float("nan"),
        "moic": float(moic),
        "initial_equity": float(initial_equity),
        "exit_equity": float(exit_equity),
        "sponsor_equity": float(sources_and_uses["sponsor_equity"]),
        "entry_sources_and_uses": sources_and_uses,
    }
