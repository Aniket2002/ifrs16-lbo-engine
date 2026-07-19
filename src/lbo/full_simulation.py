from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FullSimulationAssumptions:
    years: int = 5
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
    exit_multiple: float = 8.0


class FullSimulationModel:
    """Full simulation with separate operating, financing, and lease line items."""

    def __init__(self, assumptions: FullSimulationAssumptions | None = None) -> None:
        self.a = assumptions or FullSimulationAssumptions()

    def simulate(self) -> List[Dict[str, Any]]:
        a = self.a
        rows: List[Dict[str, Any]] = []

        revenue = a.revenue_0
        debt = a.debt_opening
        lease = a.lease_opening
        cash = a.initial_cash
        revolver = 0.0
        prev_wc = revenue * a.wc_pct_revenue

        for year in range(1, a.years + 1):
            if year > 1:
                revenue *= 1 + a.revenue_growth

            ebitda = revenue * a.ebitda_margin
            da = revenue * a.da_pct_revenue
            ebit = ebitda - da

            wc = revenue * a.wc_pct_revenue
            delta_wc = wc - prev_wc
            prev_wc = wc

            capex = revenue * a.capex_pct_revenue
            cash_interest = (debt + revolver) * a.cash_interest_rate
            lease_interest = lease * a.lease_interest_rate
            lease_additions = revenue * a.lease_additions_pct_revenue
            lease_principal = lease * a.lease_principal_pct_opening

            taxable_income = ebit - cash_interest - lease_interest
            cash_taxes = max(0.0, taxable_income * a.tax_rate)

            fcf_before_financing = (
                ebitda - capex - delta_wc - cash_taxes - cash_interest - lease_interest
            )
            scheduled_amort = min(a.scheduled_debt_amort, max(0.0, debt))
            cash_sweep = max(0.0, fcf_before_financing) * a.cash_sweep

            total_debt_paydown = min(debt, scheduled_amort + cash_sweep)
            debt -= total_debt_paydown

            cash_after = cash + fcf_before_financing - scheduled_amort
            revolver_draw = 0.0
            if cash_after < 0:
                needed = -cash_after
                revolver_draw = min(a.revolver_limit - revolver, needed)
                revolver += revolver_draw
                cash_after += revolver_draw

            lease = max(0.0, lease + lease_interest + lease_additions - lease_principal)
            cash = max(0.0, cash_after)

            exit_proceeds = 0.0
            if year == a.years:
                exit_proceeds = ebitda * a.exit_multiple - debt - revolver - lease

            rows.append(
                {
                    "year": year,
                    "revenue": revenue,
                    "ebitda": ebitda,
                    "da": da,
                    "ebit": ebit,
                    "cash_taxes": cash_taxes,
                    "capex": capex,
                    "working_capital": wc,
                    "delta_working_capital": delta_wc,
                    "cash_interest": cash_interest,
                    "lease_interest": lease_interest,
                    "lease_principal_payments": lease_principal,
                    "lease_additions": lease_additions,
                    "scheduled_debt_amortisation": scheduled_amort,
                    "cash_sweep": cash_sweep,
                    "revolver_draws": revolver_draw,
                    "debt_balance": debt,
                    "revolver_balance": revolver,
                    "lease_liability": lease,
                    "cash": cash,
                    "exit_proceeds": exit_proceeds,
                }
            )

        return rows
