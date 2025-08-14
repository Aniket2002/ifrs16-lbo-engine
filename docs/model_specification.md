# IFRS-16 LBO Model Specification

## 1. Balance Sheet Linkage

Net debt under IFRS-16 includes lease liabilities:

$$\text{NetDebt}_t = \text{GrossDebt}_t + \text{LeaseLiab}_t - \text{Cash}_t$$

Where:
- $\text{LeaseLiab}_t$ is the IFRS-16 present value of remaining lease payments using the incremental borrowing rate (IBR)
- We assume no lease modifications or short-term leases unless explicitly noted
- Lease liability amortizes over the lease term (typically 10 years for hospitality)

## 2. Operating Metrics Under IFRS-16

EBITDA is rent-stripped under IFRS-16:
- **Pre-IFRS-16**: EBITDA excludes rent expense
- **IFRS-16**: EBITDA excludes rent (now split into depreciation + interest)
- Conversion: Add back rent expense to pre-IFRS data, estimate IBR for lease liability calculation

## 3. Covenant Formulas

### Interest Coverage Ratio
$$\text{ICR}_t = \frac{\text{EBITDA}_t}{\text{InterestDebt}_t + \text{InterestLease}_t}$$

Hurdle: ICR ≥ 1.8× (annual testing)

### Leverage Ratio  
$$\text{ND/EBITDA}_t = \frac{\text{NetDebt}_t}{\text{EBITDA}_t}$$

Hurdle: ND/EBITDA ≤ 9.0× (annual testing)

### Free Cash Flow Coverage (reported metric)
$$\text{FCF/DS}_t = \frac{\text{FCF}_t}{\text{InterestDebt}_t + \text{AmortizationDebt}_t}$$

**Breach Definition**: Any single year covenant violation triggers breach status.

## 4. Equity Cash Flow Vector

The auditable IRR vector is defined as:

$$\mathbf{c} = (-E_0, E_1, E_2, \ldots, E_T)$$

Where:
- $E_0$ = Initial equity investment (negative outflow)  
- $E_t$ = Annual equity cash flows (years 1 to T-1)
- $E_T$ = Final year equity cash flow + exit equity net of sale costs
- All cash flows are end-of-period convention

Exit equity calculation:
$$\text{ExitEquity} = \text{ExitEV} - \text{FinalNetDebt} - \text{SaleCosts}$$

$$\text{ExitEV} = \text{FinalEBITDA} \times \text{ExitMultiple}$$
