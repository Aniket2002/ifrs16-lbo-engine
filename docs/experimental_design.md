# Experimental Design Specification

## Monte Carlo Protocol

### Random Inputs and Priors

| Parameter | Distribution | Parameters | Bounds | Rationale |
|-----------|-------------|------------|--------|-----------|
| Revenue CAGR | Normal | μ=4%, σ=2% | [0%, 8%] | Historical hospitality sector growth |
| Exit Multiple | Triangular | (8.0x, 9.5x, 11.0x) | [7x, 12x] | LBO exit multiples 2010-2020 |
| Terminal EBITDA Margin | Normal | μ=22%, σ=2% | [15%, 30%] | Operational efficiency bounds |
| Rate Environment | Uniform | Base + [-100bps, +200bps] | - | Interest rate cycle variability |

### Independence Assumptions
- Revenue growth and margin evolution are independent
- Exit multiple independent of operational performance (market-driven)
- Rate environment affects all debt tranches proportionally

### Sampling Protocol
- **N_requested = 400** scenarios using pseudo-random sampling with fixed seed
- **N_effective** reported separately (successful model runs only)
- **Rejection criteria**: Negative EBITDA, infeasible debt capacity, numerical errors

### Success Criterion (Mathematical Definition)
A scenario is classified as "successful" if and only if:

$$\text{Success} = \neg\text{CovenantBreach} \land \text{ExitEquity} > 0 \land \text{IRR} \geq 8\%$$

Where:
- CovenantBreach = (ICR_min < 1.8) ∨ (ND/EBITDA_max > 9.0)
- ExitEquity = ExitEV - FinalNetDebt - SaleCosts
- IRR computed from equity cash flow vector

## Sensitivity Analysis (Sobol Indices)

### Methodology
- **Scheme**: Saltelli sampling for efficient Sobol index computation
- **Base sample size**: n_base = 1,024 (total 2,048 × 4 parameters = 8,192 evaluations)
- **Output**: Equity IRR (single scalar)
- **Indices**: First-order (S₁) and total-effect (Sₜ) indices

### Parameter Vector
Four-dimensional input space:
1. Exit multiple (continuous)
2. Terminal EBITDA margin (continuous)  
3. Revenue growth rate (continuous)
4. Interest rate environment (continuous)

## Deterministic Stress Testing

### Stress Scenarios
1. **Mild Stress**: Rev -4%, Margin -100bps, Exit -1.0x, Rates +150bps
2. **Severe Stress**: Rev -8%, Margin -150bps, Exit -2.0x, Rates +300bps

### Application Method
- **Simultaneous**: All stress factors applied together (worst-case combination)
- **Deterministic**: No randomness in stress scenarios
- **Academic treatment**: Model failures recorded and reported transparently

## Reproducibility Requirements

### Seeds and Versioning
- **Fixed seed**: 42 (all random number generation)
- **Git commit hash**: Embedded in all outputs
- **Software versions**: Python 3.11, NumPy, SciPy versions logged

### Computational Environment
- **Hardware**: Consumer laptop (sufficient for academic reproducibility)
- **Runtime**: ~30 seconds for MC, ~2 minutes for Sobol
- **Complexity**: O(scenarios × years) linear scaling

## Statistical Rigor

### Confidence Intervals
- **Success Rate**: Wilson score interval (95% CI)
- **IRR Percentiles**: Bootstrap CI with 1,000 resamples
- **Reporting**: All point estimates accompanied by uncertainty bounds

### Null Hypothesis Testing
- H₀: IFRS-16 treatment has no effect on covenant metrics
- Alternative specifications tested via ablation studies
