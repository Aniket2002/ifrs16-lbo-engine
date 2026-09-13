# Final financial correctness pass

Branch: `agent/ifrs16-final-waterfall-consistency`.
Baseline: `c349687` (`final fix`), initially clean working tree on
`agent/ifrs16-financial-correctness-v2`. All implementation edits are in canonical
files. The financial-correction work was committed as `88503d2` and pushed to the
remote branch before the final pre-merge rebase.

## Reproduced bugs and corrections

1. **Amortisation refinancing was credited to cash twice.** With cash 10,
   amortisation 30 and minimum cash 25, the old code drew 45, created 20 extra
   cash, then repaid that 20 to the revolver. Cash and revolver debt each appeared
   to close at 25, but the cash reconciliation was short by 20. Draws now have
   separate amortisation and liquidity components; only the latter funds cash.
2. **Operating cash deficits disappeared.** Negative cash before financing was
   reset to zero in the amortisation branch. It is now preserved and funded by
   available liquidity capacity or reported as an unfunded deficit.
3. **Partial refinancing was not applied to mandatory principal.** With cash 10,
   amortisation 30 and capacity 10, the old implementation paid only 10 and
   reported 20 unpaid despite drawing the revolver. It now pays 20 and reports
   10 unpaid, keeping that unpaid amount in outstanding term debt.
4. **Sweeps exceeded outstanding term debt.** With cash 100, debt 40 and
   amortisation 30, a full sweep previously consumed 45 despite only 10 debt
   remaining. The sweep is now capped at 10; ending cash is 60.
5. **Opening cash funded the acquisition and remained in the business.** It is
   now a closing use. Default uses are 1,000 purchase + 30 fees + 40 cash = 1,070;
   debt 450 plus sponsor equity 620 fund those uses. Sponsor equity was 540.
6. **Entry edge cases broke reconciliation.** Excess debt was silently accepted
   by flooring sponsor equity at zero, and return vectors invented a 1e-9 equity
   contribution. Excess debt is now rejected; zero equity remains exactly zero,
   with undefined IRR/MOIC. No excess-debt distribution is inferred.
7. **Benchmark labels omitted payment default.** An isolated payment-default
   scenario with otherwise safe ratios/cash was ignored. The failure union now
   includes payment default; separate booleans retain overlapping failure causes.

No financial correction was made before a failing regression for it. The initial
regression run had 10 failures; the separate benchmark label regression failed
before its classification fix. No parameter tuning was used to improve results.

## Tests and baseline comparison

The initial system-Python attempt had six collection errors (missing package and
scikit-learn imports), and benchmark import failed. Ruff and formatting passed.
The existing `.venv` supplied a working baseline on Windows, Python 3.14.0.
A separate system-Python dependency installation also completed; benchmark
comparisons below consistently use the existing `.venv` environment.

| Check | Baseline | Final |
|---|---:|---:|
| Tests passing | 57 | 76 |
| Aggregate statements covered | 341 / 646 | 330 / 633 |
| Aggregate coverage | 52.79% | 52.13% |
| Full simulation coverage | 98.54% | 100% |
| Ruff check / format check | Pass | Pass |
| 20-scenario smoke / 200-scenario full benchmark | Pass / Pass | Pass / Pass |
| Execution exceptions in either benchmark | 0 | 0 |

The requested `--cov-fail-under=70` command was run and failed only the coverage
threshold; all 76 tests passed. CI retains 52%, which the final suite passes.
Simplifying covered waterfall code reduced the aggregate percentage despite
covering every full-simulation statement. Legacy simulation (10%) and analytic
utilities (46%) account for most remaining gaps. No coverage exclusions were added.
CI's Python 3.10-3.12 matrix was not executed locally.

Nineteen test cases were added to the existing acceptance/integration modules:

- Nine parameterized operating-cash/refinancing/liquidity/default cases, including
  negative cash and exhausted capacity, checking exact cash/debt reconciliation,
  draw components, unpaid principal and limits.
- Debt-capped sweep; multi-year revolver repayment priority and cash carryforward;
  unpaid amortisation retained in debt across defaulted years.
- Three closing-cash cases proving sources equal uses and canonical sponsor entry
  flow, plus zero equity, overfunded entry and empty-return cases.
- Isolated benchmark payment-default classification.

## Benchmark changes (seed 42)

| Metric | Smoke before | Smoke after | Full before | Full after |
|---|---:|---:|---:|---:|
| AUC | 0.805556 | 0.805556 | 0.947222 | 0.947222 |
| Leverage MAE | 0.708418 | 0.797454 | 0.702140 | 0.784173 |
| ICR MAE | 1.926828 | 1.936850 | 2.002317 | 2.011110 |
| False-negative rate | 1.00 | 1.00 | 0.85 | 0.85 |
| False-positive rate | 0.00 | 0.00 | 0.00 | 0.00 |
| Financially failed scenarios | 2 | 2 | 20 | 20 |

The full report now identifies 17 payment defaults, three other combined
reserve/covenant failures, and 180 nonfailures. The smoke report identifies one
payment default, one other combined reserve/covenant failure and 18 nonfailures.
AUC and binary labels did not change for these seeds; financing corrections
changed leverage, subsequent interest and approximation errors. The opening-cash
convention changes sponsor returns, which this benchmark does not report.
Timing measurements vary; no fixed speedup claim is made.

The last full run writes ignored `output/benchmark/benchmark_report.json`.
Temporary before/after smoke/full JSON snapshots are also retained in the Windows
TEMP directory as `ifrs16-baseline-smoke.json`, `ifrs16-baseline-full.json`,
`ifrs16-final-smoke.json` and `ifrs16-final-full.json`.

## Documentation, reproducibility and limitations

README, current model/experiment/research documentation, structure guide,
reproduction instructions and the current paper note now describe generated
metrics and explicit conventions. Obsolete calibration-curve/Brier claims,
unsupported historical success/novelty claims and nonexistent archive contracts
were removed from current documentation. Two unused benchmark expressions were
removed. Makefile targets now reference the existing case-study script and paper
note rather than missing pipeline/config/manuscript paths. The case-study command
is tested; the LaTeX build was not run.

The analytic model's fixed cash, capitalized interest, different timing, inputs
and lease-addition basis are documented without altering its methodology.
Remaining limits include simplified lease/tax schedules, no default recovery or
arrears catch-up, negative cash/exit residuals as diagnostics, no complete input
validation, and incomplete legacy/analytic test coverage. Older paper/archive
snapshots are explicitly historical and were not regenerated. The benchmark is
synthetic, not audited transaction data, and its score is not a calibrated default
probability. Preserve the exact commit and environment with results.

Recommended commit message:
`Fix financing waterfall reconciliation and fund opening cash in entry uses`
