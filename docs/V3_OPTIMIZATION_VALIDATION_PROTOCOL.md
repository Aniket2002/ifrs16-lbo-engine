# Frozen synthetic financing-optimization protocol

Specified before final financing outcomes are calculated. The reviewed v2
benchmark is not a valid return-optimization dataset: it stipulates entry EV as
7.5 times revenue while the engine calculates exit EV as 8.0 times EBITDA. No
repository evidence documents comparable entry/exit valuation bases or an
intentional revenue-to-EBITDA return convention. V2 itself disclaims equity-return
optimization. Preserve v2 unchanged; this finding does not affect its ranking or
covenant results.

The separate v3 experiment uses the five existing synthetic operator archetypes,
100 operating paths per template and seed 314159. Known entry state is base
revenue, base EBITDA, opening lease, opening cash and the archetype's lease
roll-forward assumptions. Entry EV is 7.5 times known entry EBITDA; exit EV is 8.0
times final EBITDA; transaction fees are 3% of entry EV; holding period is five
years. Initial minimum cash is half opening cash and revolver capacity is 0.75
times known entry EBITDA. Primary term borrowing costs 6%; lease interest remains
5%. These are stipulated synthetic conventions, not market estimates.

Operating regimes are drawn with probabilities 0.5 base, 0.3 downside and 0.2
distressed. Draw base growth from each archetype's recorded Normal mean/SD and
margin from Normal(archetype margin mean, 0.02). Downside subtracts 0.04 growth and
0.03 margin; distressed subtracts 0.08 and 0.06. Clamp growth to [-0.12,0.18] and
margin to [0.08,0.35]. Financing, cash and lease opening balances are never changed
by regime. Every path/candidate uses the same frozen operating draws.

The decision vector is common across templates and fixed ex ante:

- opening term debt / known entry EBITDA: 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00;
- annual scheduled amortisation / opening term debt: 0.05, 0.075, 0.10, 0.125;
- cash sweep: 0.40, 0.50, 0.60, 0.70.

This gives 112 candidates. Opening debt is multiple times known entry EBITDA;
scheduled annual amortisation is rate times opening debt. Debt must be strictly
below total entry uses, leaving positive sponsor equity. Exit multiple, operating
inputs, taxes, lease assumptions, failure definition, covenants, scenario weights
and rates are not decisions.

The reference policy is the existing synthetic archetype financing: recorded
opening debt, recorded cash sweep and the benchmark engine's unchanged annual
scheduled amortisation of 30 currency units. Report debt/EBITDA and 30/opening debt
for each archetype. It is a synthetic baseline, not market practice.

For raw engine exit equity E and positive initial sponsor equity S, analysis-layer
limited-liability proceeds are max(0,E), MOIC is max(0,E)/S, and annualized return
is MOIC^(1/5)-1. Zero recovery is MOIC 0 and annualized return -1. Preserve raw E.
Reject zero/nonpositive sponsor equity and debt at or above entry uses. The engine
continues after default and is not a recovery model; both defaults and mechanically
calculated returns are retained and this limitation must constrain interpretation.

For each leave-one-template-out fold, evaluate every candidate on only the four
training templates. A candidate is feasible when both its broad financial-failure
rate and payment-default rate are no greater than the reference policy's rates on
the identical training paths. Broad failure is the unchanged union of nonpositive
EBITDA, payment default, reserve deficit, leverage above 6.0 or ICR below 1.8.
Maximize training median limited-liability annualized sponsor return. Report mean
return, median MOIC, return 10th percentile, total-equity-loss probability,
insolvency/covenant rates and capital structure without optimizing them.

Among objectives equal within absolute 1e-12, choose lower broad failure rate,
then lower payment-default rate, lower debt multiple, lower cash sweep, and higher
amortisation rate. An infeasible fold remains infeasible. Apply the selected policy
unchanged to the held-out template and aggregate held-out rows exactly once. Never
use held-out results to change the grid, objective, constraints or tie rule.

The known-optimum toy has entry uses 100, exit EV 120, no interest, lease,
amortisation or uncertainty, and debt in [0,60]. Its independently derived MOIC
(120-D)/(100-D) has derivative 20/(100-D)^2 > 0, so the required optimum is D=60.

Rate sensitivity repeats the entire training-only selection at a fixed 7.5% term
rate, including same-rate reference constraints, without optimizing the rate. A
change in any selected policy component in at least two of five folds or a change
in aggregate median held-out return exceeding two percentage points is classified
as material rate sensitivity. Admission A additionally requires no selected debt
or other decision to sit at a grid boundary in more than two folds, all folds
feasible, and no held-out broad/default-rate increase above reference exceeding
five percentage points. If mechanics pass but these substantive criteria fail,
choose B. Choose C for incoherent mechanics, persistent infeasibility or no
defensible information. No real-world optimality or historical IRR-uplift claim.

Before publication require independent toy, feasibility, tie, boundary, risk,
determinism, held-out independence, row-order, positive-equity and entry-use tests;
runtime validation of every freshly simulated path; unchanged prior v3 artifacts
and financial engine; Ruff; focused prior-stage tests; and full suite/coverage.
