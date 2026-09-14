# V3 Bayesian validation and admission decision

## Decision

**C — EXCLUDE FROM V3.** The corrected population model shows good synthetic
recovery but fails the frozen technical gate on convergence/effective sampling and
material prior sensitivity. Independently, all ten named-firm calibration rows are
unverified/stipulated inputs, so they cannot support real-world Bayesian
calibration. No posterior samples were connected to the LBO simulator, threshold
experiment, optimization, manuscript results or publication claims.

The protocol, corrected code, validation runner and tests were committed as
`4b7179367d7b69c6870f530f24a283a83bfd0c8f` before final seeds were run. The
machine-readable outputs in `results/v3/bayesian_validation/` record exact input
and protocol hashes, environment, package versions, sampler settings and seeds.

## Implementation audit and corrections

The old `_fit_map_laplace()` was neither MAP nor Laplace: it computed sample means
and population standard deviations. It performed no posterior optimization,
Hessian calculation, covariance approximation or uncertainty inference. It is now
named `empirical_moments`; the invalid `map` label is rejected.

Requested `method="mcmc"` previously fell through to empirical moments whenever
PyMC was unavailable. It now either performs MCMC or raises a clear `ImportError`.
Exports record the actual fit method and sampler rather than PyMC availability.

The old `generate_posterior_predictive()` sampled new values conditional on ten
posterior-mean hyperparameters. It included conditional between-firm variation but
discarded uncertainty in population means and scales. It therefore was not a full
posterior predictive distribution. The corrected
`generate_population_predictive_from_trace()` selects actual joint posterior
draws and generates one new population observation per draw.
`generate_empirical_parameter_draws()` is separately and accurately named. The
compatibility dispatcher selects behavior from the method actually fitted.

Region, rating and brand tier were loaded but never used. They remain metadata;
the code no longer advertises implemented covariate effects. There is no modeled
cross-parameter dependence.

The former PyMC likelihood placed latent Normal firm parameters before a fixed
0.01 observation error for all five variables, despite their different units and
without measurement-error evidence. Deterministic clipping of latent values
created flat regions and exact boundary mass. With one value per firm, latent
heterogeneity and measurement noise were weakly identified. The correction removes
that unsupported layer and uses direct population likelihoods with proper
truncation. Input checks now reject missing variables, nonfinite/out-of-domain
values and nonpositive lease multiples before logarithms.

## Data provenance

The repository provides ten rows named Accor, Marriott, Hilton, IHG, Hyatt,
Wyndham, Choice Hotels, Extended Stay, La Quinta and Red Roof. It provides no
variable-level source documents, dates, definitions, extraction calculations,
reconstruction notes or measurement-error metadata for this CSV. Historical prose
mentions public 10-K summaries, but does not link any row or value to evidence. A
company name is not provenance.

| Field | Classification | Enters model? |
|---|---|---:|
| revenue_growth | Unverified/stipulated | Yes |
| terminal_margin | Unverified/stipulated | Yes |
| lease_multiple | Unverified/stipulated | Yes, on log scale |
| senior_rate | Unverified/stipulated | Yes |
| mezz_rate | Unverified/stipulated | Yes |
| name, region, rating, brand_tier | Unverified metadata | No |

The dataset has one cross-sectional observation per firm and quantity, not a
longitudinal panel. Fitting it cannot establish empirical measurement validity,
temporal stability, covariate effects or causal structure.

## Model tested

For firm `i`, the five dimensions are independent conditional on their population
parameters, and firms are exchangeable:

```text
mu_g ~ Normal(0.04, 0.02)       sigma_g ~ HalfNormal(0.015)
g_i  ~ TruncatedNormal(mu_g, sigma_g, lower=0, upper=0.15)

mu_m ~ Normal(0.25, 0.05)       sigma_m ~ HalfNormal(0.03)
m_i  ~ TruncatedNormal(mu_m, sigma_m, lower=0.10, upper=0.40)

mu_L ~ Normal(1.20, 0.30)       sigma_L ~ HalfNormal(0.20)
log(L_i) ~ Normal(mu_L, sigma_L)

mu_s ~ Normal(0.06, 0.01)       sigma_s ~ HalfNormal(0.008)
s_i  ~ TruncatedNormal(mu_s, sigma_s, lower=0.02, upper=0.12)

mu_z ~ Normal(0.06, 0.01)       sigma_z ~ HalfNormal(0.008)
z_i  ~ TruncatedNormal(mu_z, sigma_z, lower=0.02, upper=0.12)
```

Growth, margin and rates are fractions; lease multiple is a positive ratio and is
modeled in natural-log units. The stated bounds are modeling assumptions, not
empirically verified limits. Population scales can be weakly identified with ten
observations. The common rate prior centers mezzanine rates at 6%, far below the
stipulated sample mean of 9.18%; sensitivity confirms this matters.

## Prior predictive validation

Twenty thousand seed-701 draws use the frozen priors. Proper truncated
distributions produce no exact-boundary mass. Quantiles are distributions for one
new population observation after integrating over hyperpriors.

| Dimension | Mean | SD | 5th–95th percentile | Plausible frequency |
|---|---:|---:|---:|---:|
| Revenue growth | 0.0415 | 0.0226 | 0.0062–0.0805 | 1.0000 |
| Terminal margin | 0.2496 | 0.0557 | 0.1556–0.3417 | 1.0000 |
| Lease multiple | 3.5524 | 1.3328 | 1.8651–5.9925 | 0.9906 |
| Senior rate | 0.0600 | 0.0125 | 0.0395–0.0807 | 1.0000 |
| Mezzanine rate | 0.0600 | 0.0127 | 0.0393–0.0808 | 1.0000 |

The priors avoid impossible bounded values and clipping mass and are not
implausibly narrow for most stated domains. The mezzanine prior is poorly aligned
with the stipulated values; this was not tuned after inspection. About 0.945% of
lease draws fall outside the stipulated plausibility range [1,8].

## Synthetic recovery

The frozen experiment used 18 independent datasets: three seeds for central,
low-variance, high-variance, near-boundary, n=10 and n=100 designs. Other designs
used n=30. Every joint fit used two chains, 600 tune and 600 retained draws per
chain, target acceptance 0.95 and nutpie NUTS. All ten population hyperparameters
were evaluated in each run, giving 180 evaluations.

Overall 90% interval coverage was **0.9389**. Mean-parameter relative RMSE was
**0.0804** and scale-parameter relative RMSE **0.1199**. Absolute mean signed
relative bias by parameter ranged from 0.19% to 8.31%, within the frozen 75%
ceiling. Coverage by design was:

| Design | 90% coverage | Max R-hat | Min bulk ESS | Min tail ESS |
|---|---:|---:|---:|---:|
| Central | 0.9667 | 1.0062 | 1120.7 | 584.6 |
| Low variance | 0.9667 | 1.0114 | 921.0 | 632.0 |
| High variance | 0.9000 | 1.0067 | 710.7 | 570.6 |
| Near boundary | 0.9333 | 1.0146 | 391.5 | 176.3 |
| Small sample (n=10) | 0.9667 | 1.0090 | 547.3 | 380.4 |
| Larger sample (n=100) | 0.9000 | 1.0109 | 1451.8 | 733.0 |

Parameter coverage ranged from 0.7778 for log-lease population location to 1.0
for margin location/scale and both rate-scale families. These results demonstrate
substantial technical recovery ability. They do not override failed frozen gates.

## MCMC diagnostics

There were **zero divergences** and **zero tree-depth warnings** across all 18
recovery runs. Every bulk ESS exceeded 391.5. The near-boundary seed-1102 margin
location had R-hat 1.0146 and tail ESS 176.3; its margin scale had R-hat 1.0132.
Low-variance rate location and larger-sample rate scale produced additional R-hat
values slightly above 1.01. Thus the all-parameter R-hat and tail-ESS criteria fail.
Good diagnostics elsewhere do not establish identification; increasing draws after
seeing these results would violate this run's frozen decision rule.

## Posterior predictive findings

Predictions integrate over actual posterior hyperparameter draws. Across all 18
synthetic recovery datasets, every known generating location fell inside its
corresponding population-predictive 90% interval. Predictive summaries contain no
exact boundary mass. This supports corrected predictive mechanics, while the wide
population intervals make location coverage an intentionally modest check.

For the stipulated data, posterior-predictive means versus observed means were
0.0294/0.0293 growth, 0.2706/0.2720 margin, 3.3167/3.3000 lease multiple,
0.0530/0.0527 senior rate and 0.0895/0.0918 mezzanine rate. Predictive SDs were
0.00685, 0.04913, 0.48487, 0.00619 and 0.00919, versus observed SDs 0.00560,
0.04686, 0.38586, 0.00519 and 0.00774. This is an in-sample check with unknown
truth, not predictive improvement. A ten-row leave-one-firm-out exercise would
offer little precision and was not used to claim value.

## Prior sensitivity and information content

One pre-specified alternative doubled every mean-prior and scale-prior SD. Maximum
posterior shift was **1.0807 baseline posterior SD** for terminal-margin
heterogeneity; mezzanine-rate location shifted **0.6485 SD**. Both exceed the
frozen 0.5-SD materiality threshold. Other shifts ranged from 0.0096 to 0.2635 SD.

Most population locations and several scales are visibly data-informed, as shown
by stable posterior means and synthetic recovery. Bayesian uncertainty is not
uniformly data-informed: margin heterogeneity is prior-sensitive, and the
mezzanine location responds materially to widening a prior whose center conflicts
with the stipulated sample. Ten observations are insufficient for robust
real-world heterogeneity claims.

## Gate and limitations

Seven of nine frozen recovery/diagnostic criteria pass. R-hat and tail ESS fail;
the separate prior-sensitivity gate also fails. Data provenance independently
fails empirical validity. Classification B required both technical-gate passage
and absence of material prior sensitivity, so the result is classification C.

This is a conservative exclusion decision, not a claim that Bayesian population
modeling is intrinsically unsuitable. A future reconsideration would require
variable-level sourced data, defensible measurement definitions, a prior for
mezzanine rates grounded independently of the validation sample, more firms or
longitudinal observations, and a newly pre-specified validation protocol. It must
not reuse these final seeds as unseen confirmation data.

All 14 Bayesian tests pass, including explicit missing-PyMC behavior, truthful
method provenance, trace-integrated predictions, determinism, domain/schema
validation and a population-recovery smoke case. Together, the Bayesian, template
and foundation selections contain 62 passing tests. The full suite contains 138
passing tests with 57.95% statement coverage; full simulation remains 100%.
