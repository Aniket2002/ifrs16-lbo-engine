# Frozen Bayesian validation protocol

Specified before final recovery seeds are evaluated. The audited ten-firm CSV has
no variable-level source, date, definition, reconstruction or measurement-error
metadata in the repository. Every numeric field is therefore classified as an
unverified/stipulated calibration input. Names and categorical metadata do not
change that classification. Classification A (main-results admission) is
unavailable without empirical provenance and a validated additional contribution.

The corrected model treats each firm's single cross-sectional value as one draw
from an independent population distribution. Revenue growth, terminal margin,
senior rate and mezzanine rate use truncated Normal likelihoods at their documented
bounds. Log lease multiple uses a Normal likelihood. Population means retain the
existing Normal priors and population standard deviations retain the existing
HalfNormal priors. There is no firm-level latent/measurement-error layer because
measurement error is unknown and one observation per firm cannot identify both
latent heterogeneity and error. Region, rating and brand tier remain unused
metadata. This is a five-dimension independent population model; it does not
implement covariate effects or cross-parameter correlation.

Prior predictive validation uses seed 701 and 20,000 draws. Plausible ranges are
the model bounds for growth [0,0.15], margin [0.1,0.4] and senior/mezz rates
[0.02,0.12], and [1,8] for lease multiple. Report mean, SD, quantiles, plausible
frequency and exact-boundary frequency. Bounds are structural assumptions, not
empirically established ranges. Do not revise priors after inspecting recovery.

Recovery uses final seeds 1101, 1102 and 1103 for each design: central n=30; low
heterogeneity n=30; high heterogeneity n=30; near-boundary n=30; calibration-sized
n=10; and larger n=100. Generate observations from the corrected likelihood.
Each joint fit uses two chains, 600 tuning and 600 retained draws per chain,
target_accept 0.95. Summaries include posterior mean/median/SD, 50% and 90%
intervals, coverage, absolute and relative errors for all ten hyperparameters.

Acceptance requires zero divergences across final runs; every substantive R-hat
<=1.01; bulk and tail ESS >=200; overall 90% interval coverage >=0.80; coverage in
each design >=0.70; mean-parameter relative RMSE <=0.35; scale-parameter relative
RMSE <=0.60; and no parameter family with absolute standardized bias above 0.75,
where standardized bias is mean signed error divided by the nonzero truth. These
thresholds are screening criteria, not universal Bayesian standards. Diagnostics
do not prove identification.

Posterior predictive checks integrate over actual hyperparameter draws and draw a
new population observation for every retained hyperparameter draw. For recovery
data, report predictive mean/SD/quantiles and whether the known population mean is
within the predictive 90% interval. For stipulated inputs, report the observed
mean/SD against posterior-predictive summaries without treating agreement as
parameter recovery.

Prior sensitivity fits the stipulated inputs under the baseline prior and one
moderately wider specification (twice each mean-prior and scale-prior SD), using
seeds 2101 and 2102. Compare hyperparameter posterior means relative to the
baseline posterior SD. A maximum shift above 0.5 posterior SD is classified as
material prior sensitivity. This threshold is frozen before fitting. No search
over alternative priors is permitted.

Admission rule: choose B only if the implementation and recovery acceptance gates
pass, while explicitly limiting the component to a synthetic methodological
experiment. Choose C if recovery/diagnostics fail materially, uncertainty is
pathologically prior-driven, or repair would require disproportionate redesign.
Never choose A with the present data classification. No LBO integration,
optimization, score comparison, manuscript result or publication claim is part
of this protocol.
