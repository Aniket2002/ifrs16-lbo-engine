# Research contribution and limits

The central contribution is a validated separation between mechanical model
correctness, ranking performance, and decision usefulness in an IFRS 16 covenant
screening setting.

The full simulation passes independent schedules, runtime reconciliation checks,
and adversarial tests on the examined paths. The reduced-form screening score
ranks broad failure strongly in the pooled frozen synthetic sample. Yet thresholds
selected without access to each held-out archetype transfer unevenly, especially
for the structurally difficult Template 004. Mechanical correctness and pooled
ranking therefore do not imply a portable decision threshold across structurally
different synthetic borrower archetypes.

The repository contributes:

- explicit and tested cash, debt, lease, covenant, and sponsor-return mechanics;
- a reproducible grouped threshold-transfer design that prevents held-out leakage;
- transparent reporting of undefined small-sample metrics and structural fold
  heterogeneity;
- an evidence hierarchy that distinguishes engine checks, ranking evidence, and
  decision claims; and
- machine-readable claim governance linking manuscript statements to frozen
  sources while excluding unsupported historical claims.

The Bayesian experiment is excluded from substantive results because data
provenance and frozen admission gates are insufficient. The financing-design
experiment remains a methodological demonstration because its boundary solutions
and economic omissions preclude a substantive optimum claim. Neither is the
paper's central contribution.

The study does not establish real-company default prediction, a calibrated
probability of default, causal IFRS 16 effects, realized transaction performance,
or a universal covenant threshold. The Accor example remains an illustration of
supplied inputs. See [model and methods](MODEL_AND_METHODS.md),
[validation summary](VALIDATION_SUMMARY.md), and the
[final paper](../paper/ifrs16_lbo_ssrn_v3.pdf).
