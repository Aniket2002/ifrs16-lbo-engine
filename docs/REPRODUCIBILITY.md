# Reproducibility

`main` is the authoritative current branch. Cite the exact commit used because
future documentation or tooling changes may alter repository state without
changing the frozen research evidence.

## What is frozen

- Final manuscript source and PDF: `paper/ifrs16_lbo_ssrn_v3.*`
- Synthetic source inputs: `data/synthetic/`
- Foundation validation: `results/v3/foundation_validation.json`
- Threshold-transfer evidence: `results/v3/template_evaluation/`
- Bayesian validation and exclusion evidence: `results/v3/bayesian_validation/`
- Financing-design evidence: `results/v3/optimization_validation/`
- Structural diagnostics: `results/v3/post_optimization_diagnostics/`
- Claim freeze and source manifests: `results/v3/manuscript_freeze/`
- Final rendering, build, audit, and visual-review records:
  `results/v3/manuscript/`

Machine-readable evidence remains available even when a reader-facing document
summarizes it. The internal protocols and governance documents are indexed in
[`docs/README.md`](README.md).

## Validate the checked-out repository

Install the package and development dependencies, then run:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q --no-cov
python analysis/validate_manuscript_freeze.py
```

The freeze validator checks claim structure, exact source selectors, archived
values, admission restrictions, prohibited claims, traceability, and current protected
source identity using canonical Git blob bytes. The original checkout-byte manifest is
preserved unchanged as historical provenance. Its separate audit remains explicit about two
unresolved historical entries; see [V3 integrity governance](V3_INTEGRITY_GOVERNANCE.md).
After the manuscript build creates its LaTeX auxiliary files, the final audit
checks source and PDF wording, generated numeric cells, exhibit hashes and counts,
bibliography, layout conditions, build status, and the archived visual review.

## Rebuild the manuscript

Follow [`paper/REPRODUCE_V3.md`](../paper/REPRODUCE_V3.md). In brief:

```powershell
python -m analysis.scripts.render_manuscript_v3
python -m analysis.scripts.build_manuscript_v3
python -m analysis.scripts.audit_manuscript_v3
```

The renderer reads frozen evidence and creates seven public tables and six figures.
The build uses Tectonic 0.17.0 in the recorded environment and embeds the checked-
out source commit. A later build may therefore differ at the provenance text and
PDF hash even when its research content is unchanged. Visual approval belongs to
the archived PDF hash and must not be copied to a newly built PDF without review.

## Experiment reproduction

The final manuscript build does not rerun stochastic work. The detailed protocol
documents preserve the fixed choices for template evaluation, Bayesian recovery,
and financing optimization; the corresponding result directories record commands,
seeds, input hashes, dependency versions, and source commits. Run those workflows
only when intentionally reproducing an experiment, and archive the new provenance
rather than overwriting the interpretation of the validated artifacts.

The root [`REPRODUCE.md`](../REPRODUCE.md) retains supported setup, benchmark, and
illustration commands. The [model and methods guide](MODEL_AND_METHODS.md) states
the conventions and boundaries needed to interpret their outputs.
