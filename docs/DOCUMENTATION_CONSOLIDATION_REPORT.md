# Documentation consolidation report

## Outcome

- Starting branch: `v3/repository-cleanup`
- Starting commit: `02029425861918960ee520919e1dcfb0ca03e086`
- Documentation files before: 21
- Documentation files after: 24
- Reader-facing current documents: 6
- Research/model/result changes: none

The file count increased because the retained internal audit trail now has a
separate, small reader-facing layer and a navigation page. Internal records no
longer appear as equally prominent entry points.

## Canonical reader-facing documents

- `docs/README.md`
- `docs/MODEL_AND_METHODS.md`
- `docs/VALIDATION_SUMMARY.md`
- `docs/RESEARCH_CONTRIBUTION.md`
- `docs/REPRODUCIBILITY.md`
- `docs/REPOSITORY_HISTORY.md`

The root `README.md` now leads with the final paper title, validated status, final
PDF, reproduction commands, research boundaries, and this documentation index.
The root `REPRODUCE.md` distinguishes routine validation and frozen manuscript
rendering from optional experiment regeneration. Both identify `main` as the
authoritative current branch.

## Consolidated and removed paths

| Removed path | Consolidated into | Reason |
| --- | --- | --- |
| `docs/experimental_design.md` | `docs/MODEL_AND_METHODS.md` | Its distinct benchmark, failure, score, output, and timing conventions were retained in the current methods guide. No frozen or executable reference named the old path. |
| `docs/model_specification.md` | `docs/MODEL_AND_METHODS.md` | Its simulation, closing, waterfall, covenant, return, and analytic-model conventions were retained. No frozen artifact named the old path. |
| `docs/research_contribution.md` | `docs/RESEARCH_CONTRIBUTION.md` | The contribution page was given its canonical name and updated to the final v3 thesis and limitations. No frozen artifact named the old path. |
| `docs/REPOSITORY_CLEANUP_PLAN.md` | `docs/REPOSITORY_CLEANUP_REPORT.md` | The unique pre-deletion classification and risk rationale was folded into the final report. No manifest, audit, script, test, or documentation link referenced the plan. |

References in `README.md`, `REPRODUCE.md`, and the new reader-facing documents now
point to the consolidated paths. A repository-wide search found no remaining link
to the removed method or contribution paths.

## Provenance retained

The following detailed records remain at their original paths:

- Validation protocols: `V3_TEMPLATE_EVALUATION_PROTOCOL.md`,
  `V3_BAYESIAN_VALIDATION_PROTOCOL.md`, and
  `V3_OPTIMIZATION_VALIDATION_PROTOCOL.md`.
- Validation reports: `V3_FOUNDATION_VALIDATION.md`,
  `V3_TEMPLATE_EVALUATION.md`, `V3_BAYESIAN_VALIDATION.md`,
  `V3_OPTIMIZATION_VALIDATION.md`, `V3_POST_OPTIMIZATION_DIAGNOSTICS.md`, and
  `V3_VALIDATION_GAPS.md`.
- Manuscript governance: `V3_CLAIM_LEDGER.md`, `V3_PROHIBITED_CLAIMS.md`,
  `V3_MANUSCRIPT_BLUEPRINT.md`, `V3_TABLE_PLAN.md`, `V3_FIGURE_PLAN.md`, and
  `V3_MANUSCRIPT_FREEZE_SUMMARY.md`.
- Publication evidence: `V3_MANUSCRIPT_BUILD_REPORT.md`.
- Repository governance: `REPOSITORY_CLEANUP_REPORT.md` and this report.

Fourteen of those v3 records are named directly by an executable runner, frozen
result metadata, the source audit, claim ledger, or freeze summary. They could not
be moved or merged away without breaking reproducibility or the frozen evidence
chain. The foundation and build reports were also retained because they contain
unique validation and publication provenance.

## Verification

| Check | Result |
| --- | --- |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed; 41 Python files already formatted |
| `python -m pytest -q --no-cov` | Passed; 175 tests |
| `python analysis/validate_manuscript_freeze.py` | Passed; 100 claims, 34 quantitative claims, 50 prohibited-claim families, and 66 frozen source hashes checked |
| `python -m analysis.scripts.render_manuscript_v3` | Passed; seven tables and six figures rendered |
| `python -m analysis.scripts.build_manuscript_v3` | Passed with Tectonic 0.17.0; 16 pages, resolved bibliography, and no serious warnings |
| `python -m analysis.scripts.audit_manuscript_v3` on the restored validated artifact | Passed every check, including protected hashes and final visual review |

The build embeds the checked-out source commit, so its newly generated PDF had a
different provenance hash and did not inherit the archived PDF's visual approval.
After confirming build success, all provenance-bearing build outputs were restored
to their validated versions. A path-restricted diff from the starting cleanup
commit is empty for all v3 results, final manuscript source/PDF/instructions,
v3 figures and generated tables, and synthetic source data.

No numerical result, source datum, model behavior, threshold, score, admission
decision, manuscript conclusion, frozen claim, or protected validation artifact
was changed.
