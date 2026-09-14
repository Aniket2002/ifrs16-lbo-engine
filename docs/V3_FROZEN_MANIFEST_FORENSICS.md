# V3 frozen-manifest forensics

Date: 2026-09-15

## Conclusion

The frozen-source validator is not checkout-portable, but an EOL-only validator repair is
not safe yet. Of 66 protected paths, 59 match the current Windows worktree, five fail only
because a fresh Windows checkout has CRLF bytes where the manifest records LF bytes, and
two manifest entries match neither the recorded source-commit blob nor its LF or CRLF
representation:

- `results/v3/baseline/pipeline_step_2.txt`
- `results/v3/foundation_validation.json`

Both residual paths are protected validation evidence. Their canonical Git content has not
changed since before the freeze, but the origin of the two recorded hashes cannot be
reconstructed from repository history or common encoding transformations. This is an
historical manifest inconsistency, not evidence of a post-freeze Git content change. Under
the task's repair policy, the classification is **Option D — STOP**. The old manifest and
all protected files must remain unchanged while the two hash origins are investigated.

## Baseline and environment

- Canonical baseline: `origin/main` at
  `1dc507a7bbb55d6b83f8e550c4dd322ee230e795`.
- Diagnostic branch: `forensic/frozen-manifest`, created directly from `origin/main`.
- Initial local `main`: `e976debdf03e1a4fd99a86179703a2970a4c26b4`, the intentionally
  unpushed presentation-only `STRUCTURE.md` commit. It was preserved and excluded from
  this diagnosis.
- Initial and diagnostic worktrees: clean.
- Python: 3.14.0.
- OS: Microsoft Windows NT 10.0.26200.0.
- Git `core.autocrlf`: `true`, from `C:/Program Files/Git/etc/gitconfig`.
- Git `core.eol`: unset.
- Git `core.safecrlf`: unset.
- `.gitattributes`: absent.
- Observed checkout behavior for all seven failing paths: index `LF`, worktree `CRLF`
  (`git ls-files --eol` reports `i/lf w/crlf attr/`).

Consequently, Git stores LF blobs and converts recognized text files to CRLF on this
checkout. With no attributes, checkout byte representation depends on Git configuration
and platform conventions.

## Exact failing tests

The full command `python -m pytest -q --no-cov` collected 173 tests and produced 171
passes and two failures:

1. `tests/test_manuscript_freeze.py::test_complete_frozen_metadata_and_sources`
   asserts `validate(ROOT)["passed"]` and reports seven `frozen source changed` errors.
2. `tests/test_manuscript_rendering.py::test_render_is_deterministic_and_preserves_frozen_inputs`
   calls the same validator before rendering and raises `ValueError` with the same seven
   errors.

The first focused command, `python -m pytest -q --no-cov tests/test_manuscript_freeze.py`,
produced 12 passes and the first failure above. The assertion messages include only paths;
they do not expose expected or observed digests. The exact digests were therefore computed
independently below.

Failing paths:

1. `results/v3/baseline/branch_start.json`
2. `results/v3/baseline/checksums.json`
3. `results/v3/baseline/pipeline_step_2.txt`
4. `results/v3/bayesian_validation/verification.json`
5. `results/v3/optimization_validation/verification.json`
6. `src/lbo/validation.py`
7. `results/v3/foundation_validation.json`

## Manifest contract

`results/v3/manuscript_freeze/source_manifest.json` contains a `source_commit` field of
`0f521a28150fe56f98cdabb3d1cb3664286df124` and a map of 66 paths to SHA-256 digests.
The manifest itself was added later by freeze commit
`21d726606f3208bcee6e3e84ae4c357e3a33d79b`.

The only consuming implementation is `analysis/validate_manuscript_freeze.py::validate`.
For each entry it executes the equivalent of:

```python
hashlib.sha256((root / path).read_bytes()).hexdigest()
```

`analysis.scripts.audit_manuscript_v3` and
`analysis.scripts.render_manuscript_v3` consume the manifest indirectly by calling this
validator. The tests do the same. There is no text decoding, EOL normalization, encoding
normalization, or Git-blob lookup.

The executable contract is therefore **exact current worktree bytes**. The surrounding
metadata describes a source commit and calls the entries frozen source hashes, which
suggests source identity, but it does not define whether archival checkout bytes, Git blob
bytes, or normalized text were intended. The intended contract is therefore ambiguous.

The stored digests demonstrate that they cannot represent one consistent Git-oriented or
worktree-oriented policy:

- 57 expected digests equal a CRLF transformation of the source-commit blob;
- five equal only the raw LF source-commit blob;
- two newline-free files have representations unaffected by EOL conversion and match the
  source blob;
- two equal neither representation.

Thus 64 of 66 expected digests are explainable as source content in either LF or CRLF
form, but the chosen form is mixed. No standard clean checkout can be expected to recreate
that per-file mixture on every platform.

## Seven worktree mismatches

Lengths are bytes. `HEAD` is the canonical `origin/main` baseline. `source` is the Git
blob at the manifest's recorded source commit. `LF` and `CRLF` are transformations of the
current UTF-8 worktree bytes.

| Path | Representation | SHA-256 | Length |
|---|---|---|---:|
| `results/v3/baseline/branch_start.json` | expected | `f7a6df65dc87b6fcc9ae2499b05ac60fd4fbbfc3fe62876cec493b2a3cb06b67` | 818 |
|  | worktree / CRLF | `524bfef2473e49251b44c38ae3ec030e55ed0e91d66eab91a256a099e117ea3f` | 834 |
|  | HEAD / source / LF | `f7a6df65dc87b6fcc9ae2499b05ac60fd4fbbfc3fe62876cec493b2a3cb06b67` | 818 |
| `results/v3/baseline/checksums.json` | expected | `c79401bd32752d8c039ca27b7fb9585e465aff97c6734e0ae100fddf3eee2866` | 1066 |
|  | worktree / CRLF | `e88a6d32c4161b11f2cd3b06c954c144766b27a4d8acd8d267e49d0fb872a7e3` | 1079 |
|  | HEAD / source / LF | `c79401bd32752d8c039ca27b7fb9585e465aff97c6734e0ae100fddf3eee2866` | 1066 |
| `results/v3/baseline/pipeline_step_2.txt` | expected | `39103bb5463c39089e1dab6d3582a3c72851b93cf79494092e43c94c677a9c1d` | unknown |
|  | worktree / CRLF | `e9fa2e0f0e2cf67588cd1dbddbd3f1d097f3bb7b976179a0900de495f15375ea` | 400 |
|  | HEAD / source / LF | `dcde65838916e683f3247cef81f66b2ddb9bb723bf8aa8fd2e93faaa79b30d5a` | 393 |
| `results/v3/bayesian_validation/verification.json` | expected | `e5f94477ef92a5b36877634bbdb93df288200d1b61a54b6c3e0bd179cc7203f4` | 1258 |
|  | worktree / CRLF | `fe3464c5f969e8ea3518a9dc344861a88576231ff5693eeaf994b85172e05a02` | 1277 |
|  | HEAD / source / LF | `e5f94477ef92a5b36877634bbdb93df288200d1b61a54b6c3e0bd179cc7203f4` | 1258 |
| `results/v3/optimization_validation/verification.json` | expected | `75f91c96eec045f78dc54677f1e50243bd6c4f9c30d953e38a8d7b9c02393ae7` | 1548 |
|  | worktree / CRLF | `ad1637dce6d4800e206ebac993d409c3524bc1b5df2189e7c4be2b75ed368ac9` | 1573 |
|  | HEAD / source / LF | `75f91c96eec045f78dc54677f1e50243bd6c4f9c30d953e38a8d7b9c02393ae7` | 1548 |
| `src/lbo/validation.py` | expected | `5eee30c91d2840033657590baff3ee10953cb1d2610088496f76770031010f5b` | 14079 |
|  | worktree / CRLF | `f5e024f854d10005925f9da73159ea623ed3e8aab31805418c2b51c33e3fb4bc` | 14464 |
|  | HEAD / source / LF | `5eee30c91d2840033657590baff3ee10953cb1d2610088496f76770031010f5b` | 14079 |
| `results/v3/foundation_validation.json` | expected | `10529942dc0452dbfcc8faaba4b15faf58c11dbcbb37f7e09f223c249775460e` | unknown |
|  | worktree / CRLF | `e74ee6310e9e071ef35a8c86c913f7c0fd260f848612a4b3dea3e35b837395fc` | 4545 |
|  | HEAD / source / LF | `a34427774c939a8b23c14f4c1e6fcfff928ce1d6ee151664dd17fde34df594df` | 4397 |

The expected byte length is not recoverable from a SHA-256 digest for the two historical
inconsistencies. All actual candidate representations and their lengths are recorded.

## Per-file classification for all protected paths

`HEAD=source` compares exact Git blobs. `Expected representation` compares the manifest
digest with the source blob after only the stated EOL representation. `MATCH` means the
current worktree passes as checked out; it does not imply a portable hash contract.

| Path | Worktree | Expected representation | HEAD=source | Classification |
|---|---:|---|---:|---|
| `results/v3/baseline/branch_start.json` | mismatch | LF | yes | EOL_ONLY |
| `results/v3/baseline/checksums.json` | mismatch | LF | yes | EOL_ONLY |
| `results/v3/baseline/pipeline_step_1.txt` | match | CRLF | yes | MATCH |
| `results/v3/baseline/pipeline_step_2.txt` | mismatch | neither LF nor CRLF | yes | MANIFEST_BUG / unresolved historical bytes |
| `results/v3/baseline/pipeline_step_3.txt` | match | CRLF | yes | MATCH |
| `results/v3/baseline/reproduced_benchmark_seed42.json` | match | CRLF | yes | MATCH |
| `results/v3/baseline/reproduced_environment.txt` | match | CRLF | yes | MATCH |
| `results/v3/baseline/reproduced_manifest.json` | match | CRLF | yes | MATCH |
| `results/v3/baseline/reproduced_pdf_verification.json` | match | CRLF | yes | MATCH |
| `results/v3/baseline/reviewed_artifact_verification.json` | match | CRLF | yes | MATCH |
| `results/v3/baseline/tests.xml` | match | EOL-invariant (no newline bytes) | yes | MATCH |
| `results/v3/baseline/verification.json` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/fold_results.csv` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/heldout_predictions.csv` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/summary.json` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/threshold_curve_train_SYN_HOTEL_001.csv` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/threshold_curve_train_SYN_HOTEL_002.csv` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/threshold_curve_train_SYN_HOTEL_003.csv` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/threshold_curve_train_SYN_HOTEL_004.csv` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/threshold_curve_train_SYN_HOTEL_005.csv` | match | CRLF | yes | MATCH |
| `results/v3/template_evaluation/verification.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/admission_decision.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/data_provenance.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/implementation_audit.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/mcmc_diagnostics.csv` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/posterior_predictive_summary.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/prior_predictive_summary.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/prior_sensitivity.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/recovery_runs.csv` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/recovery_summary.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/run_metadata.json` | match | CRLF | yes | MATCH |
| `results/v3/bayesian_validation/verification.json` | mismatch | LF | yes | EOL_ONLY |
| `results/v3/optimization_validation/admission_decision.json` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/candidate_grid.csv` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/candidate_results.csv` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/economic_audit.json` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/fold_selection.csv` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/frozen_protocol.json` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/heldout_results.csv` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/rate_sensitivity.csv` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/reference_policy.csv` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/run_metadata.json` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/summary.json` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/toy_validation.json` | match | CRLF | yes | MATCH |
| `results/v3/optimization_validation/verification.json` | mismatch | LF | yes | EOL_ONLY |
| `results/v3/post_optimization_diagnostics/coverage.json` | match | EOL-invariant (no newline bytes) | yes | MATCH |
| `results/v3/post_optimization_diagnostics/cross_stage_template004.json` | match | CRLF | yes | MATCH |
| `results/v3/post_optimization_diagnostics/debt_boundary_profile.csv` | match | CRLF | yes | MATCH |
| `results/v3/post_optimization_diagnostics/debt_boundary_summary.json` | match | CRLF | yes | MATCH |
| `results/v3/post_optimization_diagnostics/diagnostic_summary.json` | match | CRLF | yes | MATCH |
| `results/v3/post_optimization_diagnostics/optimization_code_coverage.json` | match | CRLF | yes | MATCH |
| `results/v3/post_optimization_diagnostics/regime_conditioned_template_comparison.csv` | match | CRLF | yes | MATCH |
| `results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv` | match | CRLF | yes | MATCH |
| `results/v3/post_optimization_diagnostics/template004_structural_profile.json` | match | CRLF | yes | MATCH |
| `results/v3/result_validation/optimization.json` | match | CRLF | yes | MATCH |
| `src/lbo/analytic_bounds.py` | match | CRLF | yes | MATCH |
| `src/lbo/covenants.py` | match | CRLF | yes | MATCH |
| `src/lbo/data.py` | match | CRLF | yes | MATCH |
| `src/lbo/full_simulation.py` | match | CRLF | yes | MATCH |
| `src/lbo/lbo_model.py` | match | CRLF | yes | MATCH |
| `src/lbo/lbo_model_analytic.py` | match | CRLF | yes | MATCH |
| `src/lbo/validation.py` | mismatch | LF | yes | EOL_ONLY |
| `src/lbo/__init__.py` | match | CRLF | yes | MATCH |
| `results/v3/foundation_validation.json` | mismatch | neither LF nor CRLF | yes | MANIFEST_BUG / unresolved historical bytes |
| `data/synthetic/operators.csv` | match | CRLF | yes | MATCH |
| `analysis/run_benchmark.py` | match | CRLF | yes | MATCH |

## EOL findings

- Current raw worktree matches: 59/66.
- Current Git blobs matching manifest: 7/66.
- Recorded source-commit Git blobs matching manifest: 7/66.
- All 66 current Git blobs are byte-identical to their recorded source-commit blobs.
- Expected digest explained by CRLF source representation: 57/66.
- Expected digest explained by raw LF source representation: 5/66.
- Expected digest EOL-invariant because the file has no newline bytes: 2/66.
- Expected digest unexplained by either LF or CRLF: 2/66.
- Of the seven current failures, five become exact matches after LF normalization and
  zero become exact matches only after CRLF transformation because the checkout is already
  CRLF. Two remain unexplained.
- In source-oriented terms, 64/66 manifest entries are explainable by accepting either
  LF or CRLF byte representation. This does not resolve the other two.

## Git-history findings for non-EOL mismatches

### `results/v3/baseline/pipeline_step_2.txt`

- Latest and only commit touching the path:
  `655953ffbce56b812f526a6a019b9c71782dc644` (`v3: establish reviewed baseline and
  record validation gaps`).
- The commit predates both the recorded source commit and containing freeze commit.
- `git diff 0f521a2..HEAD -- <path>` is empty.
- The introduction diff adds a seven-line TeX build transcript. This is provenance/build
  evidence, not model code or a numerical result, but it is explicitly protected evidence.
- Every path revision has the same 393-byte LF blob; its 400-byte CRLF form also does not
  match the manifest.

### `results/v3/foundation_validation.json`

- Latest and only commit touching the path:
  `dbf494f27ee8aff9e0faeb530fb04e1a7df2e102` (`v3: archive foundation validation
  results`).
- The commit predates both the recorded source commit and containing freeze commit.
- `git diff 0f521a2..HEAD -- <path>` is empty.
- The introduction diff adds the archived foundation-validation result: independent return
  checks, debt schedule, runtime invariants, adversarial checks, coverage, and reproduction
  status. This is research-validation evidence.
- Every path revision has the same 4,397-byte LF blob; its 4,545-byte CRLF form also does
  not match the manifest.

For both files, UTF-8 with and without BOM, UTF-16 LE/BE with and without BOM, LF, CRLF,
and removal of the final newline were tested. None reproduce the expected digest. Their
recorded hashes therefore appear to describe transient or otherwise unavailable bytes.
There is no Git evidence that either canonical file changed after the freeze.

## Portability assessment

The current validator intentionally implements exact worktree-byte checking in code, but
the metadata and source-commit linkage suggest the higher-level purpose was frozen source
identity. These are different contracts:

- If exact archival byte identity was intended, the archive failed to specify the checkout
  EOL policy and stored a mixed representation. Normalization would change that contract.
- If cross-platform source identity was intended, hashing checkout bytes directly is a
  portability defect. A fresh Windows checkout already proves it, and a normal LF checkout
  would fail the 57 CRLF-specific entries.

The evidence supports the existence of a portability bug, but does **not** support applying
an EOL-only fix while two protected research-evidence hashes remain irreconcilable. A fix
that special-cased or broadly canonicalized those two would weaken the freeze and is not
permitted.

## Unresolved cases

The unresolved fact is the byte content originally hashed for:

- `39103bb5463c39089e1dab6d3582a3c72851b93cf79494092e43c94c677a9c1d`
  (`pipeline_step_2.txt`)
- `10529942dc0452dbfcc8faaba4b15faf58c11dbcbb37f7e09f223c249775460e`
  (`foundation_validation.json`)

The freeze's archived validation report says all 66 hashes passed when created, but the
matching byte sequences are not committed. Possible transient-worktree provenance cannot
be established from Git and is not treated as fact.

## Recommended repair options, ranked by risk

1. **Option D — STOP (lowest risk; selected).** Preserve the old manifest and protected
   files. Recover the original freeze workspace, CI artifact, backup, or generation log and
   compare the two exact byte sequences. Do not claim the historical manifest is fully
   reproducible until this is resolved.
2. **Option C — documented historical mismatch (moderate risk, later authorization).** If
   the original bytes cannot be recovered, retain the historical manifest as-is and create
   a separate, explicitly current-state integrity record based on canonical Git blobs. Do
   not portray that record as repairing or replacing the old freeze.
3. **Option B — portability fix (conditionally appropriate, not yet justified).** After the
   two anomalies are resolved and the contract is explicitly source-oriented, validate
   text by accepting only the exact LF or CRLF representation associated with the recorded
   digest, while continuing raw-byte checks for binary data. Add focused LF, CRLF,
   substantive-mutation, and binary regression tests. Preserve all existing hashes.
4. **Option A — no code change / checkout recipe (highest operational fragility).** This is
   insufficient now: the manifest's mixed per-file EOL forms cannot be recreated by a
   normal repository-wide checkout setting, and it cannot explain the two anomalous hashes.

## Integrity and action record

- No manifest was regenerated or edited.
- No protected hash was updated.
- No research result, manuscript, claim ledger, model code, validator, or test was edited.
- No validation conclusion was changed.
- No repair was implemented.
- Nothing was committed or pushed as part of this forensic step.

Final verification on the documented worktree:

- `python -m pytest -q --no-cov`: 171 passed, two validator-dependent failures listed
  above.
- `python -m ruff check .`: passed.
- `python -m ruff format --check .`: passed; 41 files already formatted.
- `python analysis/validate_manuscript_freeze.py`: failed only with the seven classified
  frozen-source paths.
- `python -m analysis.scripts.audit_manuscript_v3`: could not complete because the
  untracked render intermediate `paper/ifrs16_lbo_ssrn_v3.aux` is absent. The render test
  cannot produce it while the frozen-source gate is red.
- Git diffs against canonical `HEAD` are empty for `paper/`, `results/v3/`, both claim
  ledgers, the source manifest, all protected source files, and all protected evidence.
- The only worktree change is this report.
