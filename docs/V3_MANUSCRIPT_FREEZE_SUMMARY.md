# V3 manuscript freeze summary

Claim and architecture freeze only. No manuscript, publication figures or PDF generated.

```json
{
  "source_commit": "0f521a28150fe56f98cdabb3d1cb3664286df124",
  "documentation_repair_commit": "0f521a28150fe56f98cdabb3d1cb3664286df124",
  "central_thesis": "A screening model can be mechanically correct and rank adverse outcomes well while still failing to provide a stable, portable decision threshold across structurally different synthetic borrower archetypes.",
  "primary_claim_ids": [
    "SCR-001",
    "SCR-002",
    "SCR-004",
    "SCR-006",
    "SCR-007",
    "SCR-009",
    "SCR-011",
    "SCR-012",
    "SCR-013"
  ],
  "secondary_claim_ids": [
    "HET-001",
    "HET-002",
    "HET-009"
  ],
  "claim_count": 100,
  "prohibited_claim_count": 50,
  "bayes": "C: excluded main results; methodological exclusion appendix only; no posterior integration",
  "optimization": "B: methodological appendix only; no substantive optimum or abstract/conclusion numerical result",
  "structural_heterogeneity": "Template 004 shows weak threshold transfer and comparatively weak financing-policy outcomes, providing convergent descriptive evidence within the same synthetic system consistent with structural heterogeneity affecting downstream transfer. Same synthetic archetype system only; descriptive/convergent, not independent replication, external validation, causal evidence or empirical borrower evidence. Small regime-conditioned cells, including distressed n=21; show n with percentages.",
  "small_n_rule": "Every subgroup/regime/fold percentage must carry n or numerator/denominator in the same sentence, table row, table note or immediately adjacent text; no new confidence intervals may be manufactured.",
  "limited_liability": "Template 004 reference distressed P10 annualized sponsor return is -100% (n=21): 7/21 scenarios (33.3%) have zero recovery, placing the 10th percentile on the zero-recovery mass, consistently with the predeclared convention.",
  "table_plan": {
    "main": [
      "T1",
      "T2",
      "T3",
      "T4"
    ],
    "appendix": [
      "T5",
      "A1",
      "A2",
      "A3"
    ]
  },
  "figure_plan": {
    "main": [
      "F1",
      "F2",
      "F3"
    ],
    "appendix": [
      "A-F1",
      "A-F2",
      "A-F3"
    ]
  },
  "architecture": [
    {
      "section_id": "S1",
      "title": "Introduction",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S2",
      "title": "Research question and contribution",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S3",
      "title": "IFRS-16 covenant-screening setup",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S4",
      "title": "Financial-engine mechanics and validation",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S5",
      "title": "Synthetic benchmark and evaluation protocol",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S6",
      "title": "Ranking performance versus threshold transfer",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S7",
      "title": "Cross-archetype heterogeneity",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S8",
      "title": "Financing-design experiment: methodological demonstration only",
      "main_text_or_appendix": "appendix"
    },
    {
      "section_id": "S9",
      "title": "Excluded Bayesian calibration experiment",
      "main_text_or_appendix": "appendix"
    },
    {
      "section_id": "S10",
      "title": "Limitations",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S11",
      "title": "Conclusion",
      "main_text_or_appendix": "main"
    },
    {
      "section_id": "S12",
      "title": "Appendices and reproducibility",
      "main_text_or_appendix": "appendix"
    }
  ],
  "top_three_titles": [
    {
      "rank": 1,
      "title": "Ranking Is Not Threshold Portability: Validating IFRS-16 Covenant Screening in a Synthetic Benchmark",
      "defensibility": "Directly states SCR-001 and synthetic scope; neither calibration nor economic optimality is implied."
    },
    {
      "rank": 2,
      "title": "From Engine Correctness to Decision Usefulness: A Synthetic Evaluation of IFRS-16 Covenant Screening",
      "defensibility": "Names the three-layer architecture while keeping the evaluation explicitly synthetic."
    },
    {
      "rank": 3,
      "title": "High Ranking Performance, Uneven Threshold Transfer: IFRS-16 Screening Across Synthetic Archetypes",
      "defensibility": "Tracks SCR-004 and the observed fold heterogeneity without asserting empirical generalization."
    }
  ],
  "rounding": {
    "auc_ap": "3 decimals; AP means average precision, not trapezoidal PR area",
    "percentage": "1 decimal percentage point; numerator/denominator for observed subgroup rates",
    "threshold": "6 decimals (retain full precision in ledger and predictions; do not recompute classifications from rounded thresholds)",
    "returns": "2 decimal percentage points; label annualized limited-liability sponsor return",
    "multiples": "2 decimals",
    "coverage": "statement coverage 2 decimals, branch coverage 1 decimal; counts alongside",
    "counts": "integers; zero denominator is undefined, never zero",
    "source_artifacts": "Never round or rewrite; machine snapshots preserve full stored precision",
    "confidence_intervals": "No new intervals; old v2 bootstrap interval is not promoted"
  },
  "terminology": {
    "preferred": [
      "synthetic archetype",
      "screening score",
      "ranking performance",
      "threshold transfer",
      "held-out template",
      "broad financial failure",
      "payment default",
      "financing-design experiment",
      "methodological demonstration",
      "limited-liability sponsor return"
    ],
    "restricted": [
      "calibrated PD",
      "borrower prediction",
      "out-of-sample generalization",
      "real-world validation",
      "optimal capital structure",
      "empirical hotel evidence",
      "causal",
      "robustly generalizes"
    ],
    "rule": "Restricted terms may occur only in an explicit limitation/exclusion, not an affirmative result."
  },
  "limitations": [
    "Synthetic single-system evidence and few templates; overlapping training sets and sparse positives.",
    "Ranking is not calibrated probability or economic utility.",
    "Financing has stipulated prices, lease convention asymmetry, no endogenous pricing or recovery, and boundary/rate sensitivity.",
    "Bayesian provenance and technical gates failed; no empirical calibration.",
    "No causal identification, external validation, independent replication or invented confidence intervals."
  ],
  "artifact_lineage": {
    "reviewed_v2": "a990dd2166f47917781112a7e12bbc16f33605fd",
    "foundation": "c2a31756955004fac121984b3804f5385a6b30a8",
    "threshold_protocol": "75f1b56d7613389012f7cc7e25ae95bcb4ddd004",
    "bayes_protocol": "4b7179367d7b69c6870f530f24a283a83bfd0c8f",
    "optimization_protocol": "fdc949f4ff0c8ef4b3f318e31aff7a4a9cd6cccf",
    "source_manifest": "source_manifest.json",
    "source_audit": "source_audit.json"
  },
  "next_permitted_stage": "A separately authorized manuscript-drafting stage using only admitted claims and frozen plans; no manuscript, final figure, PDF, release/tag/DOI generated here.",
  "validation_status": {
    "passed": true,
    "errors": [],
    "claims_checked": 100,
    "quantitative_claims_checked": 34,
    "prohibited_claims_checked": 50,
    "frozen_source_files_hash_checked": 66,
    "checks": [
      "required fields and classes",
      "exact source selectors and snapshots",
      "CSV/JSON equality",
      "denominator metadata",
      "B/C admission restrictions",
      "prohibited and stale coverage restrictions",
      "traceability coverage",
      "section/abstract/visual claim links",
      "frozen file SHA-256"
    ],
    "source_commit": "0f521a28150fe56f98cdabb3d1cb3664286df124",
    "ruff_check": "passed",
    "ruff_format_check": "passed (40 files)",
    "focused_tests": {
      "command": "python -m pytest -q --no-cov tests/test_manuscript_freeze.py",
      "passed": 13
    },
    "protected_diff_from_original_validated_head": "",
    "new_experiments_run": false,
    "full_test_suite_rerun": false,
    "read_only_source_cross_checks": [
      "raw screening AUC and confusion counts",
      "raw financing aggregate medians and subtype counts",
      "Template 004 regime counts and limited-liability formulas",
      "reference distressed P10 lies on seven zero-recovery observations among 21",
      "monotonicity only across feasible observed debt levels"
    ]
  },
  "files_created": [
    "results/v3/manuscript_freeze/claim_ledger.csv",
    "results/v3/manuscript_freeze/claim_ledger.json",
    "results/v3/manuscript_freeze/claim_traceability.json",
    "results/v3/manuscript_freeze/figure_plan.json",
    "results/v3/manuscript_freeze/freeze_summary.json",
    "results/v3/manuscript_freeze/manuscript_structure.json",
    "results/v3/manuscript_freeze/prohibited_claims.json",
    "results/v3/manuscript_freeze/source_audit.json",
    "results/v3/manuscript_freeze/source_manifest.json",
    "results/v3/manuscript_freeze/table_plan.json",
    "results/v3/manuscript_freeze/validation_report.json",
    "docs/V3_CLAIM_LEDGER.md",
    "docs/V3_PROHIBITED_CLAIMS.md",
    "docs/V3_MANUSCRIPT_BLUEPRINT.md",
    "docs/V3_TABLE_PLAN.md",
    "docs/V3_FIGURE_PLAN.md",
    "docs/V3_MANUSCRIPT_FREEZE_SUMMARY.md",
    "analysis/validate_manuscript_freeze.py",
    "tests/test_manuscript_freeze.py"
  ]
}
```

The source SHA is the corrected pre-freeze commit. The containing freeze commit is available from Git history, avoiding a self-referential hash.
