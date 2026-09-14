# V3 claim ledger

Source commit: `0f521a28150fe56f98cdabb3d1cb3664286df124`. This ledger freezes wording and admissibility; it is not manuscript prose. Every quantitative future statement requires a ledger/traceability entry. The complete CSV and JSON are in `results/v3/manuscript_freeze/`.

## Audit and admissibility

```json
{
  "source_commit": "0f521a28150fe56f98cdabb3d1cb3664286df124",
  "status": "completed; prior coverage contradiction repaired in Task A; no further material numerical contradiction found in audited claims",
  "repaired_contradiction": {
    "commit": "0f521a28150fe56f98cdabb3d1cb3664286df124",
    "document": "docs/V3_POST_OPTIMIZATION_DIAGNOSTICS.md",
    "kernel_statements": "47/53 = 88.67924528301887%",
    "runner_statements": "109/182 = 59.89010989010989%",
    "runner_branches": "22/40 = 55%"
  },
  "documentation_reviewed": [
    "docs/V3_TEMPLATE_EVALUATION.md",
    "docs/V3_BAYESIAN_VALIDATION.md",
    "docs/V3_OPTIMIZATION_VALIDATION.md",
    "docs/V3_OPTIMIZATION_VALIDATION_PROTOCOL.md",
    "docs/V3_POST_OPTIMIZATION_DIAGNOSTICS.md",
    "docs/V3_VALIDATION_GAPS.md",
    "paper/ifrs16_lbo_ssrn_v2.tex",
    "paper/REVISION_AUDIT.md"
  ],
  "audit_checks": [
    "Screening raw confusion counts agree with selected/fixed summaries.",
    "All requested Template 004 distressed subtype counts verified against frozen heldout_results.csv.",
    "Reference distressed sorted annualized return at zero-based index 2 is -1, confirming n=21 linear P10 on seven zero-recovery observations.",
    "Each Template 004 regime rate matches raw subtype count divided by cell n.",
    "Reviewed baseline reports exact financial/statistical reproduction, excluding recorded provenance/timing fields.",
    "Bayes recovery and prior-sensitivity diagnostics agree with classification C.",
    "Optimization aggregate, selections, boundary and registry evidence retain classification B.",
    "Coverage count-derived statement/branch percentages agree with corrected prose."
  ],
  "interpretive_resolutions": [
    "Historical stage-end instructions to run later protocols are superseded by this no-experiments freeze request.",
    "The result registry manuscript_eligibility=false is retained: methodological appendix discussion is not substantive result admission.",
    "Lower cash cushion wording is limited to absolute cash; Template 004 does not have the lowest cash/EBITDA.",
    "The Bayesian prose minimum bulk ESS is rounded to 391.5; ledger retains 391.48627301744915 and does not repeat the literal exceeded-391.5 wording.",
    "Early stage coverage/test counts are stage-specific archived verification, not newly executed checks."
  ],
  "v2_disposition": {
    "retained": [
      "synthetic benchmark inputs",
      "fixed screening score",
      "reviewed engine mechanics",
      "pooled ranking point estimate"
    ],
    "qualified": [
      "fixed threshold result replaced by selected/fixed comparison",
      "conditional synthetic performance, not borrower generalization"
    ],
    "omitted_from_v3_results": [
      "scenario-bootstrap AUC interval",
      "cross-model RMSE and timing results",
      "Accor accounting illustration",
      "conditional theoretical bounds without validated benchmark budgets"
    ],
    "prohibited": "All historical unsupported families in prohibited_claims.json"
  },
  "new_experiments": false
}
```

## Reporting rules

```json
{
  "auc_ap": "3 decimals; AP means average precision, not trapezoidal PR area",
  "percentage": "1 decimal percentage point; numerator/denominator for observed subgroup rates",
  "threshold": "6 decimals (retain full precision in ledger and predictions; do not recompute classifications from rounded thresholds)",
  "returns": "2 decimal percentage points; label annualized limited-liability sponsor return",
  "multiples": "2 decimals",
  "coverage": "statement coverage 2 decimals, branch coverage 1 decimal; counts alongside",
  "counts": "integers; zero denominator is undefined, never zero",
  "source_artifacts": "Never round or rewrite; machine snapshots preserve full stored precision",
  "confidence_intervals": "No new intervals; old v2 bootstrap interval is not promoted"
}
```

```json
{
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
}
```

Optimization manuscript eligibility means labelled B/methodological appendix discussion only. Bayesian eligibility means C/exclusion discussion only. Neither overrides the frozen result-admission registry.

## Claims

### ENG-001 — Reviewed mechanics

The reviewed financial-engine mechanics pass the archived independent return/debt checks and runtime invariants on the tested paths.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/foundation_validation.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "independent_return_check"
      ]
    },
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "independent_debt_schedule",
        "exact_match"
      ]
    },
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "runtime_validation"
      ]
    }
  ],
  "unit": "test results and counts",
  "denominator_or_n_if_applicable": {
    "scenarios": 200,
    "scenario_years": 1000
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S4",
    "S11"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "T2",
    "F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### ENG-002 — Runtime scope

All 1,000 benchmark scenario-years across 200 paths passed the recorded runtime invariants; six specified adversarial corruptions were detected.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/foundation_validation.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "runtime_validation"
      ]
    },
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "adversarial_corruptions_detected"
      ]
    }
  ],
  "unit": "counts",
  "denominator_or_n_if_applicable": {
    "scenarios": 200,
    "scenario_years": 1000
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Validation on tested paths is not a proof for every input or economic validity.",
  "manuscript_section": [
    "S4"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### ENG-003 — Independent returns and debt

Independent geometric return/MOIC expectations and the hand-derived debt schedule match production within the recorded tolerance.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/foundation_validation.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "independent_return_check"
      ]
    },
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "independent_debt_schedule"
      ]
    }
  ],
  "unit": "fractions, multiple and currency units",
  "denominator_or_n_if_applicable": {
    "debt_schedule_years": 5
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S4"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### REP-001 — Baseline reproduction

The archived baseline reproduced financial and statistical fields exactly, excluding runtime timing and source-commit metadata.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/baseline/verification.json",
    "results/v3/foundation_validation.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/baseline/verification.json",
      "keys": [
        "financial_and_statistical_differences"
      ]
    },
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "reviewed_v2_reproduction"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Historical verification evidence; no experiment was rerun for this freeze.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### SCR-001 — Central thesis

A screening model can be mechanically correct and rank adverse outcomes well while still failing to provide a stable, portable decision threshold across structurally different synthetic borrower archetypes.

```json
{
  "claim_class": "INTERPRETIVE_SYNTHESIS",
  "evidence_type": "interpretive synthesis of frozen evidence",
  "source_artifact": [
    "results/v3/foundation_validation.json",
    "results/v3/template_evaluation/fold_results.csv",
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "runtime_validation"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "ranking"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/fold_results.csv"
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "A supported possibility illustrated by this synthetic system, not a universal or empirical borrower claim.",
  "manuscript_section": [
    "S1",
    "S2",
    "S11"
  ],
  "prominence": "PRIMARY_CONCEPTUAL_CLAIM",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### SCR-002 — Three validation layers

Engine correctness, ranking performance, and decision usefulness are distinct validation layers: passing one layer does not imply passing the next.

```json
{
  "claim_class": "INTERPRETIVE_SYNTHESIS",
  "evidence_type": "interpretive synthesis of frozen evidence",
  "source_artifact": [
    "results/v3/foundation_validation.json",
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/foundation_validation.json",
      "keys": [
        "runtime_validation"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "ranking"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "selected_threshold"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S1",
    "S2",
    "S11"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### SCR-003 — Synthetic benchmark

The screening evaluation contains 200 synthetic scenarios across five templates, with 20 broad financial failures.

```json
{
  "claim_class": "SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "n_scenarios"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "n_templates"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "n_failures"
      ]
    }
  ],
  "unit": "counts",
  "denominator_or_n_if_applicable": {
    "scenarios": 200,
    "templates": 5,
    "failures": 20,
    "nonfailures": 180
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S5"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "F2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### SCR-004 — Pooled ranking

The fixed screening score has aggregate ROC-AUC 0.947 and average precision 0.781 on the 200 synthetic scenarios (20 failures).

```json
{
  "claim_class": "SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "ranking"
      ]
    }
  ],
  "unit": "unitless",
  "denominator_or_n_if_applicable": {
    "scenarios": 200,
    "failures": 20,
    "nonfailures": 180
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S6",
    "S11"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "T3",
    "F2",
    "A-F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Retains reviewed v2 ranking under the same score-label pairs; excludes its scenario-bootstrap interval from v3.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### SCR-005 — AUC identity

Aggregate held-out and pooled AUC equality is mechanically expected because the fixed score uses the same score-label pairs once each; it is not independent evidence of generalization.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json",
    "results/v3/template_evaluation/verification.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "ranking"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "pooled_recomputed_ranking"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/verification.json",
      "keys": [
        "source_records_equal_archived_json"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S6",
    "S11"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "T3",
    "F2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### SCR-006 — Transferred thresholds

Training-only transferred thresholds yield recall 75.0% (15/20), specificity 78.9% (142/180), 38 false positives and 5 false negatives among 200 scenarios; balanced accuracy is 0.769.

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "selected_threshold"
      ]
    }
  ],
  "unit": "counts and fractions",
  "denominator_or_n_if_applicable": {
    "scenarios": 200,
    "positive": 20,
    "negative": 180,
    "predicted_positive": 53
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S6",
    "S11"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "T3"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### SCR-007 — Fixed threshold

The fixed 0.5 threshold yield recall 15.0% (3/20), specificity 100.0% (180/180), 0 false positives and 17 false negatives among 200 scenarios; balanced accuracy is 0.575.

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "fixed_0_5"
      ]
    }
  ],
  "unit": "counts and fractions",
  "denominator_or_n_if_applicable": {
    "scenarios": 200,
    "positive": 20,
    "negative": 180,
    "predicted_positive": 3
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S6",
    "S11"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "T3"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### SCR-008 — Training-only selection

Thresholds maximize training balanced accuracy using only the other templates; the frozen lowest-threshold tie rule applies, and the score itself is not trained.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "protocol"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "sanity_checks"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Exclusion during threshold fitting cannot undo earlier benchmark design choices.",
  "manuscript_section": [
    "S5"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### SCR-009 — Template 001 transfer

Template 001 misses 4/5 failures among 38 scenarios under the transferred threshold.

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/fold_results.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/fold_results.csv",
      "where": {
        "held_out_template": "SYN_HOTEL_001"
      }
    }
  ],
  "unit": "counts, score thresholds and unitless metrics",
  "denominator_or_n_if_applicable": {
    "scenarios": 38,
    "positive": 5,
    "negative": 33,
    "aggregate_false_positives": 38
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S7"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T4",
    "F3",
    "A-F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### SCR-010 — Template 002 transfer

Template 002 detects 3/3 failures and falsely flags 1/29 non-failures among 32 scenarios under the transferred threshold.

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/fold_results.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/fold_results.csv",
      "where": {
        "held_out_template": "SYN_HOTEL_002"
      }
    }
  ],
  "unit": "counts, score thresholds and unitless metrics",
  "denominator_or_n_if_applicable": {
    "scenarios": 32,
    "positive": 3,
    "negative": 29,
    "aggregate_false_positives": 38
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S7"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T4",
    "F3",
    "A-F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### SCR-011 — Template 003 transfer

Template 003 misses its only failure among 47 scenarios despite within-template AUC 1.000; this ranking estimate contains only one positive.

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/fold_results.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/fold_results.csv",
      "where": {
        "held_out_template": "SYN_HOTEL_003"
      }
    }
  ],
  "unit": "counts, score thresholds and unitless metrics",
  "denominator_or_n_if_applicable": {
    "scenarios": 47,
    "positive": 1,
    "negative": 46,
    "aggregate_false_positives": 38
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S7"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T4",
    "F3",
    "A-F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### SCR-012 — Template 004 transfer

Template 004 contributes 37/38 aggregate false positives. At threshold 0.009678, its specificity is 2.6% (1/38), FPR 97.4% (37/38), and balanced accuracy 0.513 versus 0.636 at fixed 0.5 (49 scenarios; 11 failures).

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/fold_results.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/fold_results.csv",
      "where": {
        "held_out_template": "SYN_HOTEL_004"
      }
    }
  ],
  "unit": "counts, score thresholds and unitless metrics",
  "denominator_or_n_if_applicable": {
    "scenarios": 49,
    "positive": 11,
    "negative": 38,
    "aggregate_false_positives": 38
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S7",
    "S11"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "T4",
    "F3",
    "A-F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### SCR-013 — Template 005 transfer

Template 005 has no failures among 34 scenarios; within-template AUC, average precision and balanced accuracy are undefined and cannot support ranking claims.

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/fold_results.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/fold_results.csv",
      "where": {
        "held_out_template": "SYN_HOTEL_005"
      }
    }
  ],
  "unit": "counts, score thresholds and unitless metrics",
  "denominator_or_n_if_applicable": {
    "scenarios": 34,
    "positive": 0,
    "negative": 34,
    "aggregate_false_positives": 38
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S7"
  ],
  "prominence": "primary",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T4",
    "F3",
    "A-F1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### SCR-014 — Fold dispersion

Fold dispersion is descriptive; overlapping training sets and sparse positives preclude precise population inference. The aggregate is not an equal-weight average of template metrics.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/fold_results.csv",
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "fold_metric_dispersion"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/fold_results.csv",
      "fields": [
        "held_out_template",
        "n_test",
        "n_test_failures"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S7",
    "S10",
    "S11"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "T4"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### SCR-015 — Not calibrated PD

The analytic screening score is a ranking score, not a calibrated probability of default; broad financial failure includes distinct payment, funding and covenant events.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "protocol"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S1",
    "S3",
    "S10",
    "S11"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "Source prose: docs/V3_TEMPLATE_EVALUATION.md, Protocol and provenance; do not collapse the union into payment default.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### SCR-016 — Archetype inputs

The five named hotel archetypes are stipulated synthetic inputs, not observations of five real borrowers.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "data/synthetic/operators.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "data/synthetic/operators.csv"
    }
  ],
  "unit": "input-specific: currency units, fractions, multiple, seed",
  "denominator_or_n_if_applicable": {
    "templates": 5
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Input percentages are stipulated parameters, not sample estimates.",
  "manuscript_section": [
    "S3",
    "S5"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### SCR-017 — No utility conclusion

Improved balanced accuracy and recall do not establish economic superiority without an explicit error-cost or utility model.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/template_evaluation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "protocol"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "selected_threshold"
      ]
    },
    {
      "artifact": "results/v3/template_evaluation/summary.json",
      "keys": [
        "fixed_0_5"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S6",
    "S10",
    "S11"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### BAY-001 — Exclusion decision

Bayesian calibration is classification C and excluded from v3 main results; no posterior-driven LBO integration was performed.

```json
{
  "claim_class": "EXCLUDED_METHOD",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/bayesian_validation/admission_decision.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/bayesian_validation/admission_decision.json"
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Excluded methodological experiment only; this does not imply Bayesian modeling is intrinsically invalid.",
  "manuscript_section": [
    "S9",
    "S10"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "C / excluded methodological experiment",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### BAY-002 — Input provenance

All ten named-firm calibration rows are unverified/stipulated and cannot support empirically validated calibration.

```json
{
  "claim_class": "EXCLUDED_METHOD",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/bayesian_validation/data_provenance.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/bayesian_validation/data_provenance.json"
    }
  ],
  "unit": "count",
  "denominator_or_n_if_applicable": {
    "stipulated_rows": 10
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Excluded methodological experiment; firm names do not establish provenance.",
  "manuscript_section": [
    "S9"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "C / excluded methodological experiment",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### BAY-003 — Failed diagnostic gates

Synthetic recovery did not override failed frozen convergence/effective-sampling and prior-sensitivity gates.

```json
{
  "claim_class": "EXCLUDED_METHOD",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/bayesian_validation/prior_sensitivity.json",
    "results/v3/bayesian_validation/recovery_summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/bayesian_validation/recovery_summary.json"
    },
    {
      "artifact": "results/v3/bayesian_validation/prior_sensitivity.json",
      "keys": [
        "parameter_comparison"
      ]
    },
    {
      "artifact": "results/v3/bayesian_validation/prior_sensitivity.json",
      "keys": [
        "material_prior_sensitivity"
      ]
    }
  ],
  "unit": "coverage fractions, relative RMSE, R-hat, ESS and posterior-SD shifts",
  "denominator_or_n_if_applicable": {
    "recovery_runs": 18,
    "parameter_evaluations": 180,
    "prior_fits": 2
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Appendix exclusion diagnostics only; no substantive empirical or predictive result.",
  "manuscript_section": [
    "S9"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "C / excluded methodological experiment",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### OPT-001 — Methodological admission

The financing-design experiment is classification B: retain only as a toy or methodological demonstration.

```json
{
  "claim_class": "EXCLUDED_METHOD",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/admission_decision.json",
    "results/v3/result_validation/optimization.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/admission_decision.json"
    },
    {
      "artifact": "results/v3/result_validation/optimization.json",
      "keys": [
        "notes"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Appendix demonstration only; not a substantive financing result.",
  "manuscript_section": [
    "S8",
    "S10"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": true,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T5"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "The existing result registry manuscript_eligibility=false bars substantive result promotion. Here manuscript_eligible=true permits explicitly labelled methodological appendix discussion only; admission is unchanged.",
  "admission_status": "B / methodological demonstration only",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### OPT-002 — Separate valuation experiment

The reviewed v2 benchmark uses revenue-based entry EV and EBITDA-based exit EV and is unsuitable for return optimization; the separate v3 demonstration uses EBITDA for both valuation bases.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/economic_audit.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/economic_audit.json"
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Coherent EBITDA bases do not eliminate entry/exit lease-treatment limitations or establish market calibration.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### OPT-003 — Aggregate demonstration

In the methodological demonstration, optimized/reference median annualized sponsor returns are 1.10%/0.61%; broad failure is 1.4% (7/500)/6.8% (34/500), and payment default 0.6% (3/500)/5.0% (25/500).

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/summary.json",
      "keys": [
        "primary_aggregate_heldout"
      ]
    }
  ],
  "unit": "fractions, multiples and currency units",
  "denominator_or_n_if_applicable": {
    "scenarios_per_policy": 500,
    "unique_scenarios": 500
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological only; paired synthetic paths; not economic optimality or empirical uplift.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T5"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### OPT-004 — Boundary policies

All five folds select the 1.50x minimum debt boundary, and four of five select the 70% maximum sweep.

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/fold_selection.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/fold_selection.csv",
      "fields": [
        "held_out_template",
        "selected_debt_multiple",
        "selected_amortisation_rate",
        "selected_cash_sweep",
        "train_objective",
        "train_reference_n_scenarios",
        "test_optimized_n_scenarios"
      ]
    }
  ],
  "unit": "multiple, policy fractions and return fraction",
  "denominator_or_n_if_applicable": {
    "folds": 5,
    "training_scenarios_per_fold": 400,
    "heldout_scenarios_per_fold": 100
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological only; grid boundary selection is not a substantive capital-structure optimum.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T5",
    "A-F2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### OPT-005 — Feasible observed boundary profile

All five folds show decreasing best-feasible objectives across feasible observed debt levels, supporting a systematic within-grid preference for lower leverage; no result identifies the true optimum below 1.50x.

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/debt_boundary_profile.csv",
    "results/v3/post_optimization_diagnostics/debt_boundary_summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/debt_boundary_summary.json"
    },
    {
      "artifact": "results/v3/post_optimization_diagnostics/debt_boundary_profile.csv",
      "fields": [
        "held_out_template",
        "debt_multiple",
        "feasible_candidate_exists",
        "best_feasible_median_annualized_return"
      ]
    }
  ],
  "unit": "debt multiples and return fractions",
  "denominator_or_n_if_applicable": {
    "folds": 5
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological only; never extend monotonicity to unobserved or infeasible levels.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A-F2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### OPT-006 — Fixed-rate sensitivity

The fixed-rate sensitivity changes policy components in 2/5 folds, meeting the frozen material-sensitivity criterion.

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/admission_decision.json",
    "results/v3/optimization_validation/rate_sensitivity.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/admission_decision.json",
      "keys": [
        "rate_sensitivity_changed_policy_folds"
      ]
    },
    {
      "artifact": "results/v3/optimization_validation/admission_decision.json",
      "keys": [
        "material_rate_sensitivity"
      ]
    },
    {
      "artifact": "results/v3/optimization_validation/rate_sensitivity.csv"
    }
  ],
  "unit": "policy fractions and return fractions",
  "denominator_or_n_if_applicable": {
    "folds": 5,
    "heldout_scenarios_per_fold": 100,
    "training_scenarios_per_fold": 400
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological only; fixed-rate sensitivity is not endogenous credit pricing.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### OPT-007 — Economic limitations

The demonstration omits leverage-dependent pricing and a recovery process, continues after default, and retains asymmetric entry/exit lease treatment.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/economic_audit.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/economic_audit.json",
      "keys": [
        "remaining_economic_limitations"
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological only; default-path returns are mechanical diagnostics, not recovery estimates.",
  "manuscript_section": [
    "S8",
    "S10"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### OPT-008 — Optimization mechanics

The independent known-optimum toy, training-only selection checks and runtime invariants support the search mechanics under the frozen grid.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/summary.json",
    "results/v3/optimization_validation/toy_validation.json",
    "results/v3/optimization_validation/verification.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/toy_validation.json"
    },
    {
      "artifact": "results/v3/optimization_validation/summary.json",
      "keys": [
        "sanity_checks"
      ]
    },
    {
      "artifact": "results/v3/optimization_validation/verification.json",
      "keys": [
        "validated_simulation_paths"
      ]
    }
  ],
  "unit": "toy debt, MOIC and path count",
  "denominator_or_n_if_applicable": {
    "validated_paths": 113000
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological only; mechanical validity does not establish economic optimality.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### OPT-009 — Focused optimization coverage

Focused coverage is 88.68% statements (47/53) and 70.0% branches (14/20) for financing_policy; runner coverage is 59.89% statements (109/182) and 55.0% branches (22/40). Coverage alone does not validate the optimizer.

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/coverage.json",
    "results/v3/post_optimization_diagnostics/optimization_code_coverage.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/optimization_code_coverage.json",
      "keys": [
        "statement_coverage"
      ]
    },
    {
      "artifact": "results/v3/post_optimization_diagnostics/optimization_code_coverage.json",
      "keys": [
        "branch_coverage"
      ]
    },
    {
      "artifact": "results/v3/post_optimization_diagnostics/coverage.json",
      "keys": [
        "files",
        "analysis\\optimization\\financing_policy.py",
        "summary"
      ]
    },
    {
      "artifact": "results/v3/post_optimization_diagnostics/coverage.json",
      "keys": [
        "files",
        "analysis\\run_v3_optimization.py",
        "summary"
      ]
    }
  ],
  "unit": "percentage points and counts",
  "denominator_or_n_if_applicable": {
    "kernel_statements": 53,
    "kernel_branches": 20,
    "runner_statements": 182,
    "runner_branches": 40
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological coverage diagnostic; statement and combined coverage are distinct.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A3"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Supersedes stale 84%/59% statement and 85% runner branch wording corrected in 0f521a28150fe56f98cdabb3d1cb3664286df124",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-001 — Within-system convergence

Template 004 shows weak threshold transfer and comparatively weak financing-policy outcomes, providing convergent descriptive evidence within the same synthetic system consistent with structural heterogeneity affecting downstream transfer.

```json
{
  "claim_class": "INTERPRETIVE_SYNTHESIS",
  "evidence_type": "interpretive synthesis of frozen evidence",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/cross_stage_template004.json",
    "results/v3/post_optimization_diagnostics/regime_conditioned_template_comparison.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/cross_stage_template004.json"
    },
    {
      "artifact": "results/v3/post_optimization_diagnostics/regime_conditioned_template_comparison.csv"
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Same synthetic archetype system only; descriptive/convergent, not independent replication, external validation, causal evidence or empirical borrower evidence. Small regime-conditioned cells, including distressed n=21; show n with percentages.",
  "manuscript_section": [
    "S7",
    "S11"
  ],
  "prominence": "secondary",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": true,
  "figure_or_table_candidate": [
    "A-F3"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "Financing evidence retains B/methodological status; it is secondary support, not part of the primary thesis. Wording uses consistency rather than causal attribution.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### HET-002 — Regime mix

Template 004 has base/downside/distressed counts 48/100, 31/100 and 21/100, close to stipulated 50%/30%/20%; weakness persists within regimes, so the mix alone does not explain it.

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/diagnostic_summary.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/diagnostic_summary.json",
      "keys": [
        "template004_regime_breakdown",
        "unique_regime_counts"
      ]
    },
    {
      "artifact": "results/v3/post_optimization_diagnostics/diagnostic_summary.json",
      "keys": [
        "template004_regime_breakdown",
        "intended_probabilities"
      ]
    }
  ],
  "unit": "counts and stipulated probabilities",
  "denominator_or_n_if_applicable": {
    "unique_scenarios": 100
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Descriptive synthetic comparison, not a causal decomposition; financing evidence is B/methodological.",
  "manuscript_section": [
    "S7"
  ],
  "prominence": "secondary",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-003 — Template 004 optimized base

Template 004 optimized base (n=48): median annualized sponsor return 2.58%; broad failure 0/48 (0.0%); payment default 0/48 (0.0%); insolvency 0/48 (0.0%); covenant breach 0/48 (0.0%); total equity loss 0/48 (0.0%).

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv",
      "where": {
        "evaluation_policy": "optimized",
        "scenario_type": "base"
      }
    }
  ],
  "unit": "return/risk fractions and counts",
  "denominator_or_n_if_applicable": {
    "n": 48,
    "counts": {
      "broad_failure": 0,
      "payment_default": 0,
      "insolvency": 0,
      "covenant_breach": 0,
      "total_equity_loss": 0
    }
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological financing evidence; descriptive small cell, no manufactured confidence interval.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-004 — Template 004 optimized downside

Template 004 optimized downside (n=31): median annualized sponsor return -7.84%; broad failure 0/31 (0.0%); payment default 0/31 (0.0%); insolvency 0/31 (0.0%); covenant breach 0/31 (0.0%); total equity loss 0/31 (0.0%).

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv",
      "where": {
        "evaluation_policy": "optimized",
        "scenario_type": "downside"
      }
    }
  ],
  "unit": "return/risk fractions and counts",
  "denominator_or_n_if_applicable": {
    "n": 31,
    "counts": {
      "broad_failure": 0,
      "payment_default": 0,
      "insolvency": 0,
      "covenant_breach": 0,
      "total_equity_loss": 0
    }
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological financing evidence; descriptive small cell, no manufactured confidence interval.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-005 — Template 004 optimized distressed

Template 004 optimized distressed (n=21): median annualized sponsor return -18.85%; broad failure 7/21 (33.3%); payment default 3/21 (14.3%); insolvency 6/21 (28.6%); covenant breach 7/21 (33.3%); total equity loss 1/21 (4.8%).

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv",
      "where": {
        "evaluation_policy": "optimized",
        "scenario_type": "distressed"
      }
    }
  ],
  "unit": "return/risk fractions and counts",
  "denominator_or_n_if_applicable": {
    "n": 21,
    "counts": {
      "broad_failure": 7,
      "payment_default": 3,
      "insolvency": 6,
      "covenant_breach": 7,
      "total_equity_loss": 1
    }
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological financing evidence; descriptive small cell, no manufactured confidence interval.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-006 — Template 004 reference base

Template 004 reference base (n=48): median annualized sponsor return 2.16%; broad failure 1/48 (2.1%); payment default 1/48 (2.1%); insolvency 1/48 (2.1%); covenant breach 1/48 (2.1%); total equity loss 0/48 (0.0%).

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv",
      "where": {
        "evaluation_policy": "reference",
        "scenario_type": "base"
      }
    }
  ],
  "unit": "return/risk fractions and counts",
  "denominator_or_n_if_applicable": {
    "n": 48,
    "counts": {
      "broad_failure": 1,
      "payment_default": 1,
      "insolvency": 1,
      "covenant_breach": 1,
      "total_equity_loss": 0
    }
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological financing evidence; descriptive small cell, no manufactured confidence interval.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-007 — Template 004 reference downside

Template 004 reference downside (n=31): median annualized sponsor return -11.44%; broad failure 9/31 (29.0%); payment default 7/31 (22.6%); insolvency 9/31 (29.0%); covenant breach 3/31 (9.7%); total equity loss 1/31 (3.2%).

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv",
      "where": {
        "evaluation_policy": "reference",
        "scenario_type": "downside"
      }
    }
  ],
  "unit": "return/risk fractions and counts",
  "denominator_or_n_if_applicable": {
    "n": 31,
    "counts": {
      "broad_failure": 9,
      "payment_default": 7,
      "insolvency": 9,
      "covenant_breach": 3,
      "total_equity_loss": 1
    }
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological financing evidence; descriptive small cell, no manufactured confidence interval.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-008 — Template 004 reference distressed

Template 004 reference distressed (n=21): median annualized sponsor return -30.00%; broad failure 18/21 (85.7%); payment default 16/21 (76.2%); insolvency 18/21 (85.7%); covenant breach 14/21 (66.7%); total equity loss 7/21 (33.3%).

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv",
      "where": {
        "evaluation_policy": "reference",
        "scenario_type": "distressed"
      }
    }
  ],
  "unit": "return/risk fractions and counts",
  "denominator_or_n_if_applicable": {
    "n": 21,
    "counts": {
      "broad_failure": 18,
      "payment_default": 16,
      "insolvency": 18,
      "covenant_breach": 14,
      "total_equity_loss": 7
    }
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological financing evidence; descriptive small cell, no manufactured confidence interval.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-009 — Within-regime comparison

Under selected policies, Template 004 has the lowest median annualized sponsor return among the five templates in each realized regime; base/downside broad-failure rates tie at zero, so weakness is not strict dominance on every risk metric.

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/post_optimization_diagnostics/regime_conditioned_template_comparison.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/regime_conditioned_template_comparison.csv",
      "where": {
        "evaluation_policy": "optimized"
      }
    }
  ],
  "unit": "return/risk fractions",
  "denominator_or_n_if_applicable": {
    "by_template_base_downside_distressed": {
      "001": [
        41,
        41,
        18
      ],
      "002": [
        62,
        24,
        14
      ],
      "003": [
        55,
        29,
        16
      ],
      "004": [
        48,
        31,
        21
      ],
      "005": [
        42,
        42,
        16
      ]
    }
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological; descriptive within-system comparison with small unequal cells. Not a causal attribution.",
  "manuscript_section": [
    "S7",
    "S12"
  ],
  "prominence": "secondary",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2",
    "A-F3"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### HET-010 — Structural input profile

Template 004 has the highest opening debt/EBITDA and the lowest absolute opening cash among these stipulated archetypes; its cash/EBITDA ratio is not the lowest.

```json
{
  "claim_class": "DESCRIPTIVE_DIAGNOSTIC",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "data/synthetic/operators.csv",
    "results/v3/post_optimization_diagnostics/template004_structural_profile.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_structural_profile.json",
      "keys": [
        "all_templates"
      ]
    },
    {
      "artifact": "data/synthetic/operators.csv",
      "fields": [
        "operator_id",
        "cash_0",
        "ebitda_0",
        "financial_debt_0"
      ]
    }
  ],
  "unit": "currency units and ratios",
  "denominator_or_n_if_applicable": {
    "templates": 5
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Absolute cash and cash scaled by EBITDA must not be conflated; no causal identification.",
  "manuscript_section": [
    "S7"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T1"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### LIM-001 — Limited liability

Sponsor exit proceeds equal max(0, raw exit equity); MOIC divides those proceeds by positive initial sponsor equity. Zero recovery gives MOIC 0 and annualized sponsor return -1.0 (-100%) under the predeclared convention.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/frozen_protocol.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/frozen_protocol.json",
      "keys": [
        "limited_liability"
      ]
    },
    {
      "artifact": "results/v3/optimization_validation/frozen_protocol.json",
      "keys": [
        "years"
      ]
    }
  ],
  "unit": "return fraction and multiple",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Analysis-layer limited-liability sponsor return, not unconstrained financial IRR. Raw engine equity remains unfloored; no solver-failure interpretation.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### LIM-002 — Distressed P10 floor

Template 004 reference distressed P10 annualized sponsor return is -100% (n=21): 7/21 scenarios (33.3%) have zero recovery, placing the 10th percentile on the zero-recovery mass, consistently with the predeclared convention.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/heldout_results.csv",
    "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/post_optimization_diagnostics/template004_regime_breakdown.csv",
      "where": {
        "evaluation_policy": "reference",
        "scenario_type": "distressed"
      },
      "fields": [
        "scenario_count",
        "p10_annualized_return",
        "total_equity_loss_rate"
      ]
    },
    {
      "artifact": "results/v3/optimization_validation/heldout_results.csv",
      "where": {
        "held_out_template": "SYN_HOTEL_004",
        "evaluation_policy": "reference",
        "scenario_type": "distressed"
      },
      "fields": [
        "scenario_id",
        "raw_exit_equity",
        "sponsor_exit_proceeds",
        "sponsor_moic",
        "sponsor_annualized_return",
        "total_equity_loss"
      ]
    }
  ],
  "unit": "return fraction",
  "denominator_or_n_if_applicable": {
    "n": 21,
    "zero_recovery": 7
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Methods/table note only; B/methodological financing convention, not an empirical recovery estimate.",
  "manuscript_section": [
    "S12"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "A2"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### REP-002 — Small-n rule

Every subgroup/regime/fold percentage must carry n or numerator/denominator in the same sentence, table row, table note or immediately adjacent text; no new confidence intervals may be manufactured.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Mandatory manuscript reporting rule, especially Template 004 distressed n=21; distinguish policy parameters from sample rates.",
  "manuscript_section": [
    "S10"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### NEG-001 — Historical scope

The v2 Accor reconstruction, cross-model RMSE comparisons and conditional mathematical discussion are not promoted into the v3 results; the central study concerns synthetic ranking and threshold transfer.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Conditional on the frozen synthetic system; no borrower-population inference.",
  "manuscript_section": [
    "S10"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "V2 sections on Accor, cross-model discrepancy, timing and conditional theory are omitted from the final results architecture.",
  "notes": "Audited paper/ifrs16_lbo_ssrn_v2.tex and paper/REVISION_AUDIT.md. Conditional theory is not disproved, but no benchmark error-bound validation is admitted.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### OPT-010 — Financing protocol and fold outcomes

The financing demonstration uses a frozen common policy grid and training-only selection, evaluated on the same held-out operating paths under selected and reference policies; all fold results remain methodological.

```json
{
  "claim_class": "CONDITIONAL_SYNTHETIC_RESULT",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "results/v3/optimization_validation/fold_selection.csv",
    "results/v3/optimization_validation/frozen_protocol.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "results/v3/optimization_validation/frozen_protocol.json"
    },
    {
      "artifact": "results/v3/optimization_validation/fold_selection.csv"
    }
  ],
  "unit": "protocol-specific multiples, fractions, counts and currency units",
  "denominator_or_n_if_applicable": {
    "templates": 5,
    "heldout_scenarios_per_policy_per_fold": 100,
    "training_scenarios_per_fold": 400
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "B/methodological only; no substantive financing optimum, independent replication or empirical uplift.",
  "manuscript_section": [
    "S8"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [
    "T5"
  ],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "B / methodological demonstration only",
  "quantitative": true,
  "subgroup_percentage": true
}
```

### SCR-018 — Score and failure convention

Over five annual observations, broad financial failure is the union of nonpositive EBITDA, payment default, reserve funding deficit, leverage above 6.0 or ICR below 1.8. The score is 1/(1+exp(2H)), where H is minimum analytic headroom against those ratio thresholds. Simulated covenant breach is strict; score classification uses score >= threshold.

```json
{
  "claim_class": "VERIFIED_MECHANICAL",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "analysis/run_benchmark.py",
    "results/v3/baseline/reproduced_manifest.json"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "analysis/run_benchmark.py",
      "lines": [
        340,
        368
      ]
    },
    {
      "artifact": "results/v3/baseline/reproduced_manifest.json",
      "keys": [
        "horizon_years"
      ]
    }
  ],
  "unit": "years, ratio thresholds and score formula",
  "denominator_or_n_if_applicable": {
    "observations_per_scenario": 5
  },
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Stipulated benchmark conventions, not actual loan terms or calibrated default probabilities.",
  "manuscript_section": [
    "S3"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": true,
  "subgroup_percentage": false
}
```

### SCR-019 — Different model families

The analytic screening path and financial simulation differ in timing, cash, financing and input conventions; engine validation does not establish that the screening path approximates the simulator within proved error bounds.

```json
{
  "claim_class": "LIMITATION",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [
    "analysis/run_benchmark.py"
  ],
  "source_fields_or_rows": [
    {
      "artifact": "analysis/run_benchmark.py",
      "lines": [
        220,
        310
      ]
    }
  ],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "No attribution of all cross-model discrepancy to approximation error; no benchmark-wide theoretical guarantee.",
  "manuscript_section": [
    "S3"
  ],
  "prominence": "supporting",
  "manuscript_eligible": true,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "No prior numerical claim imported.",
  "notes": "",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-001 — V1 AUC

Prohibited as a current substantive result: AUC 0.76 [0.71, 0.81].

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Use SCR-004; no protocol-equivalence claim.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-002 — V1 comparator RMSE

Prohibited as a current substantive result: RMSE 0.28 versus 0.52.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No admitted comparator improvement.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-003 — V1 RMSE reduction

Prohibited as a current substantive result: 46% RMSE reduction.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No current paired comparator.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-004 — Historical return uplift

Prohibited as a current substantive result: +3.4 percentage-point IRR uplift.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: OPT-003 only as B/methodological.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-005 — Validated Bayesian calibration

Prohibited as a current substantive result: Bayesian inputs are empirically validated.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: BAY-001 through BAY-003.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-006 — Posterior LBO integration

Prohibited as a current substantive result: Posterior-driven LBO results.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No posterior integration was performed.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-007 — Calibrated score

Prohibited as a current substantive result: Screening score is calibrated PD.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: SCR-015.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-008 — Independent AUC generalization

Prohibited as a current substantive result: Pooled/held-out AUC equality independently validates generalization.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: SCR-005.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-009 — Real hotel breaches

Prohibited as a current substantive result: Synthetic scenarios establish real hotel covenant breaches.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Synthetic broad failure only.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-010 — Audited Accor transaction

Prohibited as a current substantive result: Accor reconstruction is audited transaction evidence.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Accor reconstruction is omitted; no such provenance.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-011 — Substantive leverage optimum

Prohibited as a current substantive result: 1.50x debt is economically optimal.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: OPT-004 and OPT-005.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-012 — Empirical optimization

Prohibited as a current substantive result: Financing experiment is empirically calibrated.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: OPT-001 and OPT-007.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-013 — External replication

Prohibited as a current substantive result: Template 004 provides external validation or independent replication.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: HET-001 within-system convergence only.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-014 — Causal heterogeneity

Prohibited as a current substantive result: Structural heterogeneity is causally identified.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: HET-001 descriptive consistency only.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-015 — Real borrower transfer

Prohibited as a current substantive result: Threshold instability is established for real borrowers.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: SCR-001 conditional synthetic thesis.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-016 — Missing subgroup n

Prohibited as a current substantive result: Subgroup/regime/fold percentages without n.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: REP-002.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-017 — Misinterpreted floor

Prohibited as a current substantive result: -100% sponsor return is a bound on raw equity, unconstrained IRR, or solver failure.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: LIM-001 and LIM-002.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-018 — Stale kernel coverage

Prohibited as a current substantive result: 84% statement coverage for financing_policy.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: 88.68% = 47/53; OPT-009.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-019 — Stale runner statements

Prohibited as a current substantive result: 59% statement coverage for runner.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: 59.89% = 109/182; OPT-009.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-020 — Stale runner branches

Prohibited as a current substantive result: 85% branch coverage for runner.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: 55.0% = 22/40; OPT-009.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-021 — Bayes universally invalid

Prohibited as a current substantive result: Exclusion proves Bayes intrinsically invalid.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Only this experiment fails admission.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-022 — Unsupported theoretical guarantee

Prohibited as a current substantive result: Theoretical model or deterministic error bounds validated by current benchmark.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No benchmark error-budget certification admitted; conditional mathematical discussion is omitted.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-023 — Historical AUC improvement

Prohibited as a current substantive result: +0.18 AUC and [0.12,0.24] interval.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No current comparator.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-024 — Historical paired error change

Prohibited as a current substantive result: Delta RMSE -0.24 and [-0.31,-0.17] interval.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No current paired comparator.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-025 — Historical expected return

Prohibited as a current substantive result: 19.6% expected IRR and [18.2,21.1] interval.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No historical return estimate admitted.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-026 — Historical traditional comparator

Prohibited as a current substantive result: Traditional AUC .58, RMSE .52, IRR 16.2% or intervals.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No executed current comparator.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-027 — Historical naive comparator

Prohibited as a current substantive result: Naive AUC .64, RMSE .45, IRR 17.1% or intervals.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No executed current comparator.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-028 — Historical Bayesian comparator

Prohibited as a current substantive result: Bayesian AUC .72, RMSE .34, IRR 18.4% or intervals.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Bayesian experiment excluded.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-029 — Historical screening sample

Prohibited as a current substantive result: Ten operators with fifty scenarios each for screening.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: SCR-003; separate financing sample must not be conflated.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-030 — Quarterly evaluation

Prohibited as a current substantive result: Twenty quarterly observations in current benchmark.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Five annual observations per screening scenario.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-031 — Borrower geography and scale

Prohibited as a current substantive result: Historical geographic borrower sample and $1.2-4.8B scale.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Stipulated synthetic archetypes only.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-032 — Cluster bootstrap

Prohibited as a current substantive result: 2,000 stratified operator-clustered bootstrap samples.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: SCR-014 descriptive fold dispersion.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-033 — BCa agreement

Prohibited as a current substantive result: BCa checks agree within 0.01 AUC.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No such current verification.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-034 — RMSE intervals

Prohibited as a current substantive result: Clustered RMSE confidence intervals.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No admitted current intervals.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-035 — Historical split

Prohibited as a current substantive result: 70/30 split or held-out quarterly evaluation.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Training-only leave-one-template-out thresholds.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-036 — Within-operator CV

Prohibited as a current substantive result: Five-fold within-operator cross-validation.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Leave-one-template-out evaluation, not historical CV.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-037 — Equal template weighting

Prohibited as a current substantive result: Aggregate screening metrics equally weight templates.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Rows weighted equally; template counts differ.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-038 — Significance test

Prohibited as a current substantive result: p < 0.001 clustered permutation result.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No significance claim.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-039 — Certification percentages

Prohibited as a current substantive result: Historical time-point/scenario certification, false non-certification, bound utilization or excess-conservatism percentages.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No current certification experiment.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-040 — Validated discrepancy bounds

Prohibited as a current substantive result: Historical ICR .089/.120 or leverage .112/.143 as proved benchmark bounds.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No validated matched-model error budget.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-041 — Small relative error

Prohibited as a current substantive result: Median less than 3% relative error.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No retained result.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-042 — Posterior frontiers

Prohibited as a current substantive result: Posterior frontiers or 80%/95% credible financing bands.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No posterior integration.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-043 — Historical intervention sensitivities

Prohibited as a current substantive result: Quarterly/annual, cure, hedge or floor sensitivity return/breach/RMSE tables.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Not current experiments; OPT-006 is fixed-term-rate sensitivity only.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-044 — Historical optimized breach plot

Prohibited as a current substantive result: V2 breach composition measures optimized versus baseline financing.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: V2 flags describe synthetic failure composition.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-045 — Causal stress thresholds

Prohibited as a current substantive result: Failure caused above historical volatility/leverage/lease thresholds.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: No causal threshold experiment.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-046 — Bayesian benchmark generator

Prohibited as a current substantive result: Bayesian data-informed priors generated screening benchmark.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Fixed synthetic generator; no Bayesian integration.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-047 — Pure approximation error

Prohibited as a current substantive result: Cross-model differences measure only analytic approximation error.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Structural, timing and input/convention differences also contribute; omitted from v3 results.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-048 — V2 precise AUC interval

Prohibited as a current substantive result: V2 scenario-bootstrap interval is v3 template-transfer uncertainty.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Descriptive fold dispersion; no manufactured interval.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-049 — Unobserved leverage curve

Prohibited as a current substantive result: Debt objective decreases below 1.50x or across infeasible levels.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Feasible observed levels only; OPT-005.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```

### PROH-050 — Coverage proves validity

Prohibited as a current substantive result: Coverage alone validates the optimizer.

```json
{
  "claim_class": "PROHIBITED_CLAIM",
  "evidence_type": "frozen artifact audit",
  "source_artifact": [],
  "source_fields_or_rows": [],
  "unit": "not applicable",
  "denominator_or_n_if_applicable": null,
  "uncertainty_status": "Descriptive frozen realization; no new interval or significance test. Mechanical checks are conditional on tested paths.",
  "qualification_required": "Must not appear as an affirmative current result.",
  "manuscript_section": [],
  "prominence": "supporting",
  "manuscript_eligible": false,
  "abstract_eligible": false,
  "conclusion_eligible": false,
  "figure_or_table_candidate": [],
  "prohibited_extensions": "No empirical, causal, calibrated-PD, universal-generalization or substantive financing-optimum extension.",
  "supersedes_prior_claim": "Historical or overextended claim explicitly barred.",
  "notes": "Replacement: Independent toy, targeted checks and runtime evidence; OPT-008/009.",
  "admission_status": "not applicable",
  "quantitative": false,
  "subgroup_percentage": false
}
```
