"""Frozen rendering and adversarial manuscript checks, without model execution."""

import pytest

from analysis.scripts.audit_manuscript_v3 import scan_prohibited, source_audit
from analysis.scripts.render_manuscript_v3 import (
    FREEZE,
    Renderer,
    count_rate,
    display,
    empirical_curves,
    load,
)


def test_frozen_rounding_and_undefined():
    assert display(0.9472222222222222) == "0.947"
    assert display(0.00967805410948959, "threshold") == "0.009678"
    assert display(-1, "return") == r"-100.00\%"
    assert display("") == "undef."
    assert count_rate(7 / 21, 21) == r"7/21 (33.3\%)"


def test_rejects_manufactured_event_count():
    with pytest.raises(ValueError, match="integer count"):
        count_rate(0.35, 21)


def test_tied_scores_have_one_threshold_step_and_order_invariance():
    expected = ([0.0, 0.5, 1.0], [0.0, 1.0, 1.0], [1.0, 0.5, 1 / 3])
    assert empirical_curves([0.8, 0.8, 0.2], [1, 0, 0]) == expected
    assert empirical_curves([0.2, 0.8, 0.8], [0, 0, 1]) == expected


def test_single_class_curve_rejected():
    with pytest.raises(ValueError, match="both classes"):
        empirical_curves([0.1, 0.8], [0, 0])


def test_affirmative_prohibited_phrase_rejected():
    assert not scan_prohibited("The score robustly generalizes.")["passed"]
    assert not scan_prohibited("We report AUC 0.76.")["passed"]
    assert scan_prohibited("The score is not a calibrated probability of default.")["passed"]


def audit_snippet(body):
    source = "\\begin{abstract}\n\\end{abstract}\n" + body + "\n\\section{Conclusion}\n\\appendix\n"
    return source_audit(source, load(FREEZE + "claim_ledger.json"), {})


def test_untraced_numerical_value_rejected():
    audit = audit_snippet("% Claims: SCR-004\nThe AUC is 98765.4321.")
    assert not audit["passed"]
    assert any("untraced number" in e for e in audit["errors"])


def test_subgroup_percentage_without_n_rejected():
    audit = audit_snippet("% Claims: HET-005\nBroad failure is 33.3\\%.")
    assert not audit["passed"]
    assert any("lacks adjacent n" in e for e in audit["errors"])


def test_prohibited_claim_annotation_rejected():
    audit = audit_snippet("% Claims: PROH-001\nThis is a finding.")
    assert not audit["passed"]


def test_render_is_deterministic_and_preserves_frozen_inputs(tmp_path):
    first = Renderer(tmp_path / "first").run()
    second = Renderer(tmp_path / "second").run()
    assert first["output_sha256"] == second["output_sha256"]
    assert len(first["tables"]) == 7
    assert len(first["figures"]) == 6
    assert first["non_public_coverage_metadata"]["search_kernel"] == {
        "statement_coverage_percent": 88.68,
        "branch_coverage_percent": 70.0,
    }
    assert first["non_public_coverage_metadata"]["runner"] == {
        "statement_coverage_percent": 59.89,
        "branch_coverage_percent": 55.0,
    }
    assert first["protected_hash_check"]["passed"]
