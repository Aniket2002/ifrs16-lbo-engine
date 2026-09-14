"""Adversarial checks for the manuscript metadata gate; no numerical experiments."""

import copy
import json
from pathlib import Path

import pytest

from analysis.validate_manuscript_freeze import validate, validate_claims

ROOT = Path(__file__).resolve().parents[1]


def claim(cid):
    path = ROOT / "results/v3/manuscript_freeze/claim_ledger.json"
    return copy.deepcopy(
        next(c for c in json.loads(path.read_text(encoding="utf-8")) if c["claim_id"] == cid)
    )


def test_complete_frozen_metadata_and_sources():
    report = validate(ROOT)
    assert report["passed"], report["errors"]


@pytest.mark.parametrize(
    ("cid", "key", "value", "error"),
    [
        ("SCR-004", "source_fields_or_rows", [], "no source"),
        ("SCR-004", "numerical_value_if_any", [0.76], "snapshot mismatch"),
        ("HET-005", "denominator_or_n_if_applicable", None, "missing denominator"),
        ("SCR-004", "claim_class", "UNVALIDATED", "admissibility class"),
        ("BAY-001", "admission_status", "A", "Bayesian admission"),
        ("OPT-001", "admission_status", "A", "optimization admission"),
        ("PROH-001", "manuscript_eligible", True, "prohibited claim promoted"),
        ("OPT-003", "abstract_eligible", True, "numerical result promoted"),
        ("OPT-009", "exact_allowed_wording", "85% branch coverage", "stale coverage"),
    ],
)
def test_rejects_invalid_admission_or_trace_metadata(cid, key, value, error):
    row = claim(cid)
    row[key] = value
    assert any(error in e for e in validate_claims(ROOT, [row]))


def test_rejects_nonexistent_csv_row_selector():
    row = claim("SCR-012")
    row["source_fields_or_rows"][0]["where"]["held_out_template"] = "MISSING"
    assert any("matched no rows" in e for e in validate_claims(ROOT, [row]))


def test_rejects_duplicate_claim_ids():
    row = claim("SCR-004")
    assert any("duplicate" in e for e in validate_claims(ROOT, [row, row]))


def test_rejects_missing_required_field():
    row = claim("SCR-004")
    del row["qualification_required"]
    assert any("missing fields" in e for e in validate_claims(ROOT, [row]))
