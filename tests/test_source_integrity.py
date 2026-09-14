"""Regression tests for canonical Git-blob and historical integrity modes."""

import hashlib
import json
import subprocess
from pathlib import Path

from analysis.source_integrity import (
    HISTORICAL_MANIFEST,
    audit_historical_manifest,
    read_git_blob,
    validate_current_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "8e64d125eaa8fcb4edd3a138ae3048eaa6dea928"
UNRESOLVED = [
    "results/v3/baseline/pipeline_step_2.txt",
    "results/v3/foundation_validation.json",
]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def commit_all(repo, message):
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Integrity Test")
    git(repo, "config", "user.email", "integrity@example.invalid")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "protected.txt").write_bytes(b"alpha\nbeta\n")
    (repo / "protected.bin").write_bytes(b"\x00\xff\x10\r\n")
    historical = repo / "historical.json"
    historical.write_bytes(b'{"immutable":true}\n')
    baseline = commit_all(repo, "baseline")

    def entry(path):
        object_id, content = read_git_blob(repo, baseline, path)
        return {
            "path": path,
            "git_blob_sha": object_id,
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_length": len(content),
        }

    historical_entry = entry("historical.json")
    manifest = {
        "schema_version": 1,
        "purpose": "test",
        "created_at": "2026-09-15T00:00:00Z",
        "current_integrity_commit": baseline,
        "historical_freeze_manifest_path": "historical.json",
        "historical_freeze_manifest_git_blob_sha": historical_entry["git_blob_sha"],
        "historical_freeze_manifest_sha256": historical_entry["sha256"],
        "historical_freeze_manifest_byte_length": historical_entry["byte_length"],
        "historical_freeze_source_commit": baseline,
        "contract": "SHA-256 of canonical Git blob bytes",
        "protected_files": [entry("protected.txt"), entry("protected.bin")],
        "historical_manifest_status": {
            "total_entries": 2,
            "reproducible_from_git_or_eol": 2,
            "unresolved_historical_entries": 0,
        },
        "unresolved_historical_paths": [],
        "note": "test",
    }
    manifest_path = repo / "current.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return repo, manifest_path, baseline


def test_current_git_blob_verification_passes_for_all_protected_sources():
    report = validate_current_integrity(ROOT)
    assert report["passed"], report["errors"]
    assert report["protected_files_checked"] == 66


def test_checkout_lf_or_crlf_does_not_affect_git_blob_verification(tmp_path):
    repo, manifest, baseline = make_repo(tmp_path)
    (repo / "protected.txt").write_bytes(b"alpha\r\nbeta\r\n")
    report = validate_current_integrity(repo, baseline, manifest)
    assert report["passed"], report["errors"]


def test_substantive_committed_text_mutation_fails(tmp_path):
    repo, manifest, _ = make_repo(tmp_path)
    (repo / "protected.txt").write_bytes(b"alpha\nchanged\n")
    changed = commit_all(repo, "change text")
    report = validate_current_integrity(repo, changed, manifest)
    assert not report["passed"]
    assert any("protected.txt" in error for error in report["errors"])


def test_changed_binary_git_blob_fails(tmp_path):
    repo, manifest, _ = make_repo(tmp_path)
    (repo / "protected.bin").write_bytes(b"\x00\xff\x11\r\n")
    changed = commit_all(repo, "change binary")
    report = validate_current_integrity(repo, changed, manifest)
    assert not report["passed"]
    assert any("protected.bin" in error for error in report["errors"])


def test_historical_audit_explicitly_reports_two_unresolved_entries():
    report = audit_historical_manifest(ROOT)
    assert not report["passed"]
    assert report["reproducible_from_git_or_eol"] == 64
    assert report["unresolved_historical_entries"] == 2
    assert report["unresolved_historical_paths"] == UNRESOLVED


def test_historical_manifest_git_blob_is_byte_for_byte_unchanged():
    _, baseline = read_git_blob(ROOT, BASELINE, HISTORICAL_MANIFEST)
    _, current = read_git_blob(ROOT, "HEAD", HISTORICAL_MANIFEST)
    assert current == baseline
