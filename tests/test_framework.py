import subprocess
import sys
from pathlib import Path

from analysis.run_benchmark import run_benchmark


def test_benchmark_smoke(tmp_path):
    report = run_benchmark(seed=42, smoke_test=True)
    assert report["scenario_count"] == 20
    assert "git_sha" in report
    assert "auc_ci_95" in report
    assert "false_negative_rate" in report
    assert "false_positive_rate" in report
    assert "leverage_mae" in report
    assert report["failed_scenario_count"] == 0


def test_case_study_command_runs():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "analysis/scripts/case_study_accor.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ACCOR SA IFRS-16 CASE STUDY" in completed.stdout
