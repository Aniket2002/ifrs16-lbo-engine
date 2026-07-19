from analysis.run_benchmark import run_benchmark


def test_benchmark_smoke(tmp_path):
    report = run_benchmark(seed=42, smoke_test=True)
    assert report["scenario_count"] == 20
    assert "git_sha" in report
    assert "auc_ci_95" in report
