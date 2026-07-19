from analysis.run_benchmark import run_benchmark


def test_end_to_end_benchmark_smoke():
    out = run_benchmark(seed=42, smoke_test=True)
    assert out["failed_scenario_count"] >= 0
    assert out["speed_benchmark"]["speedup_x"] > 0
