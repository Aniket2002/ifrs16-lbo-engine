from analysis import run_benchmark as benchmark
from analysis.run_benchmark import run_benchmark


def test_end_to_end_benchmark_smoke():
    out = run_benchmark(seed=42, smoke_test=True)
    assert out["failed_scenario_count"] >= 0
    assert out["speed_benchmark"]["speedup_x"] > 0


def test_benchmark_counts_payment_default_without_other_failure(monkeypatch, tmp_path):
    calls = 0

    def simulate(self):
        nonlocal calls
        calls += 1
        # Isolate label aggregation: safe ratios and sufficient cash, but every
        # other scenario reports a contractual payment default.
        return [
            {
                "ebitda": 100,
                "debt_balance": 100,
                "revolver_balance": 0,
                "lease_liability": 0,
                "ending_cash": 25,
                "cash_interest": 5,
                "lease_interest": 0,
                "insolvency_flag": False,
                "payment_default_flag": calls % 2 == 0,
            }
        ] * 5

    monkeypatch.setattr(benchmark.FullSimulationModel, "simulate", simulate)
    monkeypatch.setattr(benchmark, "OUTPUT_DIR", tmp_path)
    report = run_benchmark(seed=42, smoke_test=True)
    assert report["failure_type_counts"] == {"payment_default": 10, "none": 10}
    assert sum(r["true_failure"] for r in report["scenario_records"]) == 10
