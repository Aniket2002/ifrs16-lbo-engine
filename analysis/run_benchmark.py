import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from lbo.analytic_bounds import AnalyticBoundsModel
from lbo.full_simulation import FullSimulationAssumptions, FullSimulationModel
from lbo.lbo_model_analytic import AnalyticAssumptions, AnalyticLBOModel


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "benchmark"
DATA_DIR = ROOT / "data" / "synthetic"


def _default_operators_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "operator_id": "SYN_HOTEL_001",
                "operator_name": "Hotel Archetype A",
                "revenue_0": 1200,
                "ebitda_0": 260,
                "revenue_growth_mean": 0.03,
                "revenue_growth_std": 0.04,
                "ebitda_margin_mean": 0.22,
                "financial_debt_0": 520,
                "lease_liability_0": 390,
                "cash_0": 45,
                "lambda_lease": 3.0,
                "cash_sweep": 0.55,
                "lease_principal_rate": 0.11,
                "lease_additions_rate": 0.01,
                "seed": 42,
            },
            {
                "operator_id": "SYN_HOTEL_002",
                "operator_name": "Hotel Archetype B",
                "revenue_0": 980,
                "ebitda_0": 205,
                "revenue_growth_mean": 0.025,
                "revenue_growth_std": 0.05,
                "ebitda_margin_mean": 0.21,
                "financial_debt_0": 470,
                "lease_liability_0": 360,
                "cash_0": 35,
                "lambda_lease": 3.2,
                "cash_sweep": 0.50,
                "lease_principal_rate": 0.12,
                "lease_additions_rate": 0.012,
                "seed": 42,
            },
            {
                "operator_id": "SYN_HOTEL_003",
                "operator_name": "Hotel Archetype C",
                "revenue_0": 1450,
                "ebitda_0": 330,
                "revenue_growth_mean": 0.035,
                "revenue_growth_std": 0.03,
                "ebitda_margin_mean": 0.23,
                "financial_debt_0": 610,
                "lease_liability_0": 470,
                "cash_0": 55,
                "lambda_lease": 2.9,
                "cash_sweep": 0.58,
                "lease_principal_rate": 0.10,
                "lease_additions_rate": 0.009,
                "seed": 42,
            },
            {
                "operator_id": "SYN_HOTEL_004",
                "operator_name": "Hotel Archetype D",
                "revenue_0": 760,
                "ebitda_0": 150,
                "revenue_growth_mean": 0.02,
                "revenue_growth_std": 0.05,
                "ebitda_margin_mean": 0.20,
                "financial_debt_0": 390,
                "lease_liability_0": 300,
                "cash_0": 30,
                "lambda_lease": 3.4,
                "cash_sweep": 0.48,
                "lease_principal_rate": 0.13,
                "lease_additions_rate": 0.013,
                "seed": 42,
            },
            {
                "operator_id": "SYN_HOTEL_005",
                "operator_name": "Hotel Archetype E",
                "revenue_0": 1680,
                "ebitda_0": 390,
                "revenue_growth_mean": 0.04,
                "revenue_growth_std": 0.035,
                "ebitda_margin_mean": 0.24,
                "financial_debt_0": 650,
                "lease_liability_0": 500,
                "cash_0": 60,
                "lambda_lease": 2.8,
                "cash_sweep": 0.60,
                "lease_principal_rate": 0.10,
                "lease_additions_rate": 0.008,
                "seed": 42,
            },
        ]
    )


def _default_scenario_params_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parameter_name": "revenue_growth",
                "definition": "Annual revenue growth used in generated scenarios",
                "unit": "ratio",
                "parameter_source": "synthetic_design",
                "reported_or_simulated": "simulated",
                "permitted_range": "-0.10 to 0.20",
                "seed": 42,
            },
            {
                "parameter_name": "ebitda_margin",
                "definition": "EBITDA margin for scenario simulation",
                "unit": "ratio",
                "parameter_source": "synthetic_design",
                "reported_or_simulated": "simulated",
                "permitted_range": "0.12 to 0.40",
                "seed": 42,
            },
            {
                "parameter_name": "cash_sweep",
                "definition": "Fraction of positive free cash flow used for debt paydown",
                "unit": "ratio",
                "parameter_source": "model_assumption",
                "reported_or_simulated": "simulated",
                "permitted_range": "0.30 to 0.80",
                "seed": 42,
            },
            {
                "parameter_name": "lease_principal_rate",
                "definition": "Fraction of opening lease liability paid as principal each year",
                "unit": "ratio",
                "parameter_source": "model_assumption",
                "reported_or_simulated": "simulated",
                "permitted_range": "0.05 to 0.20",
                "seed": 42,
            },
            {
                "parameter_name": "lease_additions_rate",
                "definition": "New lease additions as percent of revenue",
                "unit": "ratio",
                "parameter_source": "model_assumption",
                "reported_or_simulated": "simulated",
                "permitted_range": "0.00 to 0.05",
                "seed": 42,
            },
        ]
    )


def ensure_synthetic_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    operators_path = DATA_DIR / "operators.csv"
    params_path = DATA_DIR / "scenario_parameters.csv"
    checksums_path = DATA_DIR / "checksums.json"

    if not operators_path.exists():
        _default_operators_df().to_csv(operators_path, index=False)
    if not params_path.exists():
        _default_scenario_params_df().to_csv(params_path, index=False)

    checksums = {
        "operators_csv_sha256": _file_sha256(operators_path),
        "scenario_parameters_csv_sha256": _file_sha256(params_path),
    }
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode("utf-8").strip()
        )
    except Exception:
        return "unknown"


def _bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, seed: int, n_boot: int = 500
) -> List[float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return [float("nan"), float("nan")]
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class BenchmarkScenario:
    simulation: FullSimulationAssumptions
    analytic: AnalyticAssumptions
    label: str


def _build_scenario_pool(operators: pd.DataFrame) -> List[BenchmarkScenario]:
    scenarios: List[BenchmarkScenario] = []

    for _, row in operators.iterrows():
        base_sim = FullSimulationAssumptions(
            years=5,
            revenue_0=float(row["revenue_0"]),
            revenue_growth=float(row["revenue_growth_mean"]),
            ebitda_margin=float(row["ebitda_margin_mean"]),
            debt_opening=float(row["financial_debt_0"]),
            lease_opening=float(row["lease_liability_0"]),
            initial_cash=float(row["cash_0"]),
            cash_sweep=float(row["cash_sweep"]),
        )
        base_analytic = AnalyticAssumptions(
            ebitda_0=float(row["ebitda_0"]),
            growth_rate=float(row["revenue_growth_mean"]),
            financial_debt_0=float(row["financial_debt_0"]),
            lease_liability_0=float(row["lease_liability_0"]),
            lambda_lease=float(row["lambda_lease"]),
            cash_sweep=float(row["cash_sweep"]),
            n_years=5,
            lease_treatment="run_off",
            lease_principal_rate=float(row["lease_principal_rate"]),
            lease_additions_rate=float(row["lease_additions_rate"]),
        )

        downside_sim = FullSimulationAssumptions(
            years=5,
            revenue_0=float(row["revenue_0"]),
            revenue_growth=max(-0.08, float(row["revenue_growth_mean"]) - 0.05),
            ebitda_margin=max(0.10, float(row["ebitda_margin_mean"]) - 0.06),
            debt_opening=float(row["financial_debt_0"]) * 1.10,
            lease_opening=float(row["lease_liability_0"]) * 1.08,
            initial_cash=float(row["cash_0"]) * 0.75,
            cash_sweep=min(0.8, float(row["cash_sweep"]) + 0.05),
        )
        downside_analytic = AnalyticAssumptions(
            ebitda_0=float(row["ebitda_0"]),
            growth_rate=max(-0.08, float(row["revenue_growth_mean"]) - 0.05),
            alpha=0.72,
            kappa=0.08,
            financial_debt_0=float(row["financial_debt_0"]) * 1.10,
            lease_liability_0=float(row["lease_liability_0"]) * 1.08,
            lambda_lease=float(row["lambda_lease"]),
            cash_sweep=min(0.8, float(row["cash_sweep"]) + 0.05),
            n_years=5,
            lease_treatment="run_off",
            lease_principal_rate=float(row["lease_principal_rate"]),
            lease_additions_rate=float(row["lease_additions_rate"]),
        )

        distressed_sim = FullSimulationAssumptions(
            years=5,
            revenue_0=float(row["revenue_0"]) * 0.9,
            revenue_growth=max(-0.10, float(row["revenue_growth_mean"]) - 0.08),
            ebitda_margin=max(0.08, float(row["ebitda_margin_mean"]) - 0.10),
            debt_opening=float(row["financial_debt_0"]) * 1.25,
            lease_opening=float(row["lease_liability_0"]) * 1.15,
            initial_cash=float(row["cash_0"]) * 0.50,
            cash_sweep=min(0.85, float(row["cash_sweep"]) + 0.10),
            revolver_limit=100.0,
        )
        distressed_analytic = AnalyticAssumptions(
            ebitda_0=float(row["ebitda_0"]) * 0.9,
            growth_rate=max(-0.10, float(row["revenue_growth_mean"]) - 0.08),
            alpha=0.68,
            kappa=0.10,
            financial_debt_0=float(row["financial_debt_0"]) * 1.25,
            lease_liability_0=float(row["lease_liability_0"]) * 1.15,
            lambda_lease=float(row["lambda_lease"]) + 0.5,
            cash_sweep=min(0.85, float(row["cash_sweep"]) + 0.10),
            n_years=5,
            lease_treatment="run_off",
            lease_principal_rate=float(row["lease_principal_rate"]),
            lease_additions_rate=float(row["lease_additions_rate"]),
        )

        scenarios.extend(
            [
                BenchmarkScenario(base_sim, base_analytic, f"{row['operator_id']}:base"),
                BenchmarkScenario(
                    downside_sim, downside_analytic, f"{row['operator_id']}:downside"
                ),
                BenchmarkScenario(
                    distressed_sim, distressed_analytic, f"{row['operator_id']}:distressed"
                ),
            ]
        )

    return scenarios


def run_benchmark(seed: int, smoke_test: bool = False) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_synthetic_data()

    operators = pd.read_csv(DATA_DIR / "operators.csv")
    rng = np.random.default_rng(seed)

    scenario_pool = _build_scenario_pool(operators)
    n_scenarios = 20 if smoke_test else 200
    scenario_inputs = [
        scenario_pool[int(i)] for i in rng.choice(len(scenario_pool), n_scenarios, replace=True)
    ]

    # Warm up both model paths before timing.
    FullSimulationModel(scenario_inputs[0].simulation).simulate()
    AnalyticLBOModel(scenario_inputs[0].analytic).solve_paths()

    sim_times: List[float] = []
    analytic_times: List[float] = []
    n_repeats = 5
    for _ in range(n_repeats):
        start = time.perf_counter()
        for scenario in scenario_inputs:
            FullSimulationModel(scenario.simulation).simulate()
        sim_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        for scenario in scenario_inputs:
            AnalyticLBOModel(scenario.analytic).solve_paths()
        analytic_times.append(time.perf_counter() - start)

    y_true: List[int] = []
    y_score: List[float] = []
    leverage_mae: List[float] = []
    leverage_rmse: List[float] = []
    icr_mae: List[float] = []
    icr_rmse: List[float] = []
    headroom_mae: List[float] = []
    headroom_rmse: List[float] = []
    failed = 0
    failure_details: List[Dict[str, Any]] = []

    for scenario in scenario_inputs:
        try:
            sim_rows = FullSimulationModel(scenario.simulation).simulate()
            analytic = AnalyticLBOModel(scenario.analytic).solve_paths()
            sim_df = pd.DataFrame(sim_rows)

            sim_lev = (
                sim_df["debt_balance"]
                + sim_df["revolver_balance"]
                + sim_df["lease_liability"]
                - sim_df["ending_cash"]
            ) / sim_df["ebitda"]
            sim_icr = sim_df["ebitda"] / np.maximum(
                1e-6, sim_df["cash_interest"] + sim_df["lease_interest"]
            )

            simulated_max_leverage = float(np.nanmax(sim_lev.to_numpy()))
            simulated_min_icr = float(np.nanmin(sim_icr.to_numpy()))
            analytic_max_leverage = float(np.nanmax(analytic.leverage_ratio[1:]))
            analytic_min_icr = float(np.nanmin(analytic.icr_ratio[1:]))

            headroom_s = float(min(6.0 - simulated_max_leverage, simulated_min_icr - 1.8))
            headroom_a = float(min(6.0 - analytic_max_leverage, analytic_min_icr - 1.8))

            true_breach = int(simulated_max_leverage > 6.0 or simulated_min_icr < 1.8)
            pred_breach_prob = float(1.0 / (1.0 + np.exp(2.0 * headroom_a)))

            y_true.append(true_breach)
            y_score.append(pred_breach_prob)

            leverage_diff = np.abs(analytic.leverage_ratio[1:] - sim_lev.to_numpy())
            icr_diff = np.abs(analytic.icr_ratio[1:] - sim_icr.to_numpy())

            leverage_mae.append(float(np.mean(leverage_diff)))
            leverage_rmse.append(float(np.sqrt(np.mean(leverage_diff**2))))
            icr_mae.append(float(np.mean(icr_diff)))
            icr_rmse.append(float(np.sqrt(np.mean(icr_diff**2))))
            headroom_mae.append(abs(headroom_a - headroom_s))
            headroom_rmse.append((headroom_a - headroom_s) ** 2)
        except Exception as exc:
            failed += 1
            failure_details.append({"label": scenario.label, "error": str(exc)})

    if failed > 0:
        raise RuntimeError(f"Benchmark failed for {failed} scenario(s): {failure_details[:3]}")

    y_true_a = np.array(y_true)
    y_score_a = np.array(y_score)

    if len(np.unique(y_true_a)) < 2:
        raise RuntimeError(
            "Benchmark scenarios collapsed to a single class; widen scenario coverage."
        )

    auc = float(roc_auc_score(y_true_a, y_score_a))
    auc_ci = _bootstrap_auc_ci(y_true_a, y_score_a, seed)
    brier = float(brier_score_loss(y_true_a, y_score_a))

    fn = int(np.sum((y_true_a == 1) & (y_score_a < 0.5)))
    fp = int(np.sum((y_true_a == 0) & (y_score_a >= 0.5)))
    tp = int(np.sum((y_true_a == 1) & (y_score_a >= 0.5)))
    tn = int(np.sum((y_true_a == 0) & (y_score_a < 0.5)))
    actual_positive = max(1, tp + fn)
    actual_negative = max(1, tn + fp)

    leverage_mae_mean = float(np.mean(leverage_mae))
    leverage_rmse_mean = float(np.mean(leverage_rmse))
    icr_mae_mean = float(np.mean(icr_mae))
    icr_rmse_mean = float(np.mean(icr_rmse))
    headroom_mae_mean = float(np.mean(headroom_mae))
    headroom_rmse_mean = float(np.sqrt(np.mean(headroom_rmse)))

    sim_median = float(np.median(sim_times))
    sim_iqr = float(np.percentile(sim_times, 75) - np.percentile(sim_times, 25))
    analytic_median = float(np.median(analytic_times))
    analytic_iqr = float(np.percentile(analytic_times, 75) - np.percentile(analytic_times, 25))
    speedup = sim_median / max(1e-9, analytic_median)

    frac_pos, mean_pred = calibration_curve(y_true_a, y_score_a, n_bins=8, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted breach probability")
    ax.set_ylabel("Observed breach frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=200)
    plt.close(fig)

    bounds = AnalyticBoundsModel().calculate_assumption_bounds()

    report = {
        "seed": seed,
        "scenario_count": n_scenarios,
        "failed_scenario_count": failed,
        "auc": auc,
        "auc_ci_95": auc_ci,
        "brier_score": brier,
        "false_negative_rate": float(fn / actual_positive),
        "false_positive_rate": float(fp / actual_negative),
        "leverage_mae": leverage_mae_mean,
        "leverage_rmse": leverage_rmse_mean,
        "icr_mae": icr_mae_mean,
        "icr_rmse": icr_rmse_mean,
        "minimum_headroom_mae": headroom_mae_mean,
        "minimum_headroom_rmse": headroom_rmse_mean,
        "speed_benchmark": {
            "simulation_seconds_median": sim_median,
            "simulation_seconds_iqr": sim_iqr,
            "analytic_seconds_median": analytic_median,
            "analytic_seconds_iqr": analytic_iqr,
            "speedup_x": speedup,
        },
        "analytic_bounds": {
            "icr_error_bound": bounds.icr_error_bound,
            "leverage_error_bound": bounds.leverage_error_bound,
        },
        "git_sha": _git_sha(),
        "data_checksums": {
            "operators_csv": _file_sha256(DATA_DIR / "operators.csv"),
            "scenario_parameters_csv": _file_sha256(DATA_DIR / "scenario_parameters.csv"),
        },
        "scenario_failures": failure_details,
    }

    (OUTPUT_DIR / "benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(seed=args.seed, smoke_test=args.smoke_test)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
