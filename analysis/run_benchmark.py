import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
) -> list[float]:
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
    operator_id: str
    scenario_id: str
    scenario_type: str
    simulation: FullSimulationAssumptions
    analytic: AnalyticAssumptions
    label: str


def _draw_scenario(row: pd.Series, scenario_id: int, rng: np.random.Generator) -> BenchmarkScenario:
    scenario_type = str(rng.choice(["base", "downside", "distressed"], p=[0.5, 0.3, 0.2]))

    revenue_growth = float(
        np.clip(rng.normal(row["revenue_growth_mean"], row["revenue_growth_std"]), -0.12, 0.18)
    )
    ebitda_margin = float(np.clip(rng.normal(row["ebitda_margin_mean"], 0.02), 0.10, 0.35))
    cash_sweep = float(np.clip(rng.normal(row["cash_sweep"], 0.04), 0.30, 0.85))
    lease_principal_rate = float(np.clip(rng.normal(row["lease_principal_rate"], 0.01), 0.05, 0.20))
    lease_additions_rate = float(np.clip(rng.normal(row["lease_additions_rate"], 0.003), 0.0, 0.05))
    ebitda_0 = float(row["ebitda_0"])
    revenue_0 = float(row["revenue_0"])
    financial_debt_0 = float(row["financial_debt_0"])
    lease_liability_0 = float(row["lease_liability_0"])
    cash_0 = float(row["cash_0"])

    if scenario_type == "downside":
        revenue_growth -= 0.04
        ebitda_margin -= 0.03
        financial_debt_0 *= 1.05
        lease_liability_0 *= 1.04
        cash_0 *= 0.85
    elif scenario_type == "distressed":
        revenue_growth -= 0.08
        ebitda_margin -= 0.06
        financial_debt_0 *= 1.20
        lease_liability_0 *= 1.10
        cash_0 *= 0.60

    simulation = FullSimulationAssumptions(
        years=5,
        entry_enterprise_value=revenue_0 * 7.5,
        transaction_fees_pct=0.03,
        revenue_0=revenue_0,
        revenue_growth=max(-0.12, revenue_growth),
        ebitda_margin=max(0.08, ebitda_margin),
        debt_opening=financial_debt_0,
        lease_opening=lease_liability_0,
        initial_cash=cash_0,
        cash_sweep=cash_sweep,
        revolver_limit=100.0 if scenario_type == "distressed" else 200.0,
        lease_principal_pct_opening=lease_principal_rate,
        lease_additions_pct_revenue=lease_additions_rate,
    )

    analytic = AnalyticAssumptions(
        ebitda_0=ebitda_0,
        growth_rate=max(-0.12, revenue_growth),
        alpha=0.72 if scenario_type == "base" else 0.68,
        kappa=0.08 if scenario_type != "distressed" else 0.10,
        financial_debt_0=financial_debt_0,
        lease_liability_0=lease_liability_0,
        lambda_lease=float(row["lambda_lease"]),
        cash_sweep=cash_sweep,
        n_years=5,
        lease_treatment="run_off",
        lease_principal_rate=lease_principal_rate,
        lease_additions_rate=lease_additions_rate,
    )

    operator_id = str(row["operator_id"])
    scenario_label = f"{operator_id}:{scenario_id:04d}:{scenario_type}"
    return BenchmarkScenario(
        operator_id, f"{scenario_id:04d}", scenario_type, simulation, analytic, scenario_label
    )


def run_benchmark(seed: int, smoke_test: bool = False) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_synthetic_data()

    operators = pd.read_csv(DATA_DIR / "operators.csv")
    rng = np.random.default_rng(seed)

    n_scenarios = 20 if smoke_test else 200
    operator_indices = rng.integers(0, len(operators), size=n_scenarios)
    scenario_inputs = [
        _draw_scenario(operators.iloc[int(operator_index)], scenario_id=index, rng=rng)
        for index, operator_index in enumerate(operator_indices)
    ]

    # Warm up both model paths before timing.
    FullSimulationModel(scenario_inputs[0].simulation).simulate()
    AnalyticLBOModel(scenario_inputs[0].analytic).solve_paths()

    sim_times: list[float] = []
    analytic_times: list[float] = []
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

    y_true: list[int] = []
    y_score: list[float] = []
    leverage_mae: list[float] = []
    leverage_rmse: list[float] = []
    icr_mae: list[float] = []
    icr_rmse: list[float] = []
    headroom_mae: list[float] = []
    headroom_rmse: list[float] = []
    failed = 0
    failure_details: list[dict[str, Any]] = []
    failure_type_counts: Counter[str] = Counter()
    scenario_records: list[dict[str, Any]] = []

    for scenario in scenario_inputs:
        try:
            sim_rows = FullSimulationModel(scenario.simulation).simulate()
            analytic = AnalyticLBOModel(scenario.analytic).solve_paths()
            sim_df = pd.DataFrame(sim_rows)

            negative_ebitda = bool((sim_df["ebitda"] <= 0).any())
            insolvency = bool(sim_df["insolvency_flag"].any())
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

            breach = bool(simulated_max_leverage > 6.0 or simulated_min_icr < 1.8)
            true_failure = int(negative_ebitda or insolvency or breach)
            analytic_risk_score = float(1.0 / (1.0 + np.exp(2.0 * headroom_a)))

            if negative_ebitda:
                failure_type = "negative_ebitda"
            elif insolvency and breach:
                failure_type = "insolvency_and_covenant_breach"
            elif insolvency:
                failure_type = "insolvency"
            elif breach:
                failure_type = "covenant_breach"
            else:
                failure_type = "none"

            y_true.append(true_failure)
            y_score.append(analytic_risk_score)
            failure_type_counts[failure_type] += 1
            scenario_records.append(
                {
                    "operator_id": scenario.operator_id,
                    "scenario_id": scenario.scenario_id,
                    "scenario_type": scenario.scenario_type,
                    "analytic_risk_score": analytic_risk_score,
                    "true_failure": true_failure,
                    "failure_type": failure_type,
                    "simulated_max_leverage": simulated_max_leverage,
                    "simulated_min_icr": simulated_min_icr,
                }
            )

            leverage_diff = np.abs(analytic.leverage_ratio[1:] - sim_lev.to_numpy())
            icr_diff = np.abs(analytic.icr_ratio[1:] - sim_icr.to_numpy())

            leverage_mae.append(
                float(mean_absolute_error(sim_lev.to_numpy(), analytic.leverage_ratio[1:]))
            )
            leverage_rmse.append(
                float(np.sqrt(mean_squared_error(sim_lev.to_numpy(), analytic.leverage_ratio[1:])))
            )
            icr_mae.append(float(mean_absolute_error(sim_icr.to_numpy(), analytic.icr_ratio[1:])))
            icr_rmse.append(
                float(np.sqrt(mean_squared_error(sim_icr.to_numpy(), analytic.icr_ratio[1:])))
            )
            headroom_mae.append(abs(headroom_a - headroom_s))
            headroom_rmse.append((headroom_a - headroom_s) ** 2)
        except Exception as exc:
            failed += 1
            failure_details.append(
                {
                    "operator_id": scenario.operator_id,
                    "scenario_id": scenario.scenario_id,
                    "scenario_type": scenario.scenario_type,
                    "label": scenario.label,
                    "error": str(exc),
                    "failure_type": "exception",
                }
            )

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
    ax.plot(mean_pred, frac_pos, marker="o", label="Analytic risk score")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted risk score")
    ax.set_ylabel("Observed breach frequency")
    ax.set_title("Calibration Curve for Analytic Risk Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_curve.png", dpi=200)
    plt.close(fig)

    envelopes = AnalyticBoundsModel().calculate_diagnostic_envelopes()

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
        "failure_type_counts": dict(failure_type_counts),
        "scenario_records": scenario_records,
        "speed_benchmark": {
            "simulation_seconds_median": sim_median,
            "simulation_seconds_iqr": sim_iqr,
            "analytic_seconds_median": analytic_median,
            "analytic_seconds_iqr": analytic_iqr,
            "speedup_x": speedup,
        },
        "diagnostic_envelopes": {
            "icr_error_bound": envelopes.icr_error_bound,
            "leverage_error_bound": envelopes.leverage_error_bound,
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
