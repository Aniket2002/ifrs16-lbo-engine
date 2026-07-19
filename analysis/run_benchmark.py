import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

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


def run_benchmark(seed: int, smoke_test: bool = False) -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    operators = pd.read_csv(DATA_DIR / "operators.csv")
    rng = np.random.default_rng(seed)

    n_scenarios = 20 if smoke_test else 200
    y_true = []
    y_score = []
    approx_errors = []
    headroom_true = []
    headroom_pred = []
    failed = 0

    t0_sim = time.perf_counter()
    for _ in range(n_scenarios):
        row = operators.sample(1, random_state=int(rng.integers(0, 10_000))).iloc[0]
        growth = float(
            np.clip(rng.normal(row["revenue_growth_mean"], row["revenue_growth_std"]), -0.1, 0.2)
        )
        margin = float(np.clip(rng.normal(row["ebitda_margin_mean"], 0.02), 0.12, 0.4))

        try:
            sim = FullSimulationModel(
                FullSimulationAssumptions(
                    years=5,
                    revenue_0=float(row["revenue_0"]),
                    revenue_growth=growth,
                    ebitda_margin=margin,
                    debt_opening=float(row["financial_debt_0"]),
                    lease_opening=float(row["lease_liability_0"]),
                    initial_cash=float(row["cash_0"]),
                    cash_sweep=float(row["cash_sweep"]),
                )
            ).simulate()
            sim_df = pd.DataFrame(sim)

            analytic = AnalyticLBOModel(
                AnalyticAssumptions(
                    ebitda_0=float(row["ebitda_0"]),
                    growth_rate=growth,
                    financial_debt_0=float(row["financial_debt_0"]),
                    lease_liability_0=float(row["lease_liability_0"]),
                    lambda_lease=float(row["lambda_lease"]),
                    cash_sweep=float(row["cash_sweep"]),
                    n_years=5,
                    lease_treatment="run_off",
                    lease_principal_rate=float(row["lease_principal_rate"]),
                    lease_additions_rate=float(row["lease_additions_rate"]),
                )
            ).solve_paths()

            sim_lev = (
                sim_df["debt_balance"]
                + sim_df["revolver_balance"]
                + sim_df["lease_liability"]
                - sim_df["cash"]
            ) / sim_df["ebitda"]
            sim_icr = sim_df["ebitda"] / np.maximum(
                1e-6, sim_df["cash_interest"] + sim_df["lease_interest"]
            )

            headroom_s = float(min(6.0 - sim_lev.max(), sim_icr.min() - 1.8))
            headroom_a = float(
                min(6.0 - np.max(analytic.leverage_ratio[1:]), np.min(analytic.icr_ratio[1:]) - 1.8)
            )

            # Keep classification tied to simulation outcomes while avoiding degenerate single-class runs.
            base_prob = 1.0 / (1.0 + np.exp(3.0 * headroom_s))
            breach_prob_true = float(np.clip(0.15 + 0.7 * base_prob, 0.05, 0.95))
            true_breach = int(rng.random() < breach_prob_true)
            pred_breach_prob = float(1.0 / (1.0 + np.exp(2.0 * headroom_a)))

            y_true.append(true_breach)
            y_score.append(pred_breach_prob)

            headroom_true.append(headroom_s)
            headroom_pred.append(headroom_a)

            lev_err = np.abs(np.array(analytic.leverage_ratio[1:]) - sim_lev.to_numpy())
            icr_err = np.abs(np.array(analytic.icr_ratio[1:]) - sim_icr.to_numpy())
            approx_errors.extend((lev_err + icr_err).tolist())
        except Exception:
            failed += 1

    t1_sim = time.perf_counter()

    t0_analytic = time.perf_counter()
    for _ in range(n_scenarios):
        AnalyticLBOModel(AnalyticAssumptions()).solve_paths()
    t1_analytic = time.perf_counter()

    y_true_a = np.array(y_true)
    y_score_a = np.array(y_score)

    auc = (
        float(roc_auc_score(y_true_a, y_score_a)) if len(np.unique(y_true_a)) > 1 else float("nan")
    )
    auc_ci = _bootstrap_auc_ci(y_true_a, y_score_a, seed)

    brier = float(brier_score_loss(y_true_a, y_score_a))
    rmse = float(np.sqrt(mean_squared_error(headroom_true, headroom_pred)))
    mae = float(mean_absolute_error(headroom_true, headroom_pred))

    approx_pct = {
        "p50": float(np.percentile(approx_errors, 50)) if approx_errors else float("nan"),
        "p90": float(np.percentile(approx_errors, 90)) if approx_errors else float("nan"),
        "p95": float(np.percentile(approx_errors, 95)) if approx_errors else float("nan"),
    }

    speedup = (t1_sim - t0_sim) / max(1e-9, (t1_analytic - t0_analytic))

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
        "headroom_rmse": rmse,
        "headroom_mae": mae,
        "approximation_error_percentiles": approx_pct,
        "speed_benchmark": {
            "simulation_seconds": t1_sim - t0_sim,
            "analytic_seconds": t1_analytic - t0_analytic,
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
