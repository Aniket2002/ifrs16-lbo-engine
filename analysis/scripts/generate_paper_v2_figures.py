"""Regenerate the SSRN v2 figures, numerical tables and provenance from current code.

Run from the repository root: python -m analysis.scripts.generate_paper_v2_figures
Use --run-benchmark to archive a fresh full benchmark first. No model code is changed.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from analysis.run_benchmark import _draw_scenario
from analysis.scripts.case_study_accor import run_accor_case_study
from lbo.full_simulation import FullSimulationModel
from lbo.lbo_model_analytic import AnalyticLBOModel

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper_v2"
FIGURES = ROOT / "paper/figures/v2"
GENERATED = ROOT / "paper/generated"
BLUE, RED, GOLD = "#245779", "#b44c43", "#b78c36"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, name, report):
    titles = {
        "F12_theoretical_guarantees": "Analytic versus simulation diagnostics",
        "F14_method_comparison": "Ranking performance and model discrepancies",
        "F13_benchmark_overview": "Synthetic scenario composition",
        "F16_breach_composition": "Financial-failure composition",
        "F15_failure_modes": "Ranking versus failure detection",
        "accor_case_study": "Accor: stipulated accounting-convention illustration",
    }
    fig.suptitle(titles[name], fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        f"Source {report['git_sha'][:12]} | seed {report['seed']} | synthetic benchmark"
        if name != "accor_case_study"
        else f"Source {report['git_sha'][:12]} | repository Accor inputs | stipulated conventions",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=180, metadata={"Creator": "IFRS16 paper v2"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-benchmark", action="store_true")
    args = parser.parse_args()
    for folder in (RESULTS, FIGURES, GENERATED):
        folder.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS / "benchmark_seed42.json"
    if args.run_benchmark:
        subprocess.run(
            [sys.executable, "-m", "analysis.run_benchmark", "--seed", "42"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        report_path.write_bytes((ROOT / "output/benchmark/benchmark_report.json").read_bytes())
    report = json.loads(report_path.read_text())
    if report["scenario_count"] != 200 or report["seed"] != 42:
        raise ValueError("Paper protocol requires the full 200-scenario seed-42 benchmark")
    operators_path = ROOT / "data/synthetic/operators.csv"
    if sha256(operators_path) != report["data_checksums"]["operators_csv"]:
        raise ValueError("Archived benchmark and operator data differ")
    operators = pd.read_csv(operators_path)
    rng = np.random.default_rng(report["seed"])
    indices = rng.integers(0, len(operators), size=report["scenario_count"])
    scenarios = [
        _draw_scenario(operators.iloc[int(index)], i, rng) for i, index in enumerate(indices)
    ]
    paths, assumptions, headroom_differences = [], [], []
    for scenario, record in zip(scenarios, report["scenario_records"], strict=True):
        sim = pd.DataFrame(FullSimulationModel(scenario.simulation).simulate())
        analytic = AnalyticLBOModel(scenario.analytic).solve_paths()
        lev = (
            sim.debt_balance + sim.revolver_balance + sim.lease_liability - sim.ending_cash
        ) / sim.ebitda
        icr = sim.ebitda / np.maximum(1e-6, sim.cash_interest + sim.lease_interest)
        ha = min(6 - max(analytic.leverage_ratio[1:]), min(analytic.icr_ratio[1:]) - 1.8)
        score = 1 / (1 + np.exp(2 * ha))
        headroom_differences.append(ha - min(6 - max(lev), min(icr) - 1.8))
        checks = [
            (score, record["analytic_risk_score"]),
            (max(lev), record["simulated_max_leverage"]),
            (min(icr), record["simulated_min_icr"]),
        ]
        if not all(np.isclose(a, b, rtol=1e-12, atol=1e-12) for a, b in checks):
            raise ValueError(f"Archived benchmark differs from code: {scenario.label}")
        sim["scenario_id"] = scenario.scenario_id
        sim["scenario_type"] = scenario.scenario_type
        sim["analytic_leverage"] = analytic.leverage_ratio[1:]
        sim["analytic_icr"] = analytic.icr_ratio[1:]
        sim["simulated_leverage"] = lev
        sim["simulated_icr"] = icr
        sim["leverage_difference"] = analytic.leverage_ratio[1:] - lev
        sim["icr_difference"] = analytic.icr_ratio[1:] - icr
        paths.append(sim)
        assumptions.append(
            {
                "scenario_id": scenario.scenario_id,
                "operator_id": scenario.operator_id,
                "scenario_type": scenario.scenario_type,
                "simulation": asdict(scenario.simulation),
                "analytic": asdict(scenario.analytic),
            }
        )
    paths = pd.concat(paths, ignore_index=True)
    # Verify the plotted path archive reproduces the report's actual aggregation.
    for metric in ("leverage", "icr"):
        differences = paths.groupby("scenario_id")[f"{metric}_difference"]
        mae = differences.apply(lambda x: np.mean(np.abs(x))).mean()
        rmse = differences.apply(lambda x: np.sqrt(np.mean(np.square(x)))).mean()
        for suffix, value in [("mae", mae), ("rmse", rmse)]:
            if not np.isclose(value, report[f"{metric}_{suffix}"], rtol=1e-12):
                raise ValueError(f"Path archive does not reproduce {metric}_{suffix}")
    for suffix, value in [
        ("mae", np.mean(np.abs(headroom_differences))),
        ("rmse", np.sqrt(np.mean(np.square(headroom_differences)))),
    ]:
        if not np.isclose(value, report[f"minimum_headroom_{suffix}"], rtol=1e-12):
            raise ValueError(f"Path archive does not reproduce minimum_headroom_{suffix}")
    paths.to_csv(RESULTS / "scenario_paths.csv", index=False)
    (RESULTS / "scenario_inputs.json").write_text(json.dumps(assumptions, indent=2) + "\n")
    records = pd.DataFrame(report["scenario_records"])
    records.to_csv(RESULTS / "scenario_records.csv", index=False)
    accor = run_accor_case_study()
    plt.close("all")
    accor.to_csv(RESULTS / "accor_results.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, key, title in zip(axes, ["leverage_difference", "icr_difference"], ["Leverage", "ICR"]):
        ax.hist(paths[key], bins=30, color=BLUE, alpha=0.85)
        ax.axvline(0, color=RED, linestyle="--")
        ax.set(
            xlabel="Analytic minus simulation (ratio points)",
            ylabel="Scenario-years",
            title=f"{title}: structural and approximation differences",
        )
    save_figure(fig, "F12_theoretical_guarantees", report)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    fpr, tpr, _ = roc_curve(records.true_failure, records.analytic_risk_score)
    axes[0].plot(fpr, tpr, color=BLUE, label=f"Analytic ranking: AUC {report['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Chance reference")
    axes[0].set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Full simulation supplies labels",
    )
    axes[0].legend(loc="lower right", fontsize=8)
    x = np.arange(3)
    axes[1].bar(
        x - 0.18,
        [report[k] for k in ["leverage_mae", "icr_mae", "minimum_headroom_mae"]],
        0.36,
        color=BLUE,
        label="MAE",
    )
    axes[1].bar(
        x + 0.18,
        [report[k] for k in ["leverage_rmse", "icr_rmse", "minimum_headroom_rmse"]],
        0.36,
        color=GOLD,
        label="RMSE",
    )
    axes[1].set(
        xticks=x,
        xticklabels=["Leverage", "ICR", "Min. headroom"],
        ylabel="Ratio points",
        title="Cross-model discrepancies",
    )
    axes[1].legend()
    save_figure(fig, "F14_method_comparison", report)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    regime_order = ["base", "downside", "distressed"]
    counts = records.scenario_type.value_counts().reindex(regime_order, fill_value=0)
    axes[0].bar(counts.index, counts.values, color=[BLUE, GOLD, RED])
    for i, count in enumerate(counts):
        axes[0].text(i, count + 1, str(count), ha="center")
    axes[0].set(ylabel="Scenarios", title="Realized regime counts", ylim=(0, counts.max() * 1.15))
    operator_counts = records.operator_id.value_counts().sort_index()
    axes[1].bar(
        [x.rsplit("_", 1)[-1] for x in operator_counts.index], operator_counts.values, color=BLUE
    )
    axes[1].set(
        xlabel="Synthetic operator ID suffix", ylabel="Scenarios", title="Sampling with replacement"
    )
    save_figure(fig, "F13_benchmark_overview", report)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    failure_counts = [
        report["failure_type_counts"].get(k, 0)
        for k in ["none", "payment_default", "insolvency_and_covenant_breach"]
    ]
    axes[0].barh(
        ["No failure", "Payment default (priority)", "Other reserve + covenant"],
        failure_counts,
        color=[BLUE, RED, GOLD],
    )
    for i, count in enumerate(failure_counts):
        axes[0].text(count + 1, i, str(count), va="center")
    axes[0].set(xlabel="Scenarios", title="Mutually exclusive priority labels", xlim=(0, 205))
    flags = records[["payment_default", "insolvency", "covenant_breach"]].sum()
    axes[1].bar(
        ["Payment\ndefault", "Reserve\ndeficit", "Covenant\nbreach"], flags, color=[RED, GOLD, BLUE]
    )
    for i, count in enumerate(flags):
        axes[1].text(i, count + 0.3, str(count), ha="center")
    axes[1].set(
        ylabel="Scenarios (overlap allowed)",
        title="Separate failure conditions",
        ylim=(0, max(flags) * 1.2),
    )
    save_figure(fig, "F16_breach_composition", report)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for failed, color, label in [(0, BLUE, "Nonfailure"), (1, RED, "Financial failure")]:
        axes[0].hist(
            records.loc[records.true_failure == failed, "analytic_risk_score"],
            bins=np.linspace(0, 1, 21),
            alpha=0.65,
            color=color,
            label=label,
        )
    axes[0].axvline(0.5, color="black", linestyle="--", label="Decision threshold")
    axes[0].set(
        xlabel="Analytic ranking score (not probability)",
        ylabel="Scenarios",
        title="Ranking versus classification",
    )
    axes[0].legend(fontsize=8)
    cm = confusion_matrix(records.true_failure, records.analytic_risk_score >= 0.5, labels=[0, 1])
    axes[1].imshow(cm, cmap="Blues")
    for (i, j), count in np.ndenumerate(cm):
        axes[1].text(
            j,
            i,
            str(count),
            ha="center",
            va="center",
            fontsize=20,
            color="white" if count > cm.max() / 2 else "black",
        )
    axes[1].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Nonfailure", "Failure"],
        yticklabels=["Nonfailure", "Failure"],
        xlabel="Analytic classification",
        ylabel="Simulation outcome",
        title="Threshold 0.5 confusion matrix",
    )
    save_figure(fig, "F15_failure_modes", report)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, metric, threshold in zip(axes, ["icr", "leverage"], [4.0, 3.5]):
        for convention, color, label in [
            ("ifrs16", BLUE, "IFRS-16"),
            ("frozen_gaap", RED, "Stipulated frozen-GAAP"),
        ]:
            ax.plot(accor.year, accor[f"{metric}_{convention}"], "o-", color=color, label=label)
        ax.axhline(threshold, color=GOLD, linestyle="--", label="Hypothetical threshold")
        ax.axvspan(2019.8, 2020.2, color="gray", alpha=0.15)
        ax.set(
            xticks=accor.year,
            ylabel="Ratio (x)",
            title=metric.upper(),
            xlabel="2020: undefined ratios; failure in both views",
        )
        ax.legend(fontsize=7)
    save_figure(fig, "accor_case_study", report)

    metrics = [
        ("Leverage MAE", "leverage_mae"),
        ("Mean scenario leverage RMSE", "leverage_rmse"),
        ("ICR MAE", "icr_mae"),
        ("Mean scenario ICR RMSE", "icr_rmse"),
        ("Minimum-headroom MAE", "minimum_headroom_mae"),
        ("Minimum-headroom RMSE", "minimum_headroom_rmse"),
    ]
    (GENERATED / "metrics_table.tex").write_text(
        "\n".join(f"{label} & {report[key]:.6f} \\\\" for label, key in metrics) + "\n"
    )
    inputs = pd.read_csv(ROOT / "data/case_study/accor.csv")
    table = []
    for _, row in inputs.iterrows():
        table.append(
            " & ".join(
                str(int(row[key]))
                for key in [
                    "year",
                    "revenue",
                    "ebitda",
                    "net_debt",
                    "lease_liability",
                    "lease_expense",
                    "interest_expense",
                ]
            )
            + r" \\"
        )
    (GENERATED / "accor_inputs.tex").write_text("\n".join(table) + "\n")
    table = []
    for _, row in accor.iterrows():
        vals = [
            "--" if pd.isna(row[k]) else f"{row[k]:.3f}"
            for k in ["icr_frozen_gaap", "icr_ifrs16", "leverage_frozen_gaap", "leverage_ifrs16"]
        ]
        table.append(" & ".join([str(int(row.year))] + vals) + r" \\")
    (GENERATED / "accor_ratios.tex").write_text("\n".join(table) + "\n")
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=ROOT)
    (RESULTS / "environment.txt").write_bytes(environment)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = [
        "analysis/scripts/generate_paper_v2_figures.py",
        "analysis/scripts/verify_paper_v2.py",
        "analysis/run_benchmark.py",
        "src/lbo/full_simulation.py",
        "src/lbo/lbo_model_analytic.py",
        "src/lbo/covenants.py",
        "src/lbo/analytic_bounds.py",
        "analysis/scripts/case_study_accor.py",
        "paper/ifrs16_lbo_ssrn_v2.tex",
        "data/case_study/accor.csv",
        "data/synthetic/operators.csv",
        "data/synthetic/scenario_parameters.csv",
        "src/lbo/data.py",
        "src/lbo/__init__.py",
        "pyproject.toml",
        "paper/REVISION_AUDIT.md",
        "paper/REPRODUCE_V2.md",
    ]
    dirty_sources = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *source_paths], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "source_commit": source_sha,
        "benchmark_commit": report["git_sha"],
        "source_files_dirty": bool(dirty_sources),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": report["seed"],
        "scenarios": len(records),
        "horizon_years": 5,
        "bootstrap_attempts": 500,
        "benchmark_command": "python -m analysis.run_benchmark --seed 42",
        "figure_command": "python -m analysis.scripts.generate_paper_v2_figures --run-benchmark",
        "sha256": {name: sha256(ROOT / name) for name in source_paths},
        "environment_sha256": sha256(RESULTS / "environment.txt"),
        "benchmark_sha256": sha256(report_path),
        "figure_sha256": {p.name: sha256(p) for p in sorted(FIGURES.glob("*"))},
        "regime_counts": counts.to_dict(),
        "overlapping_failure_counts": flags.to_dict(),
        "confusion_matrix_actual_rows_predicted_columns": cm.tolist(),
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (GENERATED / "provenance.tex").write_text(
        "Source revision (manuscript and generator): \\path{" + source_sha + "}.\\\\\n"
        "Benchmark revision: \\path{" + report["git_sha"] + "}.\\\\\n"
        f"Python {platform.python_version()}; seed 42; 200 scenarios; five annual observations per scenario; "
        "500 scenario-bootstrap attempts.\\\\\n"
        "Environment SHA-256: \\path{" + manifest["environment_sha256"] + "}.\n"
    )
    print(
        json.dumps(
            {
                "source_commit": source_sha,
                "dirty_sources": bool(dirty_sources),
                "figures": len(list(FIGURES.glob("*.pdf"))),
                "manifest": str(RESULTS / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
