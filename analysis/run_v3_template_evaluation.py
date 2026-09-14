"""Frozen training-only threshold evaluation; see V3_TEMPLATE_EVALUATION_PROTOCOL.md."""

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/paper_v2/scenario_records.csv"
OUTPUT = ROOT / "results/v3/template_evaluation"
TIE_TOLERANCE = 1e-12
SCORE = "analytic_risk_score"
LABEL = "true_failure"


def validate_records(records: pd.DataFrame) -> pd.DataFrame:
    required = {"scenario_id", "operator_id", "scenario_type", LABEL, SCORE}
    if not required.issubset(records.columns) or records.empty:
        raise ValueError("Nonempty records with all required columns are required")
    for column in ("scenario_id", "operator_id", "scenario_type"):
        if not records[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"Missing or invalid {column}")
    if records.scenario_id.duplicated().any():
        raise ValueError("Each scenario_id must occur once and belong to exactly one template")
    if records.operator_id.nunique() < 2:
        raise ValueError("At least two templates are required")
    if not records[LABEL].isin([0, 1]).all():
        raise ValueError("Failure labels must be binary 0/1")
    scores = records[SCORE].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite and in [0,1]")
    result = records.sort_values("scenario_id").reset_index(drop=True).copy()
    result[SCORE] = result[SCORE].astype(float)
    result[LABEL] = result[LABEL].astype(int)
    return result


def classification_metrics(y, predicted) -> dict:
    y, predicted = np.asarray(y), np.asarray(predicted)
    tp = int(((y == 1) & (predicted == 1)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())

    def ratio(a, b):
        return a / b if b else None

    sensitivity, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fnr": ratio(fn, tp + fn),
        "fpr": ratio(fp, tn + fp),
        "balanced_accuracy": (
            (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None
            else None
        ),
        "precision": ratio(tp, tp + fp),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def ranking_metrics(y, scores) -> dict:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
    }


def select_threshold(train: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Fit only on supplied training rows, with the frozen lowest-tied-optimum rule."""
    y, scores = train[LABEL].to_numpy(), train[SCORE].to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Training balanced accuracy requires both classes")
    candidates = np.unique(np.r_[0.0, scores, np.nextafter(1.0, np.inf)])
    objectives = np.array(
        [classification_metrics(y, scores >= t)["balanced_accuracy"] for t in candidates]
    )
    best = float(objectives.max())
    index = int(np.flatnonzero(best - objectives <= TIE_TOLERANCE)[0])
    curve = pd.DataFrame({"threshold": candidates, "train_balanced_accuracy": objectives})
    curve["selected"] = np.arange(len(curve)) == index
    return float(candidates[index]), float(objectives[index]), curve


def verify_heldout(source: pd.DataFrame, heldout: pd.DataFrame) -> None:
    """Fail closed before aggregation or writing artifacts."""
    if len(heldout) != len(source) or heldout.scenario_id.duplicated().any():
        raise ValueError("Held-out coverage is not exactly once")
    ordered = heldout.sort_values("scenario_id").reset_index(drop=True)
    if not ordered[source.columns].equals(source):
        raise ValueError("Held-out records differ from the original source")
    if not (ordered.operator_id == ordered.held_out_template).all():
        raise ValueError("Held-out template mismatch")
    for column, threshold in (
        ("predicted_failure_selected_threshold", ordered.fold_selected_threshold),
        ("predicted_failure_threshold_0_5", 0.5),
    ):
        if not np.array_equal(ordered[column], (ordered[SCORE] >= threshold).astype(int)):
            raise ValueError(f"Incorrect prediction formula: {column}")


def dispersion(values) -> dict:
    finite = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    return {
        "n_defined": len(finite),
        "min": float(finite.min()) if len(finite) else None,
        "median": float(np.median(finite)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
    }


def evaluate(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    source = validate_records(records)
    folds, predictions, curves = [], [], {}
    for template in sorted(source.operator_id.unique()):
        train = source.loc[source.operator_id != template].copy()
        test = source.loc[source.operator_id == template].copy()
        if set(train.scenario_id) & set(test.scenario_id):
            raise ValueError("Train/test overlap")
        threshold, objective, curve = select_threshold(train)
        repeat, repeat_objective, _ = select_threshold(train.iloc[::-1])
        if (threshold, objective) != (repeat, repeat_objective):
            raise ValueError("Threshold selection is not deterministic")
        curves[template] = curve
        test["held_out_template"] = template
        test["fold_selected_threshold"] = threshold
        test["predicted_failure_selected_threshold"] = (test[SCORE] >= threshold).astype(int)
        test["predicted_failure_threshold_0_5"] = (test[SCORE] >= 0.5).astype(int)
        predictions.append(test)
        fold = {
            "held_out_template": template,
            "n_train": len(train),
            "n_train_failures": int(train[LABEL].sum()),
            "n_test": len(test),
            "n_test_failures": int(test[LABEL].sum()),
            "selected_threshold": threshold,
            "train_balanced_accuracy": objective,
        }
        fold.update(
            {
                f"test_{k}": v
                for k, v in classification_metrics(
                    test[LABEL], test.predicted_failure_selected_threshold
                ).items()
            }
        )
        fold.update({f"test_{k}": v for k, v in ranking_metrics(test[LABEL], test[SCORE]).items()})
        fold.update(
            {
                f"fixed_0_5_{k}": v
                for k, v in classification_metrics(
                    test[LABEL], test.predicted_failure_threshold_0_5
                ).items()
            }
        )
        folds.append(fold)
    heldout = pd.concat(predictions).sort_values("scenario_id").reset_index(drop=True)
    verify_heldout(source, heldout)
    fold_frame = pd.DataFrame(folds)
    summary = {
        "n_scenarios": len(heldout),
        "n_templates": len(folds),
        "n_failures": int(heldout[LABEL].sum()),
        "ranking": ranking_metrics(heldout[LABEL], heldout[SCORE]),
        "pooled_recomputed_ranking": ranking_metrics(source[LABEL], source[SCORE]),
        "selected_threshold": classification_metrics(
            heldout[LABEL], heldout.predicted_failure_selected_threshold
        ),
        "fixed_0_5": classification_metrics(
            heldout[LABEL], heldout.predicted_failure_threshold_0_5
        ),
        "threshold_dispersion": dispersion(fold_frame.selected_threshold),
        "fold_metric_dispersion": {
            column: dispersion(fold_frame[column])
            for column in fold_frame
            if column.startswith(("test_", "fixed_0_5_"))
        },
        "protocol": {
            "objective": "maximize training balanced accuracy",
            "tie_rule": "lowest threshold within absolute tolerance of maximum",
            "tie_tolerance": TIE_TOLERANCE,
            "candidates": "unique training scores plus 0 and nextafter(1, +infinity)",
            "prediction_rule": "score >= threshold",
            "pr_auc_definition": "average precision; not trapezoidal PR area",
            "undefined_policy": "null for missing denominators and single-class ranking/BA",
            "uncertainty": "fold min/median/max; no confidence interval with five templates",
            "auc_interpretation": "fixed untrained scores imply identical pooled and held-out AUC",
        },
        "sanity_checks": {
            "exhaustive_unique_heldout_coverage": True,
            "unchanged_source_records": True,
            "disjoint_train_test": True,
            "deterministic_training_selection": True,
            "exact_prediction_formulas": True,
            "fresh_simulations": 0,
        },
    }
    return fold_frame, heldout, summary, curves


def main() -> None:
    records = pd.read_csv(INPUT, dtype={"scenario_id": str}, float_precision="round_trip")
    folds, heldout, summary, curves = evaluate(records)
    historical = ROOT / "results/paper_v2/benchmark_seed42.json"
    summary["historical_v2_pooled_roc_auc"] = json.loads(historical.read_text())["auc"]
    summary["provenance"] = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "inputs_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (INPUT, historical, ROOT / "docs/V3_TEMPLATE_EVALUATION_PROTOCOL.md")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "command": "python -m analysis.run_v3_template_evaluation",
    }
    # Serialize strictly before touching output, so invalid metrics cannot be published.
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUTPUT / "fold_results.csv", index=False)
    heldout.to_csv(OUTPUT / "heldout_predictions.csv", index=False)
    for template, curve in curves.items():
        curve.to_csv(OUTPUT / f"threshold_curve_train_{template}.csv", index=False)
    (OUTPUT / "summary.json").write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
