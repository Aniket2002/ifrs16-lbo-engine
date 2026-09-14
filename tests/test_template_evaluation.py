import numpy as np
import pandas as pd
import pytest

from analysis import run_v3_template_evaluation as evaluation


@pytest.fixture
def records():
    # A's misleading low-score failures make the pooled optimum 0.2;
    # excluding A gives the independently obvious separating threshold 0.8.
    return pd.DataFrame(
        {
            "scenario_id": [f"{i:04d}" for i in range(10)],
            "operator_id": ["A"] * 6 + ["B"] * 2 + ["C"] * 2,
            "scenario_type": ["toy"] * 10,
            "true_failure": [1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
            "analytic_risk_score": [0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.4, 0.8, 0.4, 0.8],
        }
    )


def test_every_selector_call_excludes_heldout_template(records, monkeypatch):
    original = evaluation.select_threshold
    calls = []

    def spy(train):
        calls.append(train.copy())
        return original(train)

    monkeypatch.setattr(evaluation, "select_threshold", spy)
    folds, _, _, _ = evaluation.evaluate(records)
    assert len(calls) == 2 * len(folds)  # fitting plus determinism check
    for i, template in enumerate(folds.held_out_template):
        for train in calls[2 * i : 2 * i + 2]:
            expected = records.loc[records.operator_id != template]
            assert template not in set(train.operator_id)
            assert set(train.scenario_id) == set(expected.scenario_id)
            assert set(train.scenario_id).isdisjoint(
                records.loc[records.operator_id == template, "scenario_id"]
            )


@pytest.mark.parametrize("column", ["true_failure", "analytic_risk_score"])
@pytest.mark.parametrize("template", ["A", "B", "C"])
def test_heldout_mutation_cannot_change_its_threshold_or_curve(records, column, template):
    folds, _, _, curves = evaluation.evaluate(records)
    changed = records.copy()
    mask = changed.operator_id == template
    changed.loc[mask, column] = 1 - changed.loc[mask, column]
    new_folds, _, _, new_curves = evaluation.evaluate(changed)
    index = list(folds.held_out_template).index(template)
    assert new_folds.iloc[index].selected_threshold == folds.iloc[index].selected_threshold
    pd.testing.assert_frame_equal(curves[template], new_curves[template])


@pytest.mark.parametrize("cross_template", [False, True])
def test_duplicate_scenario_rejected_even_across_groups(records, cross_template):
    duplicate = records.iloc[[0]].copy()
    if cross_template:
        duplicate["operator_id"] = "B"
    with pytest.raises(ValueError, match="exactly one template"):
        evaluation.evaluate(pd.concat([records, duplicate], ignore_index=True))


def test_lowest_tied_optimum():
    train = pd.DataFrame(
        {"true_failure": [0, 1, 0, 1], "analytic_risk_score": [0.1, 0.2, 0.3, 0.4]}
    )
    threshold, objective, curve = evaluation.select_threshold(train)
    assert objective == 0.75
    assert list(curve.loc[curve.train_balanced_accuracy == 0.75, "threshold"]) == [0.2, 0.4]
    assert threshold == 0.2


def test_training_only_optimum_differs_from_pooled(records):
    pooled, _, _ = evaluation.select_threshold(records)
    folds, _, _, _ = evaluation.evaluate(records)
    threshold = folds.set_index("held_out_template").loc["A", "selected_threshold"]
    assert pooled == 0.2
    assert threshold == 0.8


def test_aggregate_is_exactly_once_heldout_and_hand_counted(records):
    folds, heldout, summary, _ = evaluation.evaluate(records)
    assert len(heldout) == len(records) == folds.n_test.sum()
    assert heldout.scenario_id.is_unique
    assert set(heldout.scenario_id) == set(records.scenario_id)
    assert (heldout.operator_id == heldout.held_out_template).all()
    assert (
        heldout.predicted_failure_selected_threshold
        == (heldout.analytic_risk_score >= heldout.fold_selected_threshold)
    ).all()
    assert (heldout.predicted_failure_threshold_0_5 == (heldout.analytic_risk_score >= 0.5)).all()
    # A contributes FN=5,TN=1; B and C each contribute TP=1,FP=1.
    expected = {"tp": 2, "fp": 2, "tn": 1, "fn": 5}
    for key, value in expected.items():
        assert summary["selected_threshold"][key] == value
        assert folds[f"test_{key}"].sum() == value
    assert summary["selected_threshold"]["balanced_accuracy"] == pytest.approx((2 / 7 + 1 / 3) / 2)
    assert summary["fixed_0_5"]["fp"] == 0
    assert summary["fixed_0_5"]["tn"] == 3
    assert summary["ranking"] == summary["pooled_recomputed_ranking"]
    assert summary["ranking"]["roc_auc"] == pytest.approx(11 / 21)
    assert summary["ranking"]["pr_auc"] == pytest.approx(2 / 7 + 5 / 9)


def test_order_determinism(records):
    folds, heldout, summary, curves = evaluation.evaluate(records)
    other = evaluation.evaluate(records.sample(frac=1, random_state=17))
    pd.testing.assert_frame_equal(folds, other[0])
    pd.testing.assert_frame_equal(heldout, other[1])
    assert summary == other[2]
    for template in curves:
        pd.testing.assert_frame_equal(curves[template], other[3][template])


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_test_fold_is_retained(records, label):
    records.loc[records.operator_id == "A", "true_failure"] = label
    folds, heldout, summary, _ = evaluation.evaluate(records)
    row = folds.set_index("held_out_template").loc["A"]
    assert pd.isna(row.test_roc_auc)
    assert pd.isna(row.test_pr_auc)
    assert pd.isna(row.test_balanced_accuracy)
    assert row.n_test == 6
    assert len(heldout) == 10
    assert summary["ranking"]["roc_auc"] is not None
    missing = "test_sensitivity" if label == 0 else "test_specificity"
    assert pd.isna(row[missing])


def test_single_class_training_rejected(records):
    records.loc[records.operator_id != "A", "true_failure"] = 0
    with pytest.raises(ValueError, match="Training balanced accuracy"):
        evaluation.evaluate(records)


@pytest.mark.parametrize(
    "column,value",
    [
        ("operator_id", None),
        ("operator_id", " "),
        ("scenario_id", ""),
        ("scenario_id", 1),
        ("scenario_type", None),
        ("true_failure", np.nan),
        ("true_failure", 2),
        ("analytic_risk_score", np.nan),
        ("analytic_risk_score", np.inf),
        ("analytic_risk_score", -0.1),
        ("analytic_risk_score", 1.1),
    ],
)
def test_invalid_input_rejected(records, column, value):
    records[column] = records[column].astype(object)
    records.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluation.evaluate(records)


@pytest.mark.parametrize("corruption", ["duplicate", "drop", "label", "group", "selected", "fixed"])
def test_sanity_gate_rejects_corrupted_predictions(records, corruption):
    _, heldout, _, _ = evaluation.evaluate(records)
    if corruption == "duplicate":
        heldout = pd.concat([heldout, heldout.iloc[[0]]])
    elif corruption == "drop":
        heldout = heldout.iloc[1:]
    else:
        column = {
            "label": "true_failure",
            "group": "held_out_template",
            "selected": "predicted_failure_selected_threshold",
            "fixed": "predicted_failure_threshold_0_5",
        }[corruption]
        heldout.loc[0, column] = "wrong" if corruption == "group" else 1 - heldout.loc[0, column]
    with pytest.raises(ValueError):
        evaluation.verify_heldout(evaluation.validate_records(records), heldout)


def test_boundary_candidates_and_undefined_metrics():
    train = pd.DataFrame({"true_failure": [0, 1], "analytic_risk_score": [0.0, 1.0]})
    threshold, objective, curve = evaluation.select_threshold(train)
    assert threshold == objective == 1.0
    assert curve.threshold.iloc[0] == 0.0
    assert curve.threshold.iloc[-1] > 1.0
    metrics = evaluation.classification_metrics([0, 0], [0, 0])
    assert metrics["sensitivity"] is metrics["balanced_accuracy"] is metrics["precision"] is None
    assert metrics["specificity"] == 1.0
    assert evaluation.ranking_metrics([0, 0], [0.1, 0.2]) == {"roc_auc": None, "pr_auc": None}
