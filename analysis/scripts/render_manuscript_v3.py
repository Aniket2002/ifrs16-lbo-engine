"""Render only the eight frozen tables and six frozen figures; never run a model."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.validate_manuscript_freeze import resolve, validate

ROOT = Path(__file__).resolve().parents[2]
FREEZE = "results/v3/manuscript_freeze/"
T = "results/v3/template_evaluation/"
OPT = "results/v3/optimization_validation/"
D = "results/v3/post_optimization_diagnostics/"
B = "results/v3/bayesian_validation/"
F = "results/v3/foundation_validation.json"
FREEZE_COMMIT = "21d726606f3208bcee6e3e84ae4c357e3a33d79b"


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ref(path, keys=None, where=None, fields=None):
    r = {"artifact": path}
    if keys is not None:
        r["keys"] = keys.split("/") if isinstance(keys, str) else keys
    if where is not None:
        r["where"] = where
    if fields is not None:
        r["fields"] = fields
    return r


def display(value, style="number"):
    if value is None or value == "":
        return "undef."
    x = float(value)
    if style == "integer":
        if not x.is_integer():
            raise ValueError("noninteger count")
        return str(int(x))
    if style in ("percent", "return"):
        digits = 2 if style == "return" else 1
        return f"{100 * x:.{digits}f}\\%"
    if style == "threshold":
        return f"{x:.6f}"
    if style == "coverage":
        return f"{x:.2f}\\%"
    if style == "branch":
        return f"{x:.1f}\\%"
    if style == "multiple":
        return f"{x:.2f}"
    return f"{x:.3f}"


def count_rate(value, n):
    count = round(float(value) * n)
    if n <= 0 or abs(count / n - float(value)) > 1e-12:
        raise ValueError("rate does not correspond to an integer count")
    return f"{count}/{n} ({display(value, 'percent')})"


def empirical_curves(scores, labels):
    """Group tied frozen scores; no fitting, tuning, bootstrap or smoothing."""
    pairs = sorted(zip(scores, labels), reverse=True)
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        raise ValueError("both classes required")
    fpr, recall, precision = [0.0], [0.0], [1.0]
    tp = fp = 0
    i = 0
    while i < len(pairs):
        score = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == score:
            tp += pairs[i][1]
            fp += 1 - pairs[i][1]
            i += 1
        fpr.append(fp / negative)
        recall.append(tp / positive)
        precision.append(tp / (tp + fp))
    return fpr, recall, precision


class Renderer:
    def __init__(self, destination=ROOT):
        self.destination = Path(destination)
        self.table_dir = self.destination / "paper/generated/v3"
        self.figure_dir = self.destination / "paper/figures/v3"
        self.report_dir = self.destination / "results/v3/manuscript"
        for folder in (self.table_dir, self.figure_dir, self.report_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.cells = []
        self.macros = {}
        self.tables = {x["table_id"]: x for x in load(FREEZE + "table_plan.json")}
        self.figures = {x["figure_id"]: x for x in load(FREEZE + "figure_plan.json")}

    def cell(self, exhibit, reference, style="number", denominator=None, macro=None):
        value = resolve(ROOT, reference)
        if isinstance(value, list):
            if len(value) != 1 or len(value[0]) != 1:
                raise ValueError("cell must select exactly one CSV value")
            value = next(iter(value[0].values()))
        text = count_rate(value, denominator) if denominator else display(value, style)
        self.cells.append(
            {
                "exhibit": exhibit,
                "reference": reference,
                "raw_value": value,
                "style": style,
                "denominator": denominator,
                "rendered": text,
                "macro": macro,
            }
        )
        if macro:
            self.macros[macro] = text
        return text

    def jc(self, exhibit, path, keys, style="number", denominator=None, macro=None):
        return self.cell(exhibit, ref(path, keys), style, denominator, macro)

    def cc(self, exhibit, path, where, field, style="number", denominator=None):
        return self.cell(exhibit, ref(path, where=where, fields=[field]), style, denominator)

    @staticmethod
    def panel(headers, data, widths=None):
        spec = widths or ("l" + "r" * (len(headers) - 1))
        return "\n".join(
            [
                r"\begin{tabular}{" + spec + "}",
                r"\toprule",
                " & ".join(headers) + r" \\",
                r"\midrule",
                *[" & ".join(row) + r" \\" for row in data],
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )

    def table(self, tid, panels, note):
        plan = self.tables[tid]
        text = "% Claims: " + ", ".join(plan["claim_ids"]) + "\n"
        text += r"\begin{table}[htbp]\centering\small" + "\n"
        text += r"\caption{" + plan["title"] + r"}\label{tab:" + tid + "}\n"
        text += "\n\\medskip\n\n".join(panels)
        text += (
            "\n\\par\\smallskip\n\\begin{minipage}{\\linewidth}\\footnotesize "
            + note
            + r"\end{minipage}"
        )
        text += "\n\\end{table}\n"
        (self.table_dir / (tid + ".tex")).write_text(text, encoding="utf-8")

    def render_tables(self):
        source = "data/synthetic/operators.csv"
        defs = rows(source)
        a, b = [], []
        for x in defs:
            w = {"operator_id": x["operator_id"]}

            def c(k, style="integer"):
                return self.cc("T1", source, w, k, style)

            a.append(
                [
                    x["operator_id"][-3:],
                    *[
                        c(k)
                        for k in [
                            "revenue_0",
                            "ebitda_0",
                            "financial_debt_0",
                            "lease_liability_0",
                            "cash_0",
                        ]
                    ],
                ]
            )
            b.append(
                [
                    x["operator_id"][-3:],
                    *[
                        c(k, "percent")
                        for k in [
                            "revenue_growth_mean",
                            "revenue_growth_std",
                            "ebitda_margin_mean",
                            "cash_sweep",
                            "lease_principal_rate",
                            "lease_additions_rate",
                        ]
                    ],
                ]
            )
        self.table(
            "T1",
            [
                self.panel(["Template", "Revenue", "EBITDA", "Debt", "Lease", "Cash"], a),
                self.panel(
                    [
                        "Template",
                        r"$\mu_g$",
                        r"$\sigma_g$",
                        r"$\mu_m$",
                        "Sweep",
                        "Lease run-off",
                        "Additions",
                    ],
                    b,
                ),
            ],
            "All values are stipulated synthetic inputs; monetary quantities share the input file's currency unit. Growth, margin, sweep and lease rates are parameters, not estimated sample frequencies.",
        )

        def c(keys, style="integer"):
            return self.jc("T2", F, keys, style)

        self.table(
            "T2",
            [
                self.panel(
                    ["Check", "Archived evidence"],
                    [
                        [
                            "Independent return",
                            c("independent_return_check/actual_irr", "return")
                            + " expected and actual",
                        ],
                        [
                            "Independent MOIC",
                            c("independent_return_check/actual_moic", "multiple")
                            + " expected and actual",
                        ],
                        ["Debt schedule", "Exact match to independent schedule"],
                        [
                            "Runtime validation",
                            c("runtime_validation/scenarios_validated")
                            + " paths; "
                            + c("runtime_validation/years_validated")
                            + " scenario-years",
                        ],
                        ["Recorded invariant failures", c("runtime_validation/failures")],
                        [
                            "Adversarial corruption checks",
                            str(len(load(F)["adversarial_corruptions_detected"]))
                            + " specified corruptions detected",
                        ],
                    ],
                )
            ],
            "The recorded return comparison uses the archived tolerance. These checks establish consistency on tested paths, not correctness for every possible input or validation against external transactions.",
        )
        self.cells.append(
            {
                "exhibit": "T2",
                "reference": ref(F, "adversarial_corruptions_detected"),
                "raw_value": load(F)["adversarial_corruptions_detected"],
                "transformation": "length",
                "rendered": "6",
            }
        )
        sums = load(T + "summary.json")
        data = []
        for label, key, style, n in [
            ("True positives", "tp", "integer", None),
            ("False positives", "fp", "integer", None),
            ("True negatives", "tn", "integer", None),
            ("False negatives", "fn", "integer", None),
            ("Recall", "sensitivity", "percent", 20),
            ("Specificity", "specificity", "percent", 180),
            ("Balanced accuracy", "balanced_accuracy", "number", None),
            ("Precision", "precision", "percent", "pred"),
        ]:
            vals = []
            for policy in ["selected_threshold", "fixed_0_5"]:
                denom = sums[policy]["tp"] + sums[policy]["fp"] if n == "pred" else n
                vals.append(self.jc("T3", T + "summary.json", policy + "/" + key, style, denom))
            data.append([label, *vals])
        self.table(
            "T3",
            [self.panel(["Metric", "Transferred threshold", "Fixed threshold"], data)],
            "Both policies classify the same 200 scenarios: 20 failures and 180 non-failures. The fixed threshold is 0.5. Rates show numerator/denominator; balanced accuracy averages recall and specificity. Aggregate ROC-AUC is "
            + self.jc("T3", T + "summary.json", "ranking/roc_auc")
            + " and average precision is "
            + self.jc("T3", T + "summary.json", "ranking/pr_auc")
            + " for the unchanged score under both policies. This identity is not independent generalization evidence.",
        )
        folds = rows(T + "fold_results.csv")
        a = []
        b = []
        for x in folds:
            w = {"held_out_template": x["held_out_template"]}

            def c(k, style="number"):
                return self.cc("T4", T + "fold_results.csv", w, k, style)

            a.append(
                [
                    x["held_out_template"][-3:],
                    c("n_train", "integer") + "/" + c("n_train_failures", "integer"),
                    c("selected_threshold", "threshold"),
                    c("train_balanced_accuracy"),
                    c("n_test", "integer") + "/" + c("n_test_failures", "integer"),
                    c("test_roc_auc"),
                    c("test_pr_auc"),
                ]
            )
            b.append(
                [
                    x["held_out_template"][-3:],
                    *[c(k, "integer") for k in ["test_tp", "test_fp", "test_tn", "test_fn"]],
                    c("test_balanced_accuracy"),
                    c("fixed_0_5_balanced_accuracy"),
                ]
            )
        self.table(
            "T4",
            [
                self.panel(
                    [
                        "Template",
                        "Train n/fail",
                        "Threshold",
                        "Train BA",
                        "Test n/fail",
                        "AUC",
                        "AP",
                    ],
                    a,
                ),
                self.panel(["Template", "TP", "FP", "TN", "FN", "Transfer BA", "Fixed BA"], b),
            ],
            "BA: balanced accuracy; AP: average precision. Counts identify each class denominator (positives = TP+FN; negatives = TN+FP). Train/test cells give scenarios/failures. All threshold fitting excludes the test template. Undefined metrics for Template 005 are not zeros; this template has no failures and supports no within-template ranking inference. Thresholds are rounded for display only.",
        )
        data = []
        for label, key, style, n in [
            ("Median annualized return", "median_annualized_return", "return", None),
            ("Mean annualized return", "mean_annualized_return", "return", None),
            ("Median MOIC", "median_moic", "multiple", None),
            ("Broad failure", "broad_failure_rate", "percent", 500),
            ("Payment default", "payment_default_rate", "percent", 500),
            ("Total equity loss", "total_equity_loss_rate", "percent", 500),
        ]:
            data.append(
                [
                    label,
                    *[
                        self.jc(
                            "T5",
                            OPT + "summary.json",
                            "primary_aggregate_heldout/" + pol + "/" + key,
                            style,
                            n,
                        )
                        for pol in ["optimized", "reference"]
                    ],
                ]
            )
        folds = rows(OPT + "fold_selection.csv")
        b = []
        for x in folds:
            w = {"held_out_template": x["held_out_template"]}

            def c(k, style):
                return self.cc("T5", OPT + "fold_selection.csv", w, k, style)

            b.append(
                [
                    x["held_out_template"][-3:],
                    c("selected_debt_multiple", "multiple"),
                    c("selected_amortisation_rate", "percent"),
                    c("selected_cash_sweep", "percent"),
                    c("test_optimized_median_annualized_return", "return"),
                    c("test_reference_median_annualized_return", "return"),
                ]
            )
        self.table(
            "T5",
            [
                self.panel(["Aggregate metric", "Selected policy", "Reference"], data),
                self.panel(
                    [
                        "Template",
                        "Debt/EBITDA",
                        "Amort.",
                        "Sweep",
                        "Selected return",
                        "Ref. return",
                    ],
                    b,
                ),
            ],
            "Classification B: methodological demonstration only. Each policy is evaluated on the same 500 operating scenarios, with 100 per held-out template; the paired rows are not independent samples. Fold returns are medians (n=100 each); amortisation and sweep are stipulated policy parameters. Returns use the limited-liability sponsor convention. Grid-boundary selection is not an economic optimum.",
        )
        load(B + "recovery_summary.json")
        data = []
        for label, key, style in [
            ("Recovery runs", "n_runs", "integer"),
            ("Parameter evaluations", "n_parameter_evaluations", "integer"),
            ("Maximum R-hat", "diagnostic_extrema/max_r_hat", "number"),
            ("Minimum bulk ESS", "diagnostic_extrema/min_bulk_ess", "number"),
            ("Minimum tail ESS", "diagnostic_extrema/min_tail_ess", "number"),
            ("Divergences", "diagnostic_extrema/total_divergences", "integer"),
        ]:
            data.append([label, self.jc("A1", B + "recovery_summary.json", key, style)])
        data += [
            ["Frozen R-hat gate", "Failed"],
            ["Frozen tail-ESS gate", "Failed"],
            ["Material prior sensitivity", "Present"],
            ["Input provenance", "Unverified/stipulated"],
            ["Main-results admission", "C: excluded"],
        ]
        self.table(
            "A1",
            [self.panel(["Diagnostic", "Archived result"], data)],
            "The 18 recovery runs yield 180 parameter evaluations, not 180 independent datasets. Calibration and prior sensitivity use ten stipulated named-firm rows. Synthetic recovery does not override the failed diagnostic and prior-sensitivity gates; posterior samples were not integrated into LBO results.",
        )
        breakdown = rows(D + "template004_regime_breakdown.csv")
        a = []
        b = []
        for x in breakdown:
            n = int(x["scenario_count"])
            w = {k: x[k] for k in ["evaluation_policy", "scenario_type"]}

            def c(k, style="return", denom=None):
                return self.cc("A2", D + "template004_regime_breakdown.csv", w, k, style, denom)

            name = (
                ("Selected" if x["evaluation_policy"] == "optimized" else "Reference")
                + " "
                + x["scenario_type"]
            )
            a.append([name, str(n), c("median_annualized_return"), c("p10_annualized_return")])
            b.append(
                [
                    name,
                    str(n),
                    *[
                        c(k + "_rate", "percent", n)
                        for k in [
                            "broad_failure",
                            "payment_default",
                            "insolvency",
                            "covenant_breach",
                            "total_equity_loss",
                        ]
                    ],
                ]
            )
        comp = rows(D + "regime_conditioned_template_comparison.csv")
        cdata = []
        for template in sorted({x["held_out_template"] for x in comp}):
            vals = []
            for regime in ["base", "downside", "distressed"]:
                w = {
                    "held_out_template": template,
                    "evaluation_policy": "optimized",
                    "scenario_type": regime,
                }
                row = next(x for x in comp if all(x[k] == v for k, v in w.items()))
                vals.append(
                    self.cc(
                        "A2",
                        D + "regime_conditioned_template_comparison.csv",
                        w,
                        "median_annualized_return",
                        "return",
                    )
                    + " ("
                    + row["scenario_count"]
                    + ")"
                )
            cdata.append([template[-3:], *vals])
        self.table(
            "A2",
            [
                self.panel(["Template 004 cell", "n", "Median return", "P10 return"], a),
                r"{\scriptsize\setlength{\tabcolsep}{3pt}"
                + self.panel(
                    [
                        "Cell",
                        "n",
                        "Broad fail",
                        "Pay default",
                        "Insolvency",
                        "Covenant",
                        "Equity loss",
                    ],
                    b,
                )
                + "}",
                self.panel(
                    [
                        "Selected-policy template",
                        "Base return (n)",
                        "Downside return (n)",
                        "Distressed return (n)",
                    ],
                    cdata,
                ),
            ],
            "Classification B; same synthetic system, descriptive small cells. Risk entries are count/n (percent); return percentages have n in the same row. Template 004 regime counts are 48/100, 31/100 and 21/100, close to stipulated probabilities 50/30/20 percent. Reference distressed P10 is the limited-liability floor: 7/21 zero recoveries (33.3 percent) place it at -100 percent. No confidence intervals are inferred.",
        )
        data = []
        cov = load(D + "coverage.json")["files"]
        for file, label in [
            ("analysis\\optimization\\financing_policy.py", "Search kernel"),
            ("analysis\\run_v3_optimization.py", "Runner"),
        ]:
            cov[file]["summary"]

            def c(k, style):
                return self.jc("A3", D + "coverage.json", ["files", file, "summary", k], style)

            data.append(
                [
                    label,
                    c("percent_statements_covered", "coverage")
                    + " ("
                    + c("covered_lines", "integer")
                    + "/"
                    + c("num_statements", "integer")
                    + ")",
                    c("percent_branches_covered", "branch")
                    + " ("
                    + c("covered_branches", "integer")
                    + "/"
                    + c("num_branches", "integer")
                    + ")",
                ]
            )
        self.table(
            "A3",
            [self.panel(["Module", "Statement coverage", "Branch coverage"], data)],
            "Focused optimization coverage only. The runner has lower branch coverage than the search kernel. Coverage alone does not validate the optimizer; independent toy, held-out independence and runtime checks provide separate evidence. Statement and branch coverage are distinct measures.",
        )

    def render_values(self):
        for name, key, style in [
            ("AUC", "ranking/roc_auc", "number"),
            ("AP", "ranking/pr_auc", "number"),
            ("N", "n_scenarios", "integer"),
            ("Failures", "n_failures", "integer"),
            ("Templates", "n_templates", "integer"),
        ]:
            self.jc("prose", T + "summary.json", key, style, macro=name)
        for prefix, policy in [("Transfer", "selected_threshold"), ("Fixed", "fixed_0_5")]:
            for suffix, key, style in [
                ("TP", "tp", "integer"),
                ("FP", "fp", "integer"),
                ("TN", "tn", "integer"),
                ("FN", "fn", "integer"),
                ("BA", "balanced_accuracy", "number"),
            ]:
                self.jc(
                    "prose", T + "summary.json", policy + "/" + key, style, macro=prefix + suffix
                )
            self.jc(
                "prose",
                T + "summary.json",
                policy + "/sensitivity",
                "percent",
                20,
                macro=prefix + "Recall",
            )
            self.jc(
                "prose",
                T + "summary.json",
                policy + "/specificity",
                "percent",
                180,
                macro=prefix + "Specificity",
            )
        (self.table_dir / "values.tex").write_text(
            "% Frozen numbers; render-time rounding only.\n"
            + "".join(
                r"\expandafter\def\csname v" + k + r"\endcsname{" + v + "}\n"
                for k, v in self.macros.items()
            ),
            encoding="utf-8",
        )

    def savefig(self, fid, fig):
        fig.savefig(
            self.figure_dir / (fid + ".pdf"),
            bbox_inches="tight",
            metadata={
                "Creator": "Frozen v3 exhibit renderer",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        fig.savefig(
            self.figure_dir / (fid + ".png"),
            bbox_inches="tight",
            dpi=180,
            metadata={"Software": "Frozen v3 exhibit renderer"},
        )
        plt.close(fig)

    def render_figures(self):
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "pdf.fonttype": 42,
                "axes.labelsize": 9,
                "axes.titlesize": 10,
                "savefig.facecolor": "white",
            }
        )
        blue = "#24577C"
        orange = "#B75B25"
        grey = "#65707A"
        fig, ax = plt.subplots(figsize=(7, 2.15))
        ax.set_axis_off()
        blocks = [
            ("Engine correctness", "Independent checks\nRuntime invariants"),
            ("Ranking performance", "Fixed score\nSynthetic discrimination"),
            ("Decision usefulness", "Threshold transfer\nArchetype heterogeneity"),
        ]
        for i, (title, body) in enumerate(blocks):
            x = 0.165 + 0.335 * i
            ax.text(
                x,
                0.62,
                title,
                ha="center",
                va="center",
                weight="bold",
                color=blue,
                transform=ax.transAxes,
            )
            ax.text(
                x,
                0.34,
                body,
                ha="center",
                va="center",
                transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=.65", "fc": "#F0F4F7", "ec": blue},
            )
            if i < 2:
                ax.text(
                    x + 0.167,
                    0.59,
                    r"$\nRightarrow$",
                    ha="center",
                    fontsize=23,
                    transform=ax.transAxes,
                )
        ax.text(
            0.5,
            -0.02,
            "Passing one layer does not establish the next.",
            ha="center",
            transform=ax.transAxes,
        )
        self.savefig("F1", fig)
        pred = rows(T + "heldout_predictions.csv")
        scores = [float(r["analytic_risk_score"]) for r in pred]
        labels = [int(r["true_failure"]) for r in pred]
        fpr, rec, pre = empirical_curves(scores, labels)
        summary = load(T + "summary.json")
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.05))
        axs[0].plot(fpr, rec, color=blue)
        axs[0].set(
            xlabel="False-positive rate",
            ylabel="Recall",
            title=f"ROC-AUC = {summary['ranking']['roc_auc']:.3f}",
            xlim=(0, 1),
            ylim=(0, 1.03),
        )
        axs[1].step(rec, pre, where="post", color=blue)
        axs[1].set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"Average precision = {summary['ranking']['pr_auc']:.3f}",
            xlim=(0, 1),
            ylim=(0, 1.03),
        )
        fig.suptitle("Fixed screening score: 200 scenarios, 20 failures", fontsize=10)
        fig.tight_layout()
        self.savefig("F2", fig)
        folds = rows(T + "fold_results.csv")
        xx = np.arange(5)
        fig, axs = plt.subplots(1, 2, figsize=(7, 3.45))
        for offset, key, label, color in [
            (-0.17, "test_balanced_accuracy", "Transferred", blue),
            (0.17, "fixed_0_5_balanced_accuracy", "Fixed 0.5", grey),
        ]:
            axs[0].bar(
                xx + offset,
                [float(x[key]) if x[key] else np.nan for x in folds],
                width=0.32,
                label=label,
                color=color,
            )
        axs[0].text(4, 0.12, "undefined", ha="center", rotation=90, color=grey)
        axs[0].set_ylim(0, 1.18)
        axs[0].legend(frameon=False, fontsize=8)
        axs[0].set_ylabel("Balanced accuracy")
        for i, x in enumerate(folds):
            axs[0].text(i, 1.07, f"{float(x['selected_threshold']):.6f}", ha="center", fontsize=6.6)
        bottom = np.zeros(5)
        for key, label, color in [
            ("test_tn", "TN", "#C5D6E3"),
            ("test_tp", "TP", blue),
            ("test_fn", "FN", orange),
            ("test_fp", "FP", "#D99264"),
        ]:
            vals = np.array([int(x[key]) for x in folds])
            axs[1].bar(xx, vals, bottom=bottom, color=color, label=label)
            bottom += vals
        axs[1].set_ylabel("Scenario count")
        axs[1].legend(frameon=False, ncol=2, fontsize=8)
        axs[1].set_ylim(0, 65)
        axs[1].text(3, 51, "37 of 38\nfalse positives", ha="center", color=orange, fontsize=8)
        tick = [
            x["held_out_template"][-3:] + "\n" + x["n_test"] + "/" + x["n_test_failures"]
            for x in folds
        ]
        for ax in axs:
            ax.set_xticks(xx, tick)
            ax.set_xlabel("Template; n / failures")
        fig.tight_layout()
        self.savefig("F3", fig)
        fig, axs = plt.subplots(1, 5, figsize=(7, 3.1), sharey=True)
        for ax, x in zip(axs, folds):
            rr = [r for r in pred if r["operator_id"] == x["held_out_template"]]
            for k, color in [(0, grey), (1, orange)]:
                yy = [float(r["analytic_risk_score"]) for r in rr if int(r["true_failure"]) == k]
                ax.scatter([k] * len(yy), yy, s=16, color=color, alpha=0.7)
            ax.axhline(float(x["selected_threshold"]), color=blue, linestyle="--", linewidth=1)
            ax.set_xticks([0, 1], ["No fail", "Fail"], rotation=45)
            ax.set_title(
                x["held_out_template"][-3:]
                + "\nn="
                + x["n_test"]
                + ", fail="
                + x["n_test_failures"],
                fontsize=8,
            )
        axs[0].set_ylabel("Screening score")
        fig.tight_layout()
        self.savefig("A-F1", fig)
        profile = rows(D + "debt_boundary_profile.csv")
        fig, ax = plt.subplots(figsize=(7, 3.1))
        for template in sorted({r["held_out_template"] for r in profile}):
            rr = sorted(
                [r for r in profile if r["held_out_template"] == template],
                key=lambda r: float(r["debt_multiple"]),
            )
            yy = [
                100 * float(r["best_feasible_median_annualized_return"])
                if r["feasible_candidate_exists"] == "True"
                else np.nan
                for r in rr
            ]
            ax.plot(
                [float(r["debt_multiple"]) for r in rr],
                yy,
                marker="o",
                markersize=4,
                label=template[-3:],
            )
        ax.set(
            xlabel="Observed debt / EBITDA",
            ylabel="Best feasible training median return (%)",
            title="Methodological demonstration; 400 training scenarios per fold",
            xlim=(1.45, 3.05),
        )
        ax.legend(title="Held-out template", ncol=5, frameon=False, fontsize=8)
        fig.tight_layout()
        self.savefig("A-F2", fig)
        comp = rows(D + "regime_conditioned_template_comparison.csv")
        fig, axs = plt.subplots(1, 3, figsize=(7, 3.1), sharey=True)
        for ax, regime in zip(axs, ["base", "downside", "distressed"]):
            rr = sorted(
                [
                    r
                    for r in comp
                    if r["evaluation_policy"] == "optimized" and r["scenario_type"] == regime
                ],
                key=lambda r: r["held_out_template"],
            )
            ax.bar(
                range(5),
                [100 * float(r["median_annualized_return"]) for r in rr],
                color=[orange if r["held_out_template"].endswith("004") else blue for r in rr],
            )
            ax.set_xticks(
                range(5),
                [r["held_out_template"][-3:] + "\n" + r["scenario_count"] for r in rr],
                fontsize=7,
            )
            ax.set_title(regime.capitalize())
            ax.set_xlabel("Template / n")
            ax.axhline(0, color=grey, linewidth=0.5)
        axs[0].set_ylabel("Median annualized sponsor return (%)")
        fig.tight_layout()
        self.savefig("A-F3", fig)

    def run(self):
        check = validate(ROOT)
        if not check["passed"]:
            raise ValueError(check["errors"])
        self.render_tables()
        self.render_values()
        self.render_figures()
        if not validate(ROOT)["passed"]:
            raise ValueError("frozen inputs changed")
        artifacts = {
            p.relative_to(self.destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [self.table_dir, self.figure_dir]
            for p in sorted(folder.iterdir())
            if p.name != "provenance.tex"
        }
        report = {
            "claim_freeze_commit": FREEZE_COMMIT,
            "input_only": True,
            "no_model_execution": True,
            "tables": list(self.tables.values()),
            "figures": list(self.figures.values()),
            "numeric_cells": self.cells,
            "output_sha256": artifacts,
            "protected_hash_check": check,
            "rounding": "Frozen manuscript policy, applied only at rendering; count/rate checks reject noninteger recovery.",
        }
        (self.report_dir / "render_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report


if __name__ == "__main__":
    result = Renderer().run()
    print(
        f"Rendered {len(result['tables'])} tables and {len(result['figures'])} figures from frozen inputs."
    )
