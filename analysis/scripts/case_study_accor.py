#!/usr/bin/env python3
"""
Real Case Study: Accor SA IFRS-16 Impact Analysis
Demonstrates dual-convention covenant analysis on public hospitality company
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lbo import load_case_csv, ratios_frozen_gaap, ratios_ifrs16

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "analysis" / "figures"
OUTPUT_DIR = ROOT / "output"


def run_accor_case_study():
    """Execute full Accor SA case study with dual-convention comparison."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    covenant_icr = 4.0
    covenant_lev = 3.5

    # Load public financial data
    df = load_case_csv(str(ROOT / "data" / "case_study" / "accor.csv"))
    accor = df[df["entity"] == "Accor SA"].copy()

    print("=== ACCOR SA IFRS-16 CASE STUDY ===")
    print(f"Analysis period: {accor['year'].min()}-{accor['year'].max()}")
    print(f"Revenue range: €{accor['revenue'].min():.0f}M - €{accor['revenue'].max():.0f}M")
    print(f"Lease liability: €{accor['lease_liability'].mean():.0f}M average")

    # Calculate dual-convention ratios
    results = []
    for _, row in accor.iterrows():
        year = int(row["year"])
        negative_ebitda = float(row["ebitda"]) <= 0

        # Create standardized row format for covenant calculations
        std_row = pd.Series(
            {
                "ebitda": row["ebitda"],
                "debt_senior": row["net_debt"] * 0.8,  # Scenario assumption: 80/20 split
                "debt_mezz": row["net_debt"] * 0.2,  # Scenario assumption: 80/20 split
                "lease_liability": row["lease_liability"],
                "cash": 0,  # Assume net debt already accounts for cash
                "fin_rate": row["interest_expense"] / row["net_debt"]
                if row["net_debt"] > 0
                else 0.06,
                "lease_rate": 0.04,  # Typical lease discount rate
                "rent": row.get("lease_expense", row["lease_liability"] * 0.04),
            }
        )

        # IFRS-16 ratios (includes lease liability)
        lev_ifrs16, icr_ifrs16 = ratios_ifrs16(std_row)

        # Frozen GAAP ratios (excludes lease liability)
        lev_frozen, icr_frozen = ratios_frozen_gaap(std_row)

        if negative_ebitda:
            ifrs16_state = "negative_ebitda_failure"
            frozen_state = "negative_ebitda_failure"
            breach_ifrs16 = True
            breach_frozen = True
        else:
            ifrs16_state = "ok"
            frozen_state = "ok"
            breach_ifrs16 = (icr_ifrs16 < covenant_icr) or (lev_ifrs16 > covenant_lev)
            breach_frozen = (icr_frozen < covenant_icr) or (lev_frozen > covenant_lev)

        results.append(
            {
                "year": year,
                "source_report": row.get("source_report", "Accor Universal Registration Document"),
                "source_page": row.get("source_page", "reconstructed"),
                "reconstruction_formula": row.get(
                    "reconstruction_formula",
                    "lease expense reconstructed from disclosed lease footprint",
                ),
                "icr_ifrs16": icr_ifrs16,
                "icr_frozen_gaap": icr_frozen,
                "leverage_ifrs16": lev_ifrs16,
                "leverage_frozen_gaap": lev_frozen,
                "covenant_impact_icr": icr_ifrs16 - icr_frozen,
                "covenant_impact_lev": lev_ifrs16 - lev_frozen,
                "ifrs16_analysis_state": ifrs16_state,
                "frozen_gaap_analysis_state": frozen_state,
                "breach_ifrs16": breach_ifrs16,
                "breach_frozen_gaap": breach_frozen,
            }
        )

    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\n=== DUAL-CONVENTION RATIO COMPARISON ===")
    print(
        f"ICR IFRS-16:     {results_df['icr_ifrs16'].mean():.2f} ± {results_df['icr_ifrs16'].std():.2f}"
    )
    print(
        f"ICR Frozen GAAP: {results_df['icr_frozen_gaap'].mean():.2f} ± {results_df['icr_frozen_gaap'].std():.2f}"
    )
    print(f"Average ICR impact: {results_df['covenant_impact_icr'].mean():.3f}")

    print(
        f"\nLeverage IFRS-16:     {results_df['leverage_ifrs16'].mean():.2f}x ± {results_df['leverage_ifrs16'].std():.2f}x"
    )
    print(
        f"Leverage Frozen GAAP: {results_df['leverage_frozen_gaap'].mean():.2f}x ± {results_df['leverage_frozen_gaap'].std():.2f}x"
    )
    print(f"Average Leverage impact: {results_df['covenant_impact_lev'].mean():.3f}x")

    # Covenant sensitivity analysis

    breaches_ifrs16 = (results_df["icr_ifrs16"] < covenant_icr) | (
        results_df["leverage_ifrs16"] > covenant_lev
    )
    breaches_frozen = (results_df["icr_frozen_gaap"] < covenant_icr) | (
        results_df["leverage_frozen_gaap"] > covenant_lev
    )
    breaches_ifrs16 = breaches_ifrs16 | results_df["ifrs16_analysis_state"].eq(
        "negative_ebitda_failure"
    )
    breaches_frozen = breaches_frozen | results_df["frozen_gaap_analysis_state"].eq(
        "negative_ebitda_failure"
    )

    print("\n=== COVENANT BREACH ANALYSIS ===")
    print(f"Hypothetical covenants: ICR >= {covenant_icr:.1f}, Leverage <= {covenant_lev:.1f}x")
    print(f"IFRS-16 breaches: {breaches_ifrs16.sum()}/{len(results_df)} years")
    print(f"Frozen GAAP breaches: {breaches_frozen.sum()}/{len(results_df)} years")
    print(
        f"IFRS-16 makes covenants {'TIGHTER' if breaches_ifrs16.sum() > breaches_frozen.sum() else 'LOOSER'}"
    )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ICR comparison
    ax1.plot(results_df["year"], results_df["icr_ifrs16"], "b-o", label="IFRS-16", linewidth=2)
    ax1.plot(
        results_df["year"], results_df["icr_frozen_gaap"], "r--s", label="Frozen GAAP", linewidth=2
    )
    ax1.axhline(
        y=covenant_icr,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Covenant ({covenant_icr:.1f})",
    )
    ax1.set_title("Interest Coverage Ratio Comparison")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("ICR (x)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Leverage comparison
    ax2.plot(results_df["year"], results_df["leverage_ifrs16"], "b-o", label="IFRS-16", linewidth=2)
    ax2.plot(
        results_df["year"],
        results_df["leverage_frozen_gaap"],
        "r--s",
        label="Frozen GAAP",
        linewidth=2,
    )
    ax2.axhline(
        y=covenant_lev,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Covenant ({covenant_lev:.1f}x)",
    )
    ax2.set_title("Leverage Ratio Comparison")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Net Debt / EBITDA (x)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accor_case_study.png", dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {FIGURES_DIR / 'accor_case_study.png'}")

    # Export detailed results
    results_df.to_csv(OUTPUT_DIR / "accor_case_study_results.csv", index=False)
    print(f"Detailed results exported: {OUTPUT_DIR / 'accor_case_study_results.csv'}")

    return results_df


if __name__ == "__main__":
    run_accor_case_study()
