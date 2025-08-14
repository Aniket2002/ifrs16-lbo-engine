# analysis/scripts/run_empirical.py
"""
Empirical Analysis - Multi-company hotel operator study
"""

from pathlib import Path
import sys
import os
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Module path safety
ROOT = Path(__file__).resolve().parents[2]  # Go up from analysis/scripts/
sys.path.append(str(ROOT))

# Imports from orchestrator (now at root level)
from orchestrator_advanced import (
    DealAssumptions,
    run_comprehensive_lbo_analysis,
    monte_carlo_analysis,
    get_output_path
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_hotel_operators_data(csv_path: str = "paper_empirical/hotel_operators.csv") -> pd.DataFrame:
    """
    Load hotel operators data from CSV.
    Expected columns: company, entry_multiple, exit_multiple, debt_ratio, hold_years, etc.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        print("⚠️ Creating sample hotel operators data...")
        sample_data = {
            'company': ['Accor', 'Marriott', 'Hilton', 'IHG', 'Hyatt'],
            'entry_multiple': [8.5, 9.2, 8.8, 9.0, 9.5],
            'exit_multiple': [10.0, 11.0, 10.5, 10.8, 11.2],
            'debt_ratio': [0.60, 0.65, 0.62, 0.63, 0.58],
            'hold_years': [5, 5, 6, 5, 4],
            'revenue_mln': [5000, 8500, 7200, 6800, 4200],
            'ebitda_margin': [0.22, 0.24, 0.23, 0.22, 0.25],
            'lease_multiple': [3.2, 2.8, 3.0, 3.1, 2.9]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(csv_path, index=False)
        return df


def csv_to_assumptions(row: pd.Series) -> DealAssumptions:
    """
    Map CSV row to DealAssumptions correctly.
    Use years=int(r.hold_years) because dataclass field is 'years', not 'hold_years'.
    """
    return DealAssumptions(
        # Valuation
        entry_ev_ebitda=float(row['entry_multiple']),
        exit_ev_ebitda=float(row['exit_multiple']),
        debt_pct_of_ev=float(row['debt_ratio']),
        
        # Operating
        revenue0=float(row['revenue_mln']),
        ebitda_margin_start=float(row['ebitda_margin']),
        ebitda_margin_end=float(row['ebitda_margin']) + 0.03,  # 300bps expansion
        
        # Timeline - use 'years' not 'hold_years'
        years=int(row['hold_years']),
        
        # IFRS-16
        lease_liability_mult_of_ebitda=float(row.get('lease_multiple', 3.2)),
        
        # Defaults for other parameters
        rev_growth_geo=0.04,
        maintenance_capex_pct=0.025,
        growth_capex_pct=0.015,
        tax_rate=0.25,
        senior_frac=0.70,
        mezz_frac=0.20,
        senior_rate=0.045,
        mezz_rate=0.08,
        cash_sweep_pct=0.85,
        icr_hurdle=1.8,
        leverage_hurdle=9.0,
        
        # Working capital (days-based)
        days_receivables=15,
        days_payables=30,
        days_deferred_revenue=20,
        
        # Other
        min_cash=150.0,
        ifrs16_method="lease_in_debt",
        lease_amort_years=10
    )


def analyze_single_operator(company: str, assumptions: DealAssumptions) -> Dict:
    """
    Run complete analysis for a single hotel operator.
    Read MC result keys that engine actually returns.
    """
    print(f"📊 Analyzing {company}...")
    
    # Base case
    try:
        base_results = run_comprehensive_lbo_analysis(assumptions)
        
        if "error" in base_results:
            return {
                'company': company,
                'status': 'FAILED',
                'error': base_results['error'],
                'base_irr': float('nan'),
                'base_moic': float('nan')
            }
        
        metrics = base_results['metrics']
        base_irr = metrics.get('IRR', float('nan'))
        base_moic = metrics.get('MOIC', float('nan'))
        
    except Exception as e:
        return {
            'company': company,
            'status': 'FAILED',
            'error': str(e),
            'base_irr': float('nan'),
            'base_moic': float('nan')
        }
    
    # Monte Carlo
    mc = None  # Initialize to avoid unbound variable error
    try:
        mc = monte_carlo_analysis(assumptions, n=200, seed=42)
        
        # Use real MC keys that engine returns
        if mc and mc.get('Count', 0) > 0:
            p50 = float(mc["Median_IRR"])
            p10 = float(mc["P10_IRR"])
            p90 = float(mc["P90_IRR"])
            breach_rate = mc["Breaches"] / mc["N"]  # No 'Breach_Rate' key
            success_rate = float(mc["Success_Rate"])
        else:
            p50 = p10 = p90 = float('nan')
            breach_rate = 1.0
            success_rate = 0.0
            
    except Exception as e:
        print(f"  ⚠️ Monte Carlo failed for {company}: {e}")
        p50 = p10 = p90 = float('nan')
        breach_rate = 1.0
        success_rate = 0.0
    
    return {
        'company': company,
        'status': 'SUCCESS',
        'base_irr': base_irr,
        'base_moic': base_moic,
        'min_icr': metrics.get('Min_ICR', float('nan')),
        'max_leverage': metrics.get('Max_LTV', float('nan')),
        'mc_p50_irr': p50,
        'mc_p10_irr': p10,
        'mc_p90_irr': p90,
        'mc_success_rate': success_rate,
        'mc_breach_rate': breach_rate,
        'mc_scenarios': mc.get('Count', 0) if mc else 0,
        'assumptions': {
            'entry_multiple': assumptions.entry_ev_ebitda,
            'exit_multiple': assumptions.exit_ev_ebitda,
            'debt_ratio': assumptions.debt_pct_of_ev,
            'hold_years': assumptions.years,
            'lease_multiple': assumptions.lease_liability_mult_of_ebitda
        }
    }


def create_empirical_summary(results: list) -> pd.DataFrame:
    """
    Create summary DataFrame from all operator results.
    """
    summary_data = []
    
    for result in results:
        if result['status'] == 'SUCCESS':
            summary_data.append({
                'Company': result['company'],
                'Base IRR': result['base_irr'],
                'Base MOIC': result['base_moic'],
                'Min ICR': result['min_icr'],
                'Max Leverage': result['max_leverage'],
                'MC P50 IRR': result['mc_p50_irr'],
                'MC P10 IRR': result['mc_p10_irr'],
                'MC P90 IRR': result['mc_p90_irr'],
                'MC Success Rate': result['mc_success_rate'],
                'MC Breach Rate': result['mc_breach_rate'],
                'Entry Multiple': result['assumptions']['entry_multiple'],
                'Exit Multiple': result['assumptions']['exit_multiple'],
                'Debt Ratio': result['assumptions']['debt_ratio'],
                'Hold Years': result['assumptions']['hold_years'],
                'Lease Multiple': result['assumptions']['lease_multiple']
            })
        else:
            summary_data.append({
                'Company': result['company'],
                'Base IRR': float('nan'),
                'Base MOIC': float('nan'),
                'Error': result.get('error', 'Unknown error')
            })
    
    return pd.DataFrame(summary_data)


def plot_empirical_results(summary_df: pd.DataFrame, output_dir: Path):
    """
    Create plots for empirical results.
    """
    # Filter successful results
    success_df = summary_df.dropna(subset=['Base IRR'])
    
    if len(success_df) == 0:
        print("⚠️ No successful results to plot")
        return
    
    # ✅ IMPROVED: Sort by median IRR once and apply everywhere
    order = success_df.sort_values("MC P50 IRR", ascending=False)["Company"].tolist()
    success_df = success_df.set_index("Company").loc[order].reset_index()
    
    # Plot 1: IRR vs MOIC scatter
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.scatter(success_df['Base MOIC'], success_df['Base IRR'] * 100, alpha=0.7)
    ax1.set_xlabel('Base Case MOIC (x)')
    ax1.set_ylabel('Base Case IRR (%)')
    ax1.set_title('IRR vs MOIC - Hotel Operators')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Monte Carlo success rates (now sorted)
    ax2.bar(success_df['Company'], success_df['MC Success Rate'] * 100)
    ax2.set_ylabel('MC Success Rate (%)')
    ax2.set_title('Monte Carlo Success Rates')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: IRR distributions (P10-P90) (now sorted)
    companies = success_df['Company']
    p10_irrs = success_df['MC P10 IRR'] * 100
    p50_irrs = success_df['MC P50 IRR'] * 100
    p90_irrs = success_df['MC P90 IRR'] * 100
    
    x_pos = range(len(companies))
    ax3.bar(x_pos, p50_irrs, alpha=0.7, label='P50')
    ax3.errorbar(x_pos, p50_irrs, 
                yerr=[p50_irrs - p10_irrs, p90_irrs - p50_irrs],
                fmt='none', capsize=3, color='black', label='P10-P90 Range')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(companies, rotation=45)
    ax3.set_ylabel('IRR (%)')
    ax3.set_title('Monte Carlo IRR Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Leverage vs Success Rate with company labels
    ax4.scatter(success_df['Max Leverage'], success_df['MC Success Rate'] * 100, alpha=0.7)
    
    # ✅ IMPROVED: Annotate the leverage vs success scatter
    for _, r in success_df.iterrows():
        ax4.annotate(r["Company"], (r["Max Leverage"], r["MC Success Rate"] * 100),
                    xytext=(4,4), textcoords="offset points", fontsize=8)
    
    ax4.set_xlabel('Max Net Debt/EBITDA (x)')
    ax4.set_ylabel('MC Success Rate (%)')
    ax4.set_title('Leverage vs Success Rate')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "empirical_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Empirical results plot saved: {output_dir / 'empirical_results.png'}")


def main():
    """
    Main pipeline for Paper B (Empirical) across multiple hotel operators.
    """
    print("🏨 Paper B (Empirical) - Multi-operator analysis pipeline...")
    
    # Create output directory
    output_dir = ROOT / "output" / "paper_b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load hotel operators data
    print("📋 Loading hotel operators data...")
    try:
        operators_df = load_hotel_operators_data("paper_empirical/hotel_operators.csv")
    except:
        operators_df = load_hotel_operators_data(str(ROOT / "paper_empirical" / "hotel_operators.csv"))
    
    print(f"✅ Loaded {len(operators_df)} hotel operators")
    
    # Analyze each operator
    all_results = []
    
    for _, row in operators_df.iterrows():
        company = row['company']
        assumptions = csv_to_assumptions(row)
        result = analyze_single_operator(company, assumptions)
        all_results.append(result)
    
    # Create summary
    print("📊 Creating empirical summary...")
    summary_df = create_empirical_summary(all_results)
    
    # Save results
    summary_df.to_csv(output_dir / "empirical_summary.csv", index=False)
    print(f"✅ Summary saved: {output_dir / 'empirical_summary.csv'}")
    
    # Generate plots
    print("📈 Generating empirical plots...")
    plot_empirical_results(summary_df, output_dir)
    
    # Print summary statistics
    success_df = summary_df.dropna(subset=['Base IRR'])
    
    print("\n" + "="*60)
    print("🎯 PAPER B (EMPIRICAL) - ANALYSIS COMPLETE")
    print("="*60)
    print(f"📊 Operators analyzed: {len(operators_df)}")
    print(f"✅ Successful analyses: {len(success_df)}")
    
    if len(success_df) > 0:
        print(f"💰 Average Base IRR: {success_df['Base IRR'].mean():.1%}")
        print(f"📈 Average Base MOIC: {success_df['Base MOIC'].mean():.2f}x")
        print(f"🎲 Average MC Success Rate: {success_df['MC Success Rate'].mean():.1%}")
        print(f"⚖️ Average Max Leverage: {success_df['Max Leverage'].mean():.1f}x")
    
    print(f"📁 Output: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
