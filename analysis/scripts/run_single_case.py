#!/usr/bin/env python3
"""
IFRS-16 LBO Academic Experiment Runner

This script runs the complete academic analysis pipeline for the paper.
Generates all figures, tables, and outputs needed for LaTeX compilation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator_advanced import (
    read_accor_assumptions,
    run_comprehensive_lbo_analysis,
    monte_carlo_analysis,
    run_deterministic_stress_scenario,
    plot_stress_grid,
    save_stress_scenarios_csv,
    compute_sobol_indices,
    plot_sobol_indices,
    log_equity_vector_details,
    save_equity_audit_trail,
    build_equity_cf_vector,
    plot_sources_and_uses,
    plot_exit_equity_bridge,
    plot_deleveraging_path,
    plot_covenant_headroom,
    plot_sensitivity_heatmap,
    plot_monte_carlo_results,
    get_output_path
)


def run_base_case_analysis(assumptions, outdir):
    """Run base case and generate F2/F3 figures"""
    print("Running base case analysis...")
    
    results = run_comprehensive_lbo_analysis(assumptions)
    if "error" in results:
        raise RuntimeError(f"Base case failed: {results['error']}")
    
    metrics = results["metrics"]
    
    # Generate core charts  
    plot_sources_and_uses(assumptions, os.path.join(outdir, "F2_sources_uses.pdf"))
    plot_exit_equity_bridge(results["financial_projections"], metrics, assumptions, 
                           os.path.join(outdir, "F3_exit_bridge.pdf"))
    plot_deleveraging_path(metrics, assumptions, os.path.join(outdir, "F4_deleveraging.pdf"))
    
    # Equity vector analysis
    eq_vector = build_equity_cf_vector(results["financial_projections"], assumptions)
    equity_audit = log_equity_vector_details(eq_vector, assumptions)
    save_equity_audit_trail(equity_audit, os.path.join(outdir, "equity_vector_audit.json"))
    
    return results, metrics


def run_monte_carlo_experiment(assumptions, n_mc, seed, outdir):
    """Run Monte Carlo and generate histogram"""
    print(f"Running Monte Carlo simulation (n={n_mc}, seed={seed})...")
    
    mc_results = monte_carlo_analysis(assumptions, n=n_mc, seed=seed)
    
    # Generate MC histogram
    plot_monte_carlo_results(mc_results, os.path.join(outdir, "F1_monte_carlo.pdf"))
    
    # Save MC summary
    mc_summary = {
        "scenarios_requested": mc_results["N_Requested"],
        "scenarios_effective": mc_results["N_Effective"],
        "success_rate": mc_results["Success_Rate"],
        "success_rate_ci": mc_results["Success_Rate_CI"],
        "conditional_median_irr": mc_results["Median_IRR"],
        "conditional_p10": mc_results["P10_IRR"],
        "conditional_p90": mc_results["P90_IRR"],
        "unconditional_median_irr": mc_results.get("Unconditional_Median", None),
        "unconditional_p10": mc_results.get("Unconditional_P10", None),
        "unconditional_p90": mc_results.get("Unconditional_P90", None),
        "breach_count": mc_results["Breaches"],
        "priors": mc_results["Priors"],
        "success_definition": mc_results["SuccessDef"]
    }
    
    with open(os.path.join(outdir, "monte_carlo_summary.json"), 'w') as f:
        json.dump(mc_summary, f, indent=2)
    
    return mc_results


def run_stress_tests(assumptions, outdir):
    """Run deterministic stress scenarios"""
    print("Running deterministic stress scenarios...")
    
    stress_results = run_deterministic_stress_scenario(assumptions)
    
    # Get base case metrics for comparison
    base_results = run_comprehensive_lbo_analysis(assumptions)
    base_metrics = base_results["metrics"]
    
    # Generate stress grid (F6)
    plot_stress_grid(base_metrics, stress_results, os.path.join(outdir, "F6_stress_grid.pdf"))
    
    # Save stress results CSV
    save_stress_scenarios_csv(base_metrics, stress_results, os.path.join(outdir, "stress_results.csv"))
    
    return stress_results


def run_sobol_analysis(assumptions, outdir, enable_sobol=True):
    """Run Sobol sensitivity analysis (optional)"""
    if not enable_sobol:
        print("Sobol analysis disabled")
        return None
        
    print("Running Sobol sensitivity analysis...")
    
    try:
        sobol_results = compute_sobol_indices(assumptions)
        if sobol_results and 'error' not in sobol_results:
            plot_sobol_indices(sobol_results, os.path.join(outdir, "F5_sobol.pdf"))
            
            # Save Sobol indices CSV
            with open(os.path.join(outdir, "sobol_indices.csv"), 'w') as f:
                f.write("Parameter,First_Order_S1,Total_Effect_ST\n")
                for param in sobol_results['parameter_names']:
                    s1 = sobol_results['first_order'][param]
                    st = sobol_results['total_effect'][param]
                    f.write(f"{param},{s1:.4f},{st:.4f}\n")
                    
            return sobol_results
        else:
            print("Sobol analysis failed or SALib not available")
            return None
    except Exception as e:
        print(f"Sobol analysis error: {e}")
        return None


def create_manifest(base_metrics, mc_results, stress_results, sobol_results, 
                   assumptions, runtime, outdir):
    """Create comprehensive analysis manifest"""
    
    import subprocess
    import hashlib
    
    # Get git hash
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                         cwd=Path(__file__).parent.parent,
                                         text=True).strip()
    except:
        git_hash = "unknown"
    
    # Calculate file hashes for reproducibility
    figure_files = [
        "F1_monte_carlo.pdf", "F2_sources_uses.pdf", "F3_exit_bridge.pdf",
        "F4_deleveraging.pdf", "F6_stress_grid.pdf"
    ]
    
    if sobol_results:
        figure_files.append("F5_sobol.pdf")
    
    file_hashes = {}
    for fname in figure_files:
        fpath = os.path.join(outdir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                file_hashes[fname] = hashlib.md5(f.read()).hexdigest()
    
    manifest = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_hash,
            "runtime_seconds": runtime,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "analysis_type": "IFRS16_LBO_Academic"
        },
        "model_parameters": {
            "exit_ev_ebitda": float(assumptions.exit_ev_ebitda),
            "icr_hurdle": float(assumptions.icr_hurdle) if assumptions.icr_hurdle else None,
            "leverage_hurdle": float(assumptions.leverage_hurdle) if assumptions.leverage_hurdle else None,
            "lease_liability_multiplier": float(assumptions.lease_liability_mult_of_ebitda),
            "lease_rate": float(assumptions.lease_rate)
        },
        "base_case": {
            "IRR": float(base_metrics["IRR"]),
            "MOIC": float(base_metrics["MOIC"]),
            "exit_equity": float(base_metrics["Exit_Equity"]),
            "min_ICR": float(base_metrics["Min_ICR"]),
            "max_leverage": float(base_metrics["Max_Leverage"]),
            "leverage_breach": bool(base_metrics.get("Leverage_Breach", False))
        },
        "monte_carlo": {
            "scenarios_requested": mc_results["N_Requested"],
            "scenarios_effective": mc_results["N_Effective"],
            "success_rate": float(mc_results["Success_Rate"]),
            "success_rate_ci": [float(ci) for ci in mc_results["Success_Rate_CI"]],
            "conditional_median_irr": float(mc_results["Median_IRR"]),
            "unconditional_median_irr": mc_results.get("Unconditional_Median"),
            "priors": mc_results["Priors"],
            "success_definition": mc_results["SuccessDef"]
        },
        "stress_test": stress_results,
        "sobol_analysis": sobol_results,
        "computational": {
            "figures_generated": list(file_hashes.keys()),
            "file_hashes": file_hashes,
            "runtime_seconds": runtime
        }
    }
    
    with open(os.path.join(outdir, "manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved with {len(file_hashes)} figures tracked")


def main():
    parser = argparse.ArgumentParser(description="Run IFRS-16 LBO academic experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-mc", type=int, default=400, help="Monte Carlo scenarios")
    parser.add_argument("--outdir", default="paper", help="Output directory")
    parser.add_argument("--sobol", type=int, default=1, help="Enable Sobol analysis (1/0)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Starting IFRS-16 LBO academic analysis...")
    print(f"  Seed: {args.seed}")
    print(f"  Monte Carlo scenarios: {args.n_mc}")
    print(f"  Sobol analysis: {'enabled' if args.sobol else 'disabled'}")
    print(f"  Output directory: {args.outdir}")
    
    start_time = time.time()
    
    # Load assumptions
    assumptions = read_accor_assumptions()
    
    # Run experiments
    base_results, base_metrics = run_base_case_analysis(assumptions, args.outdir)
    mc_results = run_monte_carlo_experiment(assumptions, args.n_mc, args.seed, args.outdir)
    stress_results = run_stress_tests(assumptions, args.outdir)
    sobol_results = run_sobol_analysis(assumptions, args.outdir, bool(args.sobol))
    
    # Create manifest
    runtime = time.time() - start_time
    create_manifest(base_metrics, mc_results, stress_results, sobol_results,
                   assumptions, runtime, args.outdir)
    
    print(f"\nAnalysis complete in {runtime:.1f} seconds")
    print(f"All outputs saved to: {args.outdir}/")
    print("Ready for LaTeX compilation!")


if __name__ == "__main__":
    main()
