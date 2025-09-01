"""
Analytic Headroom Dynamics for IFRS-16 LBO Models

This module provides closed-form approximations for covenant headroom paths
under IFRS-16, enabling transparent analysis of first-order elasticities
and rapid screening for covenant optimization.

Key Features:
- Closed-form recursions for financial debt and lease liabilities
- Analytic ICR and Leverage ratio paths with IFRS-16 treatment
- First-order elasticities w.r.t. all model parameters
- Validation against full simulation
- Fast screening for optimization algorithms

References:
- IFRS 16: Leases - Standard Implementation
- Leland (1994) - Corporate debt value, bond covenants
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.optimize import fsolve
import warnings


@dataclass
class AnalyticAssumptions:
    """Parameters for analytic approximation"""
    # Base case
    ebitda_0: float = 100.0  # Base EBITDA
    growth_rate: float = 0.03  # Revenue/EBITDA growth
    
    # Cash conversion approximation
    alpha: float = 0.8  # FCF/EBITDA base conversion (after WC, tax, maintenance capex)
    kappa: float = 0.05  # Growth capex penalty (drag on FCF/EBITDA)
    
    # Debt structure
    financial_debt_0: float = 400.0  # Initial financial debt
    senior_rate: float = 0.05  # Senior debt rate
    mezz_rate: float = 0.09  # Mezzanine rate
    senior_fraction: float = 0.70  # Senior / Total debt
    cash_sweep: float = 0.75  # Cash sweep rate
    
    # IFRS-16 lease treatment
    lease_liability_0: float = 320.0  # Initial lease liability (3.2x EBITDA typical)
    lease_rate: float = 0.045  # Lease liability discount rate
    lease_decay: float = 0.12  # Annual lease amortization rate
    lambda_lease: float = 3.2  # Lease liability / EBITDA ratio (if assumed constant)
    
    # Cash management
    min_cash: float = 10.0  # Minimum cash balance
    initial_cash: float = 25.0  # Starting cash
    
    # Model horizon
    n_years: int = 7  # Typical LBO horizon


@dataclass 
class AnalyticResults:
    """Results from analytic model"""
    years: np.ndarray
    ebitda: np.ndarray
    fcf: np.ndarray
    financial_debt: np.ndarray
    lease_liability: np.ndarray
    cash: np.ndarray
    net_debt: np.ndarray
    leverage_ratio: np.ndarray  # ND/EBITDA
    icr_ratio: np.ndarray  # EBITDA / Interest
    
    # Elasticities
    elasticities: Optional[Dict[str, np.ndarray]] = None


class AnalyticLBOModel:
    """
    Closed-form approximation for LBO covenant dynamics under IFRS-16
    """
    
    def __init__(self, assumptions: Optional[AnalyticAssumptions] = None):
        self.assumptions = assumptions or AnalyticAssumptions()
        self.results: Optional[AnalyticResults] = None
    
    def solve_paths(self) -> AnalyticResults:
        """
        Solve for analytic covenant paths
        """
        a = self.assumptions
        years = np.arange(a.n_years + 1)  # 0 to n_years
        
        # EBITDA growth path (deterministic)
        ebitda = a.ebitda_0 * (1 + a.growth_rate) ** years
        
        # FCF approximation: FCF_t ≈ α * EBITDA_t - κ * EBITDA_t
        # where κ represents growth capex drag
        fcf_conversion = a.alpha - a.kappa
        fcf = fcf_conversion * ebitda
        
        # Financial debt recursion with cash sweep
        # D_{t+1} = D_t + r_d * D_t - s * FCF_t
        # Solution: D_t = D_0 * (1+r_d)^t - s * Σ_{k=0}^{t-1} (1+r_d)^{t-1-k} * FCF_k
        
        financial_debt = np.zeros(len(years))
        financial_debt[0] = a.financial_debt_0
        
        # Weighted average rate
        r_debt = (a.senior_fraction * a.senior_rate + 
                 (1 - a.senior_fraction) * a.mezz_rate)
        
        for t in range(1, len(years)):
            debt_growth = financial_debt[t-1] * (1 + r_debt)
            cash_paydown = a.cash_sweep * fcf[t-1]
            financial_debt[t] = max(0, debt_growth - cash_paydown)
        
        # Lease liability path (IFRS-16)
        # Option 1: Proportional to EBITDA (L_t = λ * EBITDA_t)
        # Option 2: Decay model L_{t+1} = (1 + r_L - δ_L) * L_t
        
        # Use proportional model (simpler for analytics)
        lease_liability = a.lambda_lease * ebitda
        
        # Cash balance (simplified: minimum cash maintenance)
        cash = np.full(len(years), a.min_cash)
        cash[0] = a.initial_cash
        
        # Net debt (IFRS-16 treatment)
        net_debt = financial_debt + lease_liability - cash
        
        # Covenant ratios
        leverage_ratio = net_debt / ebitda
        
        # ICR: EBITDA / (Interest on financial debt + Interest on lease liability)
        interest_financial = financial_debt * r_debt
        interest_lease = lease_liability * a.lease_rate
        total_interest = interest_financial + interest_lease
        
        # Avoid division by zero
        icr_ratio = np.where(total_interest > 1e-6, ebitda / total_interest, np.inf)
        
        self.results = AnalyticResults(
            years=years,
            ebitda=ebitda,
            fcf=fcf,
            financial_debt=financial_debt,
            lease_liability=lease_liability,
            cash=cash,
            net_debt=net_debt,
            leverage_ratio=leverage_ratio,
            icr_ratio=icr_ratio
        )
        
        return self.results
    
    def compute_elasticities(self, epsilon: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Compute first-order elasticities via finite differences
        
        Returns:
            Dict with elasticity arrays for each parameter
        """
        if self.results is None:
            self.solve_paths()
        
        # Make sure we have results
        assert self.results is not None
        
        base_leverage = self.results.leverage_ratio.copy()
        base_icr = self.results.icr_ratio.copy()
        
        elasticities = {}
        
        # Parameters to shock
        params = [
            ('growth_rate', 'growth_rate'),
            ('cash_sweep', 'cash_sweep'), 
            ('senior_rate', 'senior_rate'),
            ('mezz_rate', 'mezz_rate'),
            ('lease_rate', 'lease_rate'),
            ('lambda_lease', 'lambda_lease'),
            ('alpha', 'alpha'),
            ('kappa', 'kappa')
        ]
        
        for param_name, attr_name in params:
            # Store original value
            original_value = getattr(self.assumptions, attr_name)
            
            # Shock up
            shocked_value = original_value * (1 + epsilon)
            setattr(self.assumptions, attr_name, shocked_value)
            
            # Resolve
            shocked_results = self.solve_paths()
            
            # Compute elasticities
            leverage_elasticity = ((shocked_results.leverage_ratio - base_leverage) / 
                                  base_leverage) / epsilon
            icr_elasticity = ((shocked_results.icr_ratio - base_icr) / 
                             base_icr) / epsilon
            
            elasticities[f'd_leverage_d_{param_name}'] = leverage_elasticity
            elasticities[f'd_icr_d_{param_name}'] = icr_elasticity
            
            # Restore original value
            setattr(self.assumptions, attr_name, original_value)
        
        # Restore base results
        self.solve_paths()
        self.results.elasticities = elasticities
        
        return elasticities
    
    def validate_against_simulation(self, simulation_results: Dict[str, np.ndarray], 
                                  max_error_leverage: float = 0.2,
                                  max_error_icr: float = 0.25) -> Dict[str, float]:
        """
        Validate analytic approximation against full simulation
        
        Args:
            simulation_results: Dict with 'leverage' and 'icr' arrays
            max_error_leverage: Maximum acceptable relative error for leverage
            max_error_icr: Maximum acceptable relative error for ICR
        
        Returns:
            Dict with error statistics
        """
        if self.results is None:
            self.solve_paths()
        
        # Make sure we have results
        assert self.results is not None
        
        # Compare leverage ratios
        sim_leverage = simulation_results['leverage']
        analytic_leverage = self.results.leverage_ratio[:len(sim_leverage)]
        
        leverage_rel_error = np.abs(analytic_leverage - sim_leverage) / np.abs(sim_leverage)
        max_leverage_error = np.max(leverage_rel_error)
        mean_leverage_error = np.mean(leverage_rel_error)
        
        # Compare ICR ratios
        sim_icr = simulation_results['icr']
        analytic_icr = self.results.icr_ratio[:len(sim_icr)]
        
        # Handle infinite ICR values
        finite_mask = np.isfinite(sim_icr) & np.isfinite(analytic_icr)
        if np.sum(finite_mask) > 0:
            icr_rel_error = np.abs(analytic_icr[finite_mask] - sim_icr[finite_mask]) / np.abs(sim_icr[finite_mask])
            max_icr_error = np.max(icr_rel_error)
            mean_icr_error = np.mean(icr_rel_error)
        else:
            max_icr_error = 0.0
            mean_icr_error = 0.0
        
        # Validation flags
        leverage_valid = max_leverage_error <= max_error_leverage
        icr_valid = max_icr_error <= max_error_icr
        
        validation_results = {
            'leverage_max_error': max_leverage_error,
            'leverage_mean_error': mean_leverage_error,
            'icr_max_error': max_icr_error,
            'icr_mean_error': mean_icr_error,
            'leverage_valid': leverage_valid,
            'icr_valid': icr_valid,
            'overall_valid': leverage_valid and icr_valid
        }
        
        if not validation_results['overall_valid']:
            warnings.warn(
                f"Analytic approximation validation failed: "
                f"Leverage error {max_leverage_error:.3f} > {max_error_leverage}, "
                f"ICR error {max_icr_error:.3f} > {max_error_icr}"
            )
        
        return validation_results
    
    def plot_paths(self, output_path: Optional[str] = None) -> Figure:
        """
        Plot analytic covenant paths (for F8_analytic_vs_sim.pdf)
        """
        if self.results is None:
            self.solve_paths()
        
        # Make sure we have results
        assert self.results is not None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Analytic LBO Dynamics', fontsize=16)
        
        years = self.results.years
        
        # Leverage ratio
        ax = axes[0, 0]
        ax.plot(years, self.results.leverage_ratio, 'b-', linewidth=2, label='ND/EBITDA')
        ax.set_xlabel('Year')
        ax.set_ylabel('Net Debt / EBITDA')
        ax.set_title('Leverage Ratio Path')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # ICR ratio
        ax = axes[0, 1]
        # Cap ICR for plotting
        icr_capped = np.minimum(self.results.icr_ratio, 20)
        ax.plot(years, icr_capped, 'g-', linewidth=2, label='EBITDA/Interest')
        ax.set_xlabel('Year')
        ax.set_ylabel('Interest Coverage Ratio')
        ax.set_title('ICR Path')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Debt components
        ax = axes[1, 0]
        ax.plot(years, self.results.financial_debt, 'r-', linewidth=2, label='Financial Debt')
        ax.plot(years, self.results.lease_liability, 'orange', linewidth=2, label='Lease Liability')
        ax.plot(years, self.results.net_debt, 'k--', linewidth=2, label='Net Debt')
        ax.set_xlabel('Year')
        ax.set_ylabel('Amount ($M)')
        ax.set_title('Debt Components')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # EBITDA and FCF
        ax = axes[1, 1]
        ax.plot(years, self.results.ebitda, 'b-', linewidth=2, label='EBITDA')
        ax.plot(years, self.results.fcf, 'g-', linewidth=2, label='Free Cash Flow')
        ax.set_xlabel('Year')
        ax.set_ylabel('Amount ($M)')
        ax.set_title('Cash Generation')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_elasticities(self, output_path: Optional[str] = None) -> Figure:
        """
        Plot first-order elasticities (F9_elasticities.pdf)
        """
        if self.results is None or self.results.elasticities is None:
            self.compute_elasticities()
        
        # Make sure we have results with elasticities
        assert self.results is not None and self.results.elasticities is not None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('First-Order Elasticities', fontsize=16)
        
        # Extract elasticities at t=1, 3, 5 (representative years)
        years_subset = [1, 3, 5]
        
        leverage_params = []
        icr_params = []
        
        for key, values in self.results.elasticities.items():
            if 'd_leverage_d_' in key:
                param_name = key.replace('d_leverage_d_', '')
                leverage_params.append((param_name, values[years_subset]))
            elif 'd_icr_d_' in key:
                param_name = key.replace('d_icr_d_', '')
                icr_params.append((param_name, values[years_subset]))
        
        # Leverage elasticities
        ax = axes[0]
        param_names = [p[0] for p in leverage_params]
        elasticity_matrix = np.array([p[1] for p in leverage_params])
        
        im = ax.imshow(elasticity_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-1, vmax=1)
        ax.set_xticks(range(len(years_subset)))
        ax.set_xticklabels([f'Year {y}' for y in years_subset])
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names)
        ax.set_title('∂(Leverage)/∂(Parameter)')
        
        # Add text annotations
        for i in range(len(param_names)):
            for j in range(len(years_subset)):
                text = ax.text(j, i, f'{elasticity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(elasticity_matrix[i, j]) < 0.5 else "white")
        
        # ICR elasticities
        ax = axes[1]
        param_names = [p[0] for p in icr_params]
        elasticity_matrix = np.array([p[1] for p in icr_params])
        
        # Cap elasticities for visualization
        elasticity_matrix_capped = np.clip(elasticity_matrix, -2, 2)
        
        im = ax.imshow(elasticity_matrix_capped, aspect='auto', cmap='RdBu_r', 
                      vmin=-2, vmax=2)
        ax.set_xticks(range(len(years_subset)))
        ax.set_xticklabels([f'Year {y}' for y in years_subset])
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(param_names)
        ax.set_title('∂(ICR)/∂(Parameter)')
        
        # Add text annotations
        for i in range(len(param_names)):
            for j in range(len(years_subset)):
                text = ax.text(j, i, f'{elasticity_matrix_capped[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(elasticity_matrix_capped[i, j]) < 1 else "white")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """CLI interface for analytic model"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analytic LBO Headroom Dynamics')
    parser.add_argument('--output-dir', default='./analysis/figures', help='Output directory')
    parser.add_argument('--validate', action='store_true', help='Run validation against simulation')
    parser.add_argument('--max-error', type=float, default=0.2, help='Maximum validation error')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Initialize model
    model = AnalyticLBOModel()
    
    # Solve paths
    print("Solving analytic covenant paths...")
    results = model.solve_paths()
    
    # Compute elasticities
    print("Computing first-order elasticities...")
    elasticities = model.compute_elasticities()
    
    # Print summary
    print("\nAnalytic Results Summary:")
    print(f"Initial Leverage: {results.leverage_ratio[0]:.2f}x")
    print(f"Year 5 Leverage: {results.leverage_ratio[5]:.2f}x")
    print(f"Initial ICR: {results.icr_ratio[0]:.1f}x")
    print(f"Year 5 ICR: {results.icr_ratio[5]:.1f}x")
    
    # Key elasticities at Year 1
    print(f"\nYear 1 Elasticities:")
    for param in ['growth_rate', 'cash_sweep', 'senior_rate']:
        lev_key = f'd_leverage_d_{param}'
        icr_key = f'd_icr_d_{param}'
        if lev_key in elasticities:
            print(f"  {param}: ∂Leverage = {elasticities[lev_key][1]:.3f}, ∂ICR = {elasticities[icr_key][1]:.3f}")
    
    # Validation against simulation (mock data for demo)
    if args.validate:
        print("\nRunning validation against simulation...")
        # Mock simulation data (would come from orchestrator_advanced.py)
        mock_sim = {
            'leverage': results.leverage_ratio + np.random.normal(0, 0.05, len(results.leverage_ratio)),
            'icr': results.icr_ratio + np.random.normal(0, 0.1, len(results.icr_ratio))
        }
        
        validation = model.validate_against_simulation(mock_sim, max_error_leverage=args.max_error)
        print(f"Validation Results:")
        print(f"  Leverage Max Error: {validation['leverage_max_error']:.3f}")
        print(f"  ICR Max Error: {validation['icr_max_error']:.3f}")
        print(f"  Overall Valid: {validation['overall_valid']}")
    
    # Generate plots
    if args.plot:
        print("Generating plots...")
        
        # Path plots
        fig1 = model.plot_paths()
        plt.savefig(f"{args.output_dir}/F8_analytic_vs_sim.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Elasticity plots  
        fig2 = model.plot_elasticities()
        plt.savefig(f"{args.output_dir}/F9_elasticities.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {args.output_dir}/")
    
    print("Analytic model completed successfully!")


if __name__ == "__main__":
    main()
