"""
Covenant Design Optimization for LBO Structures

This module implements stochastic optimization of covenant packages 
(ICR threshold, Leverage threshold, Cash sweep) under posterior 
parameter uncertainty, delivering efficient frontiers of IRR vs breach risk.

Key Features:
- ε-constraint optimization (risk-first approach)
- Pareto frontier computation (multi-objective)
- Bayesian optimization with analytic screening
- Sample Average Approximation (SAA) under posterior uncertainty
- Policy maps and visualization

References:
- Birge & Louveaux (2011) - Introduction to Stochastic Programming
- Jones et al. (1998) - Efficient Global Optimization of Expensive Black-Box Functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from scipy.stats import qmc
import time

# Optional: scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize  # type: ignore
    from skopt.space import Real  # type: ignore
    from skopt.acquisition import gaussian_ei  # type: ignore
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    gp_minimize = None
    Real = None
    gaussian_ei = None
    warnings.warn("scikit-optimize not available. Using grid search.")

# Import our modules
from lbo_model_analytic import AnalyticLBOModel, AnalyticAssumptions


@dataclass
class CovenantPackage:
    """Covenant package specification"""
    icr_threshold: float  # Minimum ICR (e.g., 2.0x)
    leverage_threshold: float  # Maximum ND/EBITDA (e.g., 5.5x) 
    cash_sweep: float  # Cash sweep rate (e.g., 0.75)
    
    def __post_init__(self):
        # Validation
        if not (1.0 <= self.icr_threshold <= 10.0):
            raise ValueError(f"ICR threshold {self.icr_threshold} out of range [1.0, 10.0]")
        if not (2.0 <= self.leverage_threshold <= 8.0):
            raise ValueError(f"Leverage threshold {self.leverage_threshold} out of range [2.0, 8.0]")
        if not (0.0 <= self.cash_sweep <= 1.0):
            raise ValueError(f"Cash sweep {self.cash_sweep} out of range [0.0, 1.0]")


@dataclass
class OptimizationBounds:
    """Parameter bounds for optimization"""
    icr_min: float = 1.5
    icr_max: float = 4.0
    leverage_min: float = 4.0
    leverage_max: float = 7.0
    sweep_min: float = 0.50
    sweep_max: float = 0.95


@dataclass
class OptimizationResult:
    """Results from covenant optimization"""
    optimal_package: CovenantPackage
    expected_irr: float
    breach_probability: float
    optimization_time: float
    n_evaluations: int
    convergence_flag: bool
    # Additional statistics
    irr_std: float = 0.0
    breach_ci: Tuple[float, float] = (0.0, 0.0)
    success_rate: float = 0.0


class LBOSimulator:
    """
    Lightweight LBO simulator for optimization
    
    This would typically interface with orchestrator_advanced.py,
    but for now we'll use a simplified version
    """
    
    def __init__(self, use_analytic_screen: bool = True):
        self.use_analytic_screen = use_analytic_screen
        self.analytic_model = AnalyticLBOModel()
        
    def evaluate_package(self, package: CovenantPackage, 
                        parameter_samples: pd.DataFrame) -> Dict:
        """
        Evaluate covenant package under parameter uncertainty
        
        Args:
            package: Covenant thresholds and sweep rate
            parameter_samples: Posterior predictive samples
            
        Returns:
            Dict with IRR and breach statistics
        """
        n_samples = len(parameter_samples)
        irr_samples = []
        breach_indicators = []
        
        for i, params in parameter_samples.iterrows():
            # Update assumptions
            assumptions = AnalyticAssumptions(
                growth_rate=params['revenue_growth'],
                senior_rate=params['senior_rate'],
                mezz_rate=params['mezz_rate'],
                lambda_lease=params['lease_multiple'],
                cash_sweep=package.cash_sweep
            )
            
            # Analytic screening (fast rejection)
            if self.use_analytic_screen:
                is_feasible = self._analytic_screen(package, assumptions)
                if not is_feasible:
                    irr_samples.append(-0.5)  # Large penalty for infeasible
                    breach_indicators.append(1)
                    continue
            
            # Full simulation (simplified)
            try:
                results = self._simulate_lbo(package, assumptions)
                irr_samples.append(results['irr'])
                breach_indicators.append(results['breach'])
            except Exception as e:
                # Handle simulation failures
                irr_samples.append(-0.5)
                breach_indicators.append(1)
        
        # Compute statistics
        irr_array = np.array(irr_samples)
        breach_array = np.array(breach_indicators)
        
        # Success rate (no breach + positive IRR)
        success_mask = (breach_array == 0) & (irr_array > 0)
        success_rate = np.mean(success_mask)
        
        # Expected IRR (conditional on success vs unconditional)
        if success_rate > 0:
            expected_irr = np.mean(irr_array[success_mask])
        else:
            expected_irr = np.mean(irr_array)  # Includes penalties
        
        # Breach probability with Wilson CI
        breach_prob = np.mean(breach_array)
        n_breach = np.sum(breach_array)
        
        # Wilson confidence interval for breach probability
        if n_samples >= 10:
            z = 1.96  # 95% CI
            p_hat = breach_prob
            n = n_samples
            
            center = (p_hat + z**2/(2*n)) / (1 + z**2/n)
            half_width = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / (1 + z**2/n)
            
            breach_ci_lower = max(0, center - half_width)
            breach_ci_upper = min(1, center + half_width)
            breach_ci = (breach_ci_lower, breach_ci_upper)
        else:
            breach_ci = (0.0, 1.0)
        
        return {
            'expected_irr': expected_irr,
            'breach_probability': breach_prob,
            'success_rate': success_rate,
            'irr_std': np.std(irr_array),
            'breach_ci': breach_ci,
            'n_success': np.sum(success_mask),
            'n_breach': n_breach,
            'irr_samples': irr_array  # For debugging
        }
    
    def _analytic_screen(self, package: CovenantPackage, 
                        assumptions: AnalyticAssumptions) -> bool:
        """
        Fast analytic screening for obvious infeasibility
        
        Returns:
            True if package appears feasible, False if obviously infeasible
        """
        try:
            # Update analytic model
            self.analytic_model.assumptions = assumptions
            self.analytic_model.assumptions.cash_sweep = package.cash_sweep
            
            # Solve paths
            results = self.analytic_model.solve_paths()
            
            # Check for obvious breaches
            min_icr = np.min(results.icr_ratio[1:6])  # Years 1-5
            max_leverage = np.max(results.leverage_ratio[1:6])
            
            # Screen with small buffer
            icr_buffer = 0.1
            leverage_buffer = 0.2
            
            icr_feasible = min_icr >= (package.icr_threshold - icr_buffer)
            leverage_feasible = max_leverage <= (package.leverage_threshold + leverage_buffer)
            
            return icr_feasible and leverage_feasible
            
        except Exception:
            return True  # Conservative: allow full simulation if analytic fails
    
    def _simulate_lbo(self, package: CovenantPackage, 
                     assumptions: AnalyticAssumptions) -> Dict[str, Union[float, int]]:
        """
        Simplified LBO simulation
        
        In practice, this would call orchestrator_advanced.py
        """
        # Update analytic model (as proxy for full simulation)
        self.analytic_model.assumptions = assumptions
        self.analytic_model.assumptions.cash_sweep = package.cash_sweep
        
        # Solve
        results = self.analytic_model.solve_paths()
        
        # Check covenants
        icr_breach = np.any(results.icr_ratio[1:6] < package.icr_threshold)
        leverage_breach = np.any(results.leverage_ratio[1:6] > package.leverage_threshold)
        breach = int(icr_breach or leverage_breach)
        
        # Simple IRR calculation (simplified)
        # In practice, would use full equity cash flow vector
        final_ebitda = results.ebitda[-1]
        exit_multiple = 12.0  # Simplified
        exit_equity = max(0, exit_multiple * final_ebitda - results.net_debt[-1])
        
        if exit_equity <= 0:
            irr = -0.3  # Wipeout
        else:
            # Rough IRR approximation
            initial_equity = 150.0  # Simplified
            n_years = len(results.years) - 1
            irr = (exit_equity / initial_equity) ** (1/n_years) - 1
        
        return {
            'irr': irr,
            'breach': breach,
            'exit_equity': exit_equity,
            'min_icr': np.min(results.icr_ratio[1:6]),
            'max_leverage': np.max(results.leverage_ratio[1:6])
        }


class CovenantOptimizer:
    """
    Covenant package optimization engine
    """
    
    def __init__(self, bounds: Optional[OptimizationBounds] = None,
                 use_analytic_screen: bool = True, seed: int = 42):
        self.bounds = bounds or OptimizationBounds()
        self.simulator = LBOSimulator(use_analytic_screen)
        self.seed = seed
        
        np.random.seed(seed)
    
    def solve_epsilon_constraint(self, parameter_samples: pd.DataFrame,
                                alpha: float = 0.10,
                                method: str = 'bayesian') -> OptimizationResult:
        """
        Solve ε-constraint problem: max E[IRR] s.t. P(breach) ≤ α
        
        Args:
            parameter_samples: Posterior predictive samples
            alpha: Maximum acceptable breach probability
            method: 'bayesian', 'grid', or 'differential_evolution'
        """
        start_time = time.time()
        
        # Objective function (negative for minimization)
        def objective(x):
            icr_threshold, leverage_threshold, cash_sweep = x
            package = CovenantPackage(icr_threshold, leverage_threshold, cash_sweep)
            
            results = self.simulator.evaluate_package(package, parameter_samples)
            
            # Penalty for constraint violation
            breach_penalty = max(0, results['breach_probability'] - alpha) * 10
            
            # Return negative IRR plus penalty
            return -(results['expected_irr'] - breach_penalty)
        
        # Constraint function
        def constraint(x):
            icr_threshold, leverage_threshold, cash_sweep = x
            package = CovenantPackage(icr_threshold, leverage_threshold, cash_sweep)
            
            results = self.simulator.evaluate_package(package, parameter_samples)
            return alpha - results['breach_probability']  # ≥ 0 for feasibility
        
        # Bounds
        bounds = [
            (self.bounds.icr_min, self.bounds.icr_max),
            (self.bounds.leverage_min, self.bounds.leverage_max),
            (self.bounds.sweep_min, self.bounds.sweep_max)
        ]
        
        if method == 'bayesian' and HAS_SKOPT and Real is not None and gp_minimize is not None:
            # Bayesian optimization
            space = [
                Real(self.bounds.icr_min, self.bounds.icr_max, name='icr'),
                Real(self.bounds.leverage_min, self.bounds.leverage_max, name='leverage'),
                Real(self.bounds.sweep_min, self.bounds.sweep_max, name='sweep')
            ]
            
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=50,
                random_state=self.seed,
                acq_func='EI'
            )
            
            optimal_x = result.x
            n_evaluations = len(result.func_vals)
            convergence_flag = True  # GP always "converges"
            
        elif method == 'differential_evolution':
            # Differential evolution
            result = differential_evolution(
                objective,
                bounds,
                seed=self.seed,
                maxiter=50,
                popsize=10
            )
            
            optimal_x = result.x
            n_evaluations = result.nfev
            convergence_flag = result.success
            
        else:
            # Grid search fallback
            optimal_x, n_evaluations = self._grid_search(objective, bounds)
            convergence_flag = True
        
        # Evaluate optimal package
        optimal_package = CovenantPackage(*optimal_x)
        final_results = self.simulator.evaluate_package(optimal_package, parameter_samples)
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            optimal_package=optimal_package,
            expected_irr=final_results['expected_irr'],
            breach_probability=final_results['breach_probability'],
            optimization_time=optimization_time,
            n_evaluations=n_evaluations,
            convergence_flag=convergence_flag,
            irr_std=final_results['irr_std'],
            breach_ci=tuple(final_results['breach_ci']) if isinstance(final_results['breach_ci'], (list, tuple)) else (0.0, 0.0),
            success_rate=final_results['success_rate']
        )
    
    def solve_pareto_frontier(self, parameter_samples: pd.DataFrame,
                             alpha_list: List[float] = [0.05, 0.10, 0.15, 0.20, 0.30],
                             method: str = 'bayesian') -> pd.DataFrame:
        """
        Compute Pareto frontier for multiple risk tolerance levels
        
        Returns:
            DataFrame with columns: alpha, icr_threshold, leverage_threshold, 
                                   cash_sweep, expected_irr, breach_probability
        """
        frontier_results = []
        
        for alpha in alpha_list:
            print(f"Solving for α = {alpha:.2f}...")
            
            try:
                result = self.solve_epsilon_constraint(
                    parameter_samples, alpha=alpha, method=method
                )
                
                frontier_results.append({
                    'alpha': alpha,
                    'icr_threshold': result.optimal_package.icr_threshold,
                    'leverage_threshold': result.optimal_package.leverage_threshold,
                    'cash_sweep': result.optimal_package.cash_sweep,
                    'expected_irr': result.expected_irr,
                    'breach_probability': result.breach_probability,
                    'success_rate': result.success_rate,
                    'irr_std': result.irr_std,
                    'optimization_time': result.optimization_time,
                    'convergence': result.convergence_flag
                })
                
            except Exception as e:
                warnings.warn(f"Optimization failed for α = {alpha}: {e}")
                continue
        
        return pd.DataFrame(frontier_results)
    
    def _grid_search(self, objective: Callable, bounds: List[Tuple[float, float]], 
                    n_points: int = 10) -> Tuple[List[float], int]:
        """
        Simple grid search fallback
        """
        # Create grid
        icr_grid = np.linspace(bounds[0][0], bounds[0][1], n_points)
        leverage_grid = np.linspace(bounds[1][0], bounds[1][1], n_points)
        sweep_grid = np.linspace(bounds[2][0], bounds[2][1], n_points)
        
        best_x = None
        best_val = np.inf
        n_evaluations = 0
        
        for icr in icr_grid:
            for leverage in leverage_grid:
                for sweep in sweep_grid:
                    x = [icr, leverage, sweep]
                    try:
                        val = objective(x)
                        n_evaluations += 1
                        
                        if val < best_val:
                            best_val = val
                            best_x = x
                    except:
                        continue
        
        return best_x or [2.0, 5.5, 0.75], n_evaluations
    
    def plot_frontier(self, frontier_df: pd.DataFrame, 
                     output_path: Optional[str] = None) -> Figure:
        """
        Plot Pareto frontier (F10_frontier.pdf)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Covenant Optimization: Pareto Frontier', fontsize=16)
        
        # Main frontier plot
        ax = axes[0]
        ax.scatter(frontier_df['breach_probability'], frontier_df['expected_irr'], 
                  s=100, c=frontier_df['alpha'], cmap='viridis', alpha=0.8)
        
        # Connect points
        ax.plot(frontier_df['breach_probability'], frontier_df['expected_irr'], 
               'k--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Breach Probability')
        ax.set_ylabel('Expected IRR')
        ax.set_title('Risk-Return Frontier')
        ax.grid(True, alpha=0.3)
        
        # Add alpha labels
        for _, row in frontier_df.iterrows():
            ax.annotate(f'α={row["alpha"]:.2f}', 
                       (row['breach_probability'], row['expected_irr']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add baseline reference (if available)
        # baseline_breach = 0.15  # Example
        # baseline_irr = 0.20     # Example
        # ax.scatter(baseline_breach, baseline_irr, s=150, c='red', marker='x', 
        #           label='Baseline Package')
        # ax.legend()
        
        # Success rate vs alpha
        ax = axes[1]
        ax.scatter(frontier_df['alpha'], frontier_df['success_rate'], 
                  s=100, c='green', alpha=0.7)
        ax.plot(frontier_df['alpha'], frontier_df['success_rate'], 'g-', alpha=0.7)
        
        ax.set_xlabel('Risk Tolerance (α)')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate vs Risk Tolerance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_policy_maps(self, frontier_df: pd.DataFrame,
                        output_path: Optional[str] = None) -> Figure:
        """
        Plot optimal policy maps (F11_policy_maps.pdf)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Optimal Covenant Policy Maps', fontsize=16)
        
        alpha_values = frontier_df['alpha'].values
        
        # ICR threshold vs alpha
        ax = axes[0, 0]
        ax.plot(alpha_values, frontier_df['icr_threshold'], 'bo-', linewidth=2)
        ax.set_xlabel('Risk Tolerance (α)')
        ax.set_ylabel('Optimal ICR Threshold')
        ax.set_title('ICR Threshold Policy')
        ax.grid(True, alpha=0.3)
        
        # Leverage threshold vs alpha
        ax = axes[0, 1]
        ax.plot(alpha_values, frontier_df['leverage_threshold'], 'ro-', linewidth=2)
        ax.set_xlabel('Risk Tolerance (α)')
        ax.set_ylabel('Optimal Leverage Threshold')
        ax.set_title('Leverage Threshold Policy')
        ax.grid(True, alpha=0.3)
        
        # Cash sweep vs alpha
        ax = axes[1, 0]
        ax.plot(alpha_values, frontier_df['cash_sweep'], 'go-', linewidth=2)
        ax.set_xlabel('Risk Tolerance (α)')
        ax.set_ylabel('Optimal Cash Sweep')
        ax.set_title('Cash Sweep Policy')
        ax.grid(True, alpha=0.3)
        
        # 2D heatmap: ICR vs Leverage with sweep as color
        ax = axes[1, 1]
        scatter = ax.scatter(frontier_df['icr_threshold'], frontier_df['leverage_threshold'],
                           c=frontier_df['cash_sweep'], s=100, cmap='plasma', alpha=0.8)
        
        ax.set_xlabel('ICR Threshold')
        ax.set_ylabel('Leverage Threshold')
        ax.set_title('Policy Space Map')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cash Sweep Rate')
        
        # Add alpha annotations
        for _, row in frontier_df.iterrows():
            ax.annotate(f'{row["alpha"]:.2f}', 
                       (row['icr_threshold'], row['leverage_threshold']),
                       xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """CLI interface for covenant optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Covenant Design Optimization')
    parser.add_argument('--priors', default='./analysis/calibration/output/posterior_samples.parquet',
                       help='Path to posterior samples')
    parser.add_argument('--output-dir', default='./analysis/optimization', help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.10, help='Risk tolerance for single optimization')
    parser.add_argument('--frontier', action='store_true', help='Compute full Pareto frontier')
    parser.add_argument('--method', choices=['bayesian', 'grid', 'differential'], default='bayesian',
                       help='Optimization method')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of posterior samples to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--screen', type=int, default=1, help='Use analytic screening (1=yes, 0=no)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load posterior samples
    print(f"Loading posterior samples from {args.priors}...")
    try:
        parameter_samples = pd.read_parquet(args.priors)
        if len(parameter_samples) > args.n_samples:
            parameter_samples = parameter_samples.sample(n=args.n_samples, random_state=args.seed)
        print(f"Using {len(parameter_samples)} samples")
    except FileNotFoundError:
        print("Posterior samples not found. Using mock data...")
        # Create mock data for testing
        np.random.seed(args.seed)
        parameter_samples = pd.DataFrame({
            'revenue_growth': np.random.normal(0.03, 0.01, args.n_samples),
            'terminal_margin': np.random.normal(0.25, 0.03, args.n_samples),
            'lease_multiple': np.random.lognormal(1.2, 0.2, args.n_samples),
            'senior_rate': np.random.normal(0.05, 0.01, args.n_samples),
            'mezz_rate': np.random.normal(0.09, 0.015, args.n_samples)
        })
    
    # Initialize optimizer
    print("Initializing covenant optimizer...")
    optimizer = CovenantOptimizer(use_analytic_screen=bool(args.screen), seed=args.seed)
    
    if args.frontier:
        # Compute Pareto frontier
        print("Computing Pareto frontier...")
        alpha_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
        
        frontier_df = optimizer.solve_pareto_frontier(
            parameter_samples, 
            alpha_list=alpha_list,
            method=args.method
        )
        
        print("\nPareto Frontier Results:")
        print(frontier_df[['alpha', 'icr_threshold', 'leverage_threshold', 
                          'cash_sweep', 'expected_irr', 'breach_probability']].round(3))
        
        # Save results
        frontier_df.to_csv(output_dir / 'pareto_frontier.csv', index=False)
        
        # Generate plots
        print("Generating frontier plots...")
        fig1 = optimizer.plot_frontier(frontier_df)
        plt.savefig(output_dir / 'F10_frontier.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig2 = optimizer.plot_policy_maps(frontier_df)
        plt.savefig(output_dir / 'F11_policy_maps.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # Single optimization
        print(f"Solving ε-constraint problem for α = {args.alpha}...")
        
        result = optimizer.solve_epsilon_constraint(
            parameter_samples,
            alpha=args.alpha,
            method=args.method
        )
        
        print(f"\nOptimal Covenant Package (α = {args.alpha}):")
        print(f"  ICR Threshold: {result.optimal_package.icr_threshold:.2f}x")
        print(f"  Leverage Threshold: {result.optimal_package.leverage_threshold:.2f}x")
        print(f"  Cash Sweep: {result.optimal_package.cash_sweep:.1%}")
        print(f"  Expected IRR: {result.expected_irr:.1%}")
        print(f"  Breach Probability: {result.breach_probability:.1%}")
        print(f"  Success Rate: {result.success_rate:.1%}")
        print(f"  Optimization Time: {result.optimization_time:.1f}s")
        print(f"  Evaluations: {result.n_evaluations}")
        
        # Save single result
        result_dict = {
            'alpha': args.alpha,
            'icr_threshold': result.optimal_package.icr_threshold,
            'leverage_threshold': result.optimal_package.leverage_threshold,
            'cash_sweep': result.optimal_package.cash_sweep,
            'expected_irr': result.expected_irr,
            'breach_probability': result.breach_probability,
            'success_rate': result.success_rate,
            'optimization_time': result.optimization_time,
            'n_evaluations': result.n_evaluations,
            'convergence': result.convergence_flag
        }
        
        with open(output_dir / 'optimal_package.json', 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    print(f"Results saved to {output_dir}/")
    print("Covenant optimization completed successfully!")


if __name__ == "__main__":
    main()
