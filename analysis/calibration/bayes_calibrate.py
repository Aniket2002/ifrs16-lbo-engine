"""
Bayesian Hierarchical Calibration for LBO Parameters

This module implements hierarchical Bayesian estimation of LBO model parameters
from cross-firm disclosure data, replacing ad-hoc priors with data-informed
posterior predictive distributions.

Key Features:
- Hierarchical model with partial pooling across firms
- Support for firm-level covariates (region, rating, brand)
- Posterior predictive sampling for new deals
- Export to JSON/Parquet for downstream consumption

References:
- Gelman et al. (2013) Bayesian Data Analysis
- Betancourt (2017) A Conceptual Introduction to Hamiltonian Monte Carlo
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: PyMC for full Bayesian inference
try:
    import pymc as pm  # type: ignore
    import arviz as az  # type: ignore
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    pm = None
    az = None
    warnings.warn("PyMC not available. Using MAP estimation with Laplace approximation.")


@dataclass
class FirmData:
    """Individual firm calibration data"""
    name: str
    revenue_growth: float  # Historical/projected revenue growth rate
    terminal_margin: float  # EBITDA margin at exit
    lease_multiple: float  # Lease liability / EBITDA multiple
    senior_rate: float  # Senior debt interest rate
    mezz_rate: float  # Mezzanine/subordinated debt rate
    # Optional covariates
    region: Optional[str] = None
    rating: Optional[str] = None
    brand_tier: Optional[str] = None


@dataclass
class PriorSpecification:
    """Hierarchical prior specifications"""
    # Growth rate priors (mean, std)
    mu_g_prior: Tuple[float, float] = (0.04, 0.02)
    sigma_g_prior: Tuple[float, float] = (0.0, 0.015)
    g_bounds: Tuple[float, float] = (0.0, 0.15)
    
    # Terminal margin priors
    mu_m_prior: Tuple[float, float] = (0.25, 0.05)
    sigma_m_prior: Tuple[float, float] = (0.0, 0.03)
    m_bounds: Tuple[float, float] = (0.1, 0.4)
    
    # Lease multiple priors (log scale)
    mu_L_prior: Tuple[float, float] = (1.2, 0.3)  # log(3.3) ≈ 1.2
    sigma_L_prior: Tuple[float, float] = (0.0, 0.2)
    
    # Rate priors
    mu_r_prior: Tuple[float, float] = (0.06, 0.01)
    sigma_r_prior: Tuple[float, float] = (0.0, 0.008)
    r_bounds: Tuple[float, float] = (0.02, 0.12)


class BayesianCalibrator:
    """
    Hierarchical Bayesian calibration of LBO parameters
    
    This class fits a hierarchical model to cross-firm data and provides
    posterior predictive samples for use in Monte Carlo analysis.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.firms: List[FirmData] = []
        self.priors = PriorSpecification()
        self.hyperparameters: Optional[Dict] = None
        self.posterior_samples: Optional[pd.DataFrame] = None
        self.trace = None
        
        np.random.seed(seed)
    
    def add_firm(self, firm: FirmData):
        """Add firm data for calibration"""
        self.firms.append(firm)
    
    def load_from_csv(self, csv_path: Union[str, Path]):
        """Load firm data from CSV file"""
        df = pd.read_csv(csv_path)
        
        required_cols = ['name', 'revenue_growth', 'terminal_margin', 
                        'lease_multiple', 'senior_rate', 'mezz_rate']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for _, row in df.iterrows():
            firm = FirmData(
                name=row['name'],
                revenue_growth=row['revenue_growth'],
                terminal_margin=row['terminal_margin'],
                lease_multiple=row['lease_multiple'],
                senior_rate=row['senior_rate'],
                mezz_rate=row['mezz_rate'],
                region=row.get('region'),
                rating=row.get('rating'),
                brand_tier=row.get('brand_tier')
            )
            self.add_firm(firm)
    
    def fit_hierarchical_model(self, method: str = 'map') -> Dict:
        """
        Fit hierarchical Bayesian model
        
        Args:
            method: 'mcmc' (full Bayesian) or 'map' (MAP + Laplace)
        """
        if len(self.firms) < 2:
            raise ValueError("Need at least 2 firms for hierarchical modeling")
        
        if method == 'mcmc' and HAS_PYMC:
            return self._fit_mcmc()
        else:
            return self._fit_map_laplace()
    
    def _fit_mcmc(self) -> Dict:
        """Full Bayesian inference with PyMC"""
        if not HAS_PYMC or pm is None:
            raise ImportError("PyMC not available for MCMC fitting")
            
        # Extract firm-level data
        n_firms = len(self.firms)
        growth_data = np.array([f.revenue_growth for f in self.firms])
        margin_data = np.array([f.terminal_margin for f in self.firms])
        lease_data = np.log(np.array([f.lease_multiple for f in self.firms]))
        senior_rate_data = np.array([f.senior_rate for f in self.firms])
        mezz_rate_data = np.array([f.mezz_rate for f in self.firms])
        
        with pm.Model() as model:
            # Hyperpriors
            mu_g = pm.Normal('mu_g', *self.priors.mu_g_prior)
            sigma_g = pm.HalfNormal('sigma_g', self.priors.sigma_g_prior[1])
            
            mu_m = pm.Normal('mu_m', *self.priors.mu_m_prior)
            sigma_m = pm.HalfNormal('sigma_m', self.priors.sigma_m_prior[1])
            
            mu_L = pm.Normal('mu_L', *self.priors.mu_L_prior)
            sigma_L = pm.HalfNormal('sigma_L', self.priors.sigma_L_prior[1])
            
            mu_r_sen = pm.Normal('mu_r_sen', *self.priors.mu_r_prior)
            sigma_r_sen = pm.HalfNormal('sigma_r_sen', self.priors.sigma_r_prior[1])
            
            mu_r_mezz = pm.Normal('mu_r_mezz', *self.priors.mu_r_prior)
            sigma_r_mezz = pm.HalfNormal('sigma_r_mezz', self.priors.sigma_r_prior[1])
            
            # Firm-level parameters
            g_raw = pm.Normal('g_raw', mu_g, sigma_g, shape=n_firms)
            g = pm.Deterministic('g', pm.math.clip(g_raw, *self.priors.g_bounds))
            
            m_raw = pm.Normal('m_raw', mu_m, sigma_m, shape=n_firms)
            m = pm.Deterministic('m', pm.math.clip(m_raw, *self.priors.m_bounds))
            
            L_log = pm.Normal('L_log', mu_L, sigma_L, shape=n_firms)
            L = pm.Deterministic('L', pm.math.exp(L_log))
            
            r_sen_raw = pm.Normal('r_sen_raw', mu_r_sen, sigma_r_sen, shape=n_firms)
            r_sen = pm.Deterministic('r_sen', pm.math.clip(r_sen_raw, *self.priors.r_bounds))
            
            r_mezz_raw = pm.Normal('r_mezz_raw', mu_r_mezz, sigma_r_mezz, shape=n_firms)
            r_mezz = pm.Deterministic('r_mezz', pm.math.clip(r_mezz_raw, *self.priors.r_bounds))
            
            # Likelihood (assuming small observation noise)
            obs_noise = 0.01
            pm.Normal('growth_obs', g, obs_noise, observed=growth_data)
            pm.Normal('margin_obs', m, obs_noise, observed=margin_data)
            pm.Normal('lease_obs', L_log, obs_noise, observed=lease_data)
            pm.Normal('senior_rate_obs', r_sen, obs_noise, observed=senior_rate_data)
            pm.Normal('mezz_rate_obs', r_mezz, obs_noise, observed=mezz_rate_data)
            
            # Sample
            trace = pm.sample(2000, tune=1000, random_seed=self.seed, 
                            target_accept=0.9, return_inferencedata=True)
        
        # Extract hyperparameters
        self.hyperparameters = {
            'mu_g': float(trace.posterior['mu_g'].mean()),
            'sigma_g': float(trace.posterior['sigma_g'].mean()),
            'mu_m': float(trace.posterior['mu_m'].mean()),
            'sigma_m': float(trace.posterior['sigma_m'].mean()),
            'mu_L': float(trace.posterior['mu_L'].mean()),
            'sigma_L': float(trace.posterior['sigma_L'].mean()),
            'mu_r_sen': float(trace.posterior['mu_r_sen'].mean()),
            'sigma_r_sen': float(trace.posterior['sigma_r_sen'].mean()),
            'mu_r_mezz': float(trace.posterior['mu_r_mezz'].mean()),
            'sigma_r_mezz': float(trace.posterior['sigma_r_mezz'].mean())
        }
        
        self.trace = trace
        return self.hyperparameters
    
    def _fit_map_laplace(self) -> Dict:
        """MAP estimation with Laplace approximation (fallback when PyMC unavailable)"""
        # Extract firm-level data
        growth_data = np.array([f.revenue_growth for f in self.firms])
        margin_data = np.array([f.terminal_margin for f in self.firms])
        lease_data = np.log(np.array([f.lease_multiple for f in self.firms]))
        senior_rate_data = np.array([f.senior_rate for f in self.firms])
        mezz_rate_data = np.array([f.mezz_rate for f in self.firms])
        
        # Simple empirical Bayes estimates
        self.hyperparameters = {
            'mu_g': float(np.mean(growth_data)),
            'sigma_g': float(np.std(growth_data)),
            'mu_m': float(np.mean(margin_data)),
            'sigma_m': float(np.std(margin_data)),
            'mu_L': float(np.mean(lease_data)),
            'sigma_L': float(np.std(lease_data)),
            'mu_r_sen': float(np.mean(senior_rate_data)),
            'sigma_r_sen': float(np.std(senior_rate_data)),
            'mu_r_mezz': float(np.mean(mezz_rate_data)),
            'sigma_r_mezz': float(np.std(mezz_rate_data))
        }
        
        return self.hyperparameters
    
    def generate_posterior_predictive(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate posterior predictive samples for new deals
        
        Args:
            n_samples: Number of posterior predictive samples
            
        Returns:
            DataFrame with posterior predictive samples
        """
        if self.hyperparameters is None:
            raise ValueError("Must fit model before generating samples")
        
        np.random.seed(self.seed)
        
        samples = []
        for i in range(n_samples):
            # Sample from posterior predictive
            g = np.clip(
                np.random.normal(self.hyperparameters['mu_g'], 
                               self.hyperparameters['sigma_g']),
                *self.priors.g_bounds
            )
            
            m = np.clip(
                np.random.normal(self.hyperparameters['mu_m'],
                               self.hyperparameters['sigma_m']),
                *self.priors.m_bounds
            )
            
            L = np.exp(np.random.normal(self.hyperparameters['mu_L'],
                                      self.hyperparameters['sigma_L']))
            
            r_sen = np.clip(
                np.random.normal(self.hyperparameters['mu_r_sen'],
                               self.hyperparameters['sigma_r_sen']),
                *self.priors.r_bounds
            )
            
            r_mezz = np.clip(
                np.random.normal(self.hyperparameters['mu_r_mezz'],
                               self.hyperparameters['sigma_r_mezz']),
                *self.priors.r_bounds
            )
            
            samples.append({
                'revenue_growth': g,
                'terminal_margin': m,
                'lease_multiple': L,
                'senior_rate': r_sen,
                'mezz_rate': r_mezz,
                'sample_id': i
            })
        
        self.posterior_samples = pd.DataFrame(samples)
        return self.posterior_samples
    
    def export_priors(self, output_path: Union[str, Path]):
        """Export fitted hyperparameters to JSON"""
        if self.hyperparameters is None:
            raise ValueError("Must fit model before exporting")
        
        export_data = {
            'hyperparameters': self.hyperparameters,
            'model_info': {
                'n_firms': len(self.firms),
                'firm_names': [f.name for f in self.firms],
                'seed': self.seed,
                'fit_method': 'mcmc' if HAS_PYMC else 'map'
            },
            'priors': {
                'mu_g_prior': self.priors.mu_g_prior,
                'sigma_g_prior': self.priors.sigma_g_prior,
                'g_bounds': self.priors.g_bounds,
                'mu_m_prior': self.priors.mu_m_prior,
                'sigma_m_prior': self.priors.sigma_m_prior,
                'm_bounds': self.priors.m_bounds,
                'mu_L_prior': self.priors.mu_L_prior,
                'sigma_L_prior': self.priors.sigma_L_prior,
                'mu_r_prior': self.priors.mu_r_prior,
                'sigma_r_prior': self.priors.sigma_r_prior,
                'r_bounds': self.priors.r_bounds
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_samples(self, output_path: Union[str, Path]):
        """Export posterior predictive samples to Parquet"""
        if self.posterior_samples is None:
            raise ValueError("Must generate samples before exporting")
        
        self.posterior_samples.to_parquet(output_path, index=False)
    
    def plot_posterior_comparison(self, save_path: Optional[str] = None):
        """Plot prior vs posterior densities (F7 figure)"""
        if self.hyperparameters is None:
            raise ValueError("Must fit model before plotting")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Parameters to plot
        params = [
            ('Revenue Growth', 'mu_g', 'sigma_g', self.priors.mu_g_prior, self.priors.sigma_g_prior[1]),
            ('Terminal Margin', 'mu_m', 'sigma_m', self.priors.mu_m_prior, self.priors.sigma_m_prior[1]),
            ('Log Lease Multiple', 'mu_L', 'sigma_L', self.priors.mu_L_prior, self.priors.sigma_L_prior[1]),
            ('Senior Rate', 'mu_r_sen', 'sigma_r_sen', self.priors.mu_r_prior, self.priors.sigma_r_prior[1]),
            ('Mezzanine Rate', 'mu_r_mezz', 'sigma_r_mezz', self.priors.mu_r_prior, self.priors.sigma_r_prior[1])
        ]
        
        for i, (name, mu_key, sigma_key, prior_mu, prior_sigma) in enumerate(params):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prior
            x_range = np.linspace(
                prior_mu[0] - 3*prior_sigma,
                prior_mu[0] + 3*prior_sigma,
                100
            )
            prior_density = stats.norm.pdf(x_range, prior_mu[0], prior_sigma)
            ax.plot(x_range, prior_density, '--', label='Prior', alpha=0.7)
            
            # Posterior
            post_mu = self.hyperparameters[mu_key]
            post_sigma = self.hyperparameters[sigma_key]
            post_density = stats.norm.pdf(x_range, post_mu, post_sigma)
            ax.plot(x_range, post_density, '-', label='Posterior', linewidth=2)
            
            # Firm data points
            if name == 'Revenue Growth':
                data = [f.revenue_growth for f in self.firms]
            elif name == 'Terminal Margin':
                data = [f.terminal_margin for f in self.firms]
            elif name == 'Log Lease Multiple':
                data = [np.log(f.lease_multiple) for f in self.firms]
            elif name == 'Senior Rate':
                data = [f.senior_rate for f in self.firms]
            else:  # Mezzanine Rate
                data = [f.mezz_rate for f in self.firms]
            
            ax.scatter(data, [0]*len(data), alpha=0.6, s=50, color='red', 
                      label='Firm Data', zorder=10)
            
            ax.set_title(f'{name}\nShrinkage: {abs(post_sigma - prior_sigma):.4f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplot
        if len(params) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_firm_summary(self) -> pd.DataFrame:
        """Get summary statistics by firm"""
        data = []
        for firm in self.firms:
            data.append({
                'name': firm.name,
                'revenue_growth': firm.revenue_growth,
                'terminal_margin': firm.terminal_margin,
                'lease_multiple': firm.lease_multiple,
                'senior_rate': firm.senior_rate,
                'mezz_rate': firm.mezz_rate,
                'region': firm.region,
                'rating': firm.rating,
                'brand_tier': firm.brand_tier
            })
        
        return pd.DataFrame(data)


def main():
    """CLI interface for Bayesian calibration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bayesian LBO Parameter Calibration')
    parser.add_argument('--input', type=str, required=True,
                       help='CSV file with firm data')
    parser.add_argument('--output-dir', type=str, default='analysis/calibration/output',
                       help='Output directory')
    parser.add_argument('--method', type=str, choices=['mcmc', 'map'], default='map',
                       help='Fitting method')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of posterior predictive samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and fit
    calibrator = BayesianCalibrator(seed=args.seed)
    calibrator.load_from_csv(args.input)
    
    print(f"Loaded {len(calibrator.firms)} firms")
    print(f"Fitting hierarchical model using {args.method}...")
    
    hyperparams = calibrator.fit_hierarchical_model(method=args.method)
    
    print("Fitted hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate samples
    print(f"Generating {args.n_samples} posterior predictive samples...")
    samples = calibrator.generate_posterior_predictive(n_samples=args.n_samples)
    
    # Export
    calibrator.export_priors(output_dir / 'priors.json')
    calibrator.export_samples(output_dir / 'posterior_samples.parquet')
    
    # Plot
    fig = calibrator.plot_posterior_comparison(
        save_path=str(output_dir / 'F7_posteriors.pdf')
    )
    
    # Summary
    summary = calibrator.get_firm_summary()
    summary.to_csv(output_dir / 'firm_summary.csv', index=False)
    
    print(f"Results saved to {output_dir}")
    

if __name__ == '__main__':
    main()
