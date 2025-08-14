# IFRS-16 LBO Engine: Academic Research Implementation

**A covenant-aware leveraged buyout model with IFRS-16 lease treatment for academic research and industry application.**

[![Academic Pipeline](https://img.shields.io/badge/Pipeline-Academic-blue)](./Makefile)
[![Reproducible Research](https://img.shields.io/badge/Research-Reproducible-green)](./docs/)
[![IFRS-16 Compliant](https://img.shields.io/badge/IFRS--16-Compliant-orange)](./docs/model_specification.md)

## 🎯 **Research Overview**

This repository implements a comprehensive LBO analysis framework with academic rigor, featuring:

- **IFRS-16 lease capitalization** with proper debt treatment
- **Covenant-aware modeling** with ICR and leverage monitoring  
- **Monte Carlo simulation** with statistical confidence intervals
- **Global sensitivity analysis** using Sobol indices
- **Deterministic stress testing** with academic visualization
- **Complete reproducibility** with manifests and version control

## 🚀 **Quick Start for Researchers**

### Single Case Study (Accor Analysis)
```bash
# Install dependencies
make install
make sobol-optional  # Optional: SALib for sensitivity analysis

# Run complete academic pipeline  
make paper
```

### Multi-Company Empirical Study
```bash
# Run empirical analysis across hotel operators
make empirical
```

### Interactive Exploration
```bash
# Launch Streamlit dashboard for assumption testing
streamlit run streamlit_app.py
```

## 📁 **Repository Structure**

```
├── 📊 Core Academic Model
│   ├── orchestrator_advanced.py         # Main LBO engine with academic rigor
│   ├── lbo_model.py                     # Core LBO mechanics
│   ├── fund_waterfall.py               # PE fund distribution logic
│   └── streamlit_app.py                 # Interactive dashboard
│
├── 🔬 Analysis Pipeline
│   ├── analysis/scripts/
│   │   ├── run_single_case.py          # Accor case study pipeline
│   │   └── run_empirical.py            # Multi-company analysis
│   ├── analysis/data/                   # Empirical datasets
│   └── analysis/figures/               # Generated academic figures
│
├── 📚 Academic Documentation  
│   ├── docs/model_specification.md     # Mathematical formulation
│   ├── docs/experimental_design.md     # Research methodology
│   └── ACADEMIC_SUMMARY.md             # Implementation notes
│
├── 📊 Input Data
│   ├── data/accor_assumptions.csv      # Base case assumptions
│   └── data/accor_historical_recreated.csv  # Historical validation
│
├── ✅ Quality Assurance
│   ├── tests/test_acceptance.py        # Academic acceptance tests
│   └── Makefile                        # Reproducibility pipeline
```

## 🔧 **Dependencies**

### Core Requirements
```bash
pip install numpy pandas matplotlib scipy seaborn openpyxl
```

### Optional (Enhanced Academic Features)
```bash
pip install SALib              # Sobol global sensitivity analysis
pip install numpy-financial    # Alternative IRR calculations
pip install streamlit fpdf2    # Interactive dashboard & reporting
```

### Automated Installation
```bash
make install         # Core dependencies
make sobol-optional  # SALib for Sobol analysis
make academic        # Full installation + analysis
```

## 🎯 **Academic Features**

### Statistical Rigor
- **Wilson confidence intervals** for success rate estimation
- **Bootstrap percentile CIs** for robust IRR quantiles  
- **Multiple IRR calculation methods** for cross-validation
- **Unconditional and conditional statistics** reporting

### Sensitivity Analysis
- **Sobol global sensitivity** with first-order (S₁) and total-effect (Sₜ) indices
- **Monte Carlo stress testing** with 400+ scenarios
- **Deterministic stress scenarios** (Base/Mild/Severe)
- **Two-way sensitivity heatmaps** for parameter exploration

### Reproducibility Standards
- **Deterministic seeding** (seed=42) for Monte Carlo
- **Git commit tracking** in analysis manifests
- **Complete parameter logging** with assumption fingerprinting
- **File hash verification** for figure integrity

### Academic Outputs
- **F1-F6 figure series** ready for LaTeX inclusion:
  - F1: Monte Carlo IRR distribution
  - F2: Sources & Uses waterfall  
  - F3: Exit equity bridge
  - F4: Deleveraging timeline
  - F5: Sobol sensitivity indices
  - F6: Stress scenario grid
- **CSV exports** for academic tables
- **JSON manifests** for complete provenance

## 🏗️ **IFRS-16 Implementation**

The model implements proper IFRS-16 lease treatment:

- **Lease liability capitalization** at 3.2× EBITDA
- **Lease rate assumption** of 4.5-5.0%
- **Net debt inclusion** without funding source treatment
- **Covenant impact** on leverage and ICR calculations

See [`docs/model_specification.md`](docs/model_specification.md) for mathematical details.

## 📊 **Usage Examples**

### Academic Paper Pipeline
```bash
# Generate all figures and tables for academic paper
make paper

# Output: analysis/figures/F1_monte_carlo.pdf, F2_sources_uses.pdf, etc.
# Tables: analysis/figures/stress_results.csv, sobol_indices.csv
# Manifest: analysis/figures/manifest.json
```

### Empirical Research
```bash
# Multi-company analysis across hotel operators  
make empirical

# Custom parameters
python analysis/scripts/run_single_case.py --seed 123 --n-mc 1000 --sobol 1
```

### Model Validation
```bash
# Run acceptance tests
make test

# Full academic pipeline validation
make academic
```

## 📈 **Expected Outputs**

Academic pipeline generates publication-ready materials:

- **Figures**: 6 PDF figures (F1-F6) formatted for academic journals
- **Tables**: CSV files ready for LaTeX table import
- **Manifest**: Complete reproducibility record with git hash, parameters, runtime
- **Audit Trail**: Equity vector verification and IRR cross-validation

## 🤝 **For Reviewers**

This implementation prioritizes **transparency and reproducibility**:

1. **Single source of truth**: `orchestrator_advanced.py` contains all model logic
2. **Clear documentation**: Mathematical specifications in `docs/`
3. **Automated testing**: `make test` validates core model properties
4. **Complete pipeline**: `make academic` reproduces all results
5. **Version control**: Git tracking with commit hashes in manifests

## 📚 **Academic Citations**

When using this model in academic research, please reference:

- IFRS-16 lease treatment methodology
- Monte Carlo simulation parameters (400 scenarios, seed=42)  
- Sobol sensitivity analysis implementation
- Statistical confidence interval methods (Wilson, Bootstrap)

## 🔗 **Related Documentation**

- [`docs/model_specification.md`](docs/model_specification.md) - Mathematical formulation
- [`docs/experimental_design.md`](docs/experimental_design.md) - Research methodology  
- [`ACADEMIC_SUMMARY.md`](ACADEMIC_SUMMARY.md) - Implementation transformation notes
- [`Makefile`](Makefile) - Complete reproducibility pipeline

---

**Research-grade LBO modeling with academic rigor and industry applicability.**
