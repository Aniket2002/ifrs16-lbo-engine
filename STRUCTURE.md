# Repository Structure Documentation

## Rationale for Organization

This repository follows scientific software best practices for reproducibility and maintainability:

### Core Package (`src/lbo/`)
- **`lbo_model.py`** - Core LBO simulation classes
- **`lbo_model_analytic.py`** - Closed-form approximations
- **`data.py`** - Data loading utilities
- **`covenants.py`** - Covenant calculation functions
- **`optimization/`** - Optimization algorithms
- **`workflows/`** - Pipeline orchestration

### Analysis & Experiments (`analysis/`)
- **`scripts/`** - Executable analysis scripts
- **`calibration/`** - Bayesian parameter fitting
- **`data/`** - Analysis-specific datasets
- **`figures/`** - Generated plots and visualizations

### Tests (`tests/`)
- Comprehensive test suite for all modules
- Integration tests and acceptance tests

### Paper (`analysis/paper/`)
- Single source of truth for LaTeX manuscript (moved into analysis/ for reproducibility)
- All figures and bibliography

### Data (`data/`, `benchmark_dataset_v1.0/`)
- Input datasets and benchmarks
- Properly versioned with metadata

### Documentation (`docs/`)
- Technical documentation
- Research methodology

## Benefits
1. **Clear separation** of concerns (core vs analysis vs tests)
2. **Proper Python packaging** with __init__.py files
3. **Reproducible experiments** in dedicated analysis/ folder  
4. **No tracked artifacts** (build files, __pycache__, outputs)
5. **Single source of truth** for paper and dependencies
