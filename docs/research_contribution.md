# Research Contribution Statement

## Primary Contribution

This paper presents the first open-source, academically-rigorous LBO modeling framework that fully incorporates **IFRS-16 lease accounting** with **quantified uncertainty analysis**. The contribution is methodological: we provide a reproducible engine for lease-adjusted covenant analysis in leveraged buyout transactions.

## Technical Innovations

### 1. IFRS-16 Integration in LBO Models
- **Novel treatment**: Lease liabilities included in net debt calculations for covenant testing
- **Academic rigor**: Clear mathematical specification of balance sheet linkages
- **Practical relevance**: Addresses post-2019 accounting standard impact on LBO structuring

### 2. Covenant-Aware Monte Carlo Framework  
- **Methodological advance**: Success probability quantification with explicit covenant modeling
- **Statistical rigor**: Wilson confidence intervals, bootstrap percentile CIs
- **Reproducibility**: Fixed seeds, Git commits, comprehensive manifests

### 3. Global Sensitivity Analysis for LBO Returns
- **Sobol indices**: First application to LBO equity returns with IFRS-16 considerations  
- **Academic standard**: Saltelli sampling, total-effect indices for parameter interactions
- **Interpretability**: Quantifies relative importance of exit multiples vs operational metrics

## Empirical Validation

### Cross-Sectional Evidence
- **Sample**: 6 major hotel operators with standardized assumptions
- **Methodology**: Consistent Monte Carlo protocol across entities
- **Findings**: Success rates vary 60-85%, driven primarily by leverage capacity

### Ablation Studies
- **IFRS-16 effect**: Quantified impact on covenant headroom and IRR distributions
- **Robustness checks**: Cash sweep rates, working capital methodologies
- **Academic transparency**: All specifications and sensitivity reported

## Reproducibility Architecture

### Technical Implementation
- **Full source code**: Open-source Python with comprehensive test suite
- **Computational reproducibility**: Fixed seeds, version tracking, runtime documentation
- **Academic standards**: Peer-reviewable algorithms, transparent assumptions

### Data and Methods Transparency
- **Manifest system**: JSON artifacts capturing full experimental provenance
- **Statistical rigor**: Confidence intervals, hypothesis testing frameworks
- **Replication package**: Makefile for end-to-end pipeline reproduction

## Limitations and Scope

### Model Boundaries
- **Scope**: Methods contribution, not causal inference on realized returns
- **Assumptions**: Stationary priors, simplified lease modification treatment
- **Generalizability**: Hospitality sector focus with broader methodology applicability

### Future Extensions
- **Industry expansion**: Manufacturing, retail, technology sector applications
- **Dynamic modeling**: Time-varying covenants, macro feedback effects
- **Behavioral factors**: Management responses to covenant proximity

## Impact Statement

This framework enables:

1. **Academic researchers**: Reproducible LBO analysis with modern accounting standards
2. **Industry practitioners**: Quantified uncertainty in covenant modeling  
3. **Policy makers**: Evidence-based assessment of IFRS-16 impact on credit markets
4. **Students**: Open-source learning tool for advanced corporate finance methods

The methodological contribution addresses a gap between practitioner LBO models (proprietary, limited uncertainty quantification) and academic corporate finance (stylized, pre-IFRS-16). We provide a bridge: rigorous, reproducible, and practically relevant.
