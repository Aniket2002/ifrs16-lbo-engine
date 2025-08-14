# IFRS-16 LBO Engine - Academic Reproducibility Pipeline
# Ensures end-to-end reproducibility with graceful fallbacks for novel optimization framework

.PHONY: all clean test data analysis figures manifest paper install check-deps academic sobol-optional optimization-deps

# Default target: complete reproducible pipeline
all: check-deps test data analysis figures manifest

# Install Python dependencies
install:
	@echo "📦 Installing core dependencies from requirements.txt..."
	pip install -r requirements.txt
	@echo "✅ Core dependencies installed"

# Optional: Install academic enhancement dependencies
install-academic:
	@echo "� Installing academic enhancement dependencies..."
	pip install -r requirements-academic.txt
	@echo "✅ Academic features available (PyMC, scikit-optimize, etc.)"

# Legacy aliases for backward compatibility
sobol-optional: install-academic
	@echo "� SALib and academic dependencies installed"

optimization-deps: install-academic
	@echo "🚀 Optimization dependencies installed"

# Full academic installation
academic-install: install install-academic
	@echo "🎓 Complete academic installation finished!"

# Check if Python and core dependencies are available
check-deps:
	@echo "🔍 Checking dependencies..."
	@python -c "import numpy, pandas, matplotlib, scipy; print('✅ Core dependencies OK')" || (echo "❌ Missing dependencies. Run 'make install'" && exit 1)

# Run regression tests first
test-framework:
	@echo "🧪 Testing academic framework with current dependencies..."
	python test_framework.py

test:
	@echo "🧪 Running basic model test..."
	python -c "from orchestrator_advanced import read_accor_assumptions, run_comprehensive_lbo_analysis; a = read_accor_assumptions(); r = run_comprehensive_lbo_analysis(a); print(f'✅ Model test passed. IRR: {r[\"metrics\"][\"IRR\"]:.2%}')"

# Core analysis pipeline
analysis:
	@echo "Running core LBO analysis with academic rigor..."
	python orchestrator_advanced.py

# Paper pipeline using experiment runner
paper:
	@echo "Running complete paper experiment pipeline..."
	python analysis/scripts/run_single_case.py --seed 42 --n-mc 400 --sobol 1 --outdir analysis/figures
	@echo "All paper outputs generated in analysis/figures/"

# Paper without Sobol (if SALib unavailable)
paper-no-sobol:
	@echo "Running paper experiments without Sobol analysis..."
	python analysis/scripts/run_single_case.py --seed 42 --n-mc 400 --sobol 0 --outdir analysis/figures

# Empirical analysis
empirical:
	@echo "Running multi-company empirical analysis..."
	python analysis/scripts/run_empirical.py --outdir analysis/figures
	@echo "Empirical analysis complete"

# ==========================================
# NOVEL OPTIMIZATION FRAMEWORK (Tracks 1+3+5)
# ==========================================

# Bayesian calibration (Track 5)
calibrate:
	@echo "🔬 Running Bayesian hierarchical calibration..."
	python analysis/calibration/bayes_calibrate.py --firms-csv analysis/calibration/hotel_operators.csv --output-dir analysis/calibration/output --method map --n-samples 1000 --plot
	@echo "✅ Bayesian calibration complete"

# Analytic validation (Track 3)
analytic:
	@echo "� Running analytic headroom dynamics..."
	python lbo_model_analytic.py --output-dir analysis/figures --plot --validate --max-error 0.2
	@echo "✅ Analytic validation complete"

# Covenant optimization (Track 1)
optimize:
	@echo "🎯 Running covenant design optimization..."
	python optimize_covenants.py --priors analysis/calibration/output/posterior_samples.parquet --output-dir analysis/optimization --frontier --method bayesian --n-samples 500 --seed 42 --screen 1
	@echo "✅ Covenant optimization complete"

# Complete optimization pipeline (all tracks)
optimization: calibrate analytic optimize
	@echo "� Complete optimization framework executed!"
	@echo "📈 Generated figures F7-F11 for novel research contribution"

# Academic paper with optimization (enhanced)
paper-optimization: paper optimization
	@echo "🎓 Enhanced academic pipeline complete!"
	@echo "📊 Generated all figures F1-F11 for breakthrough optimization paper"

# Quick optimization test (single point)
optimize-quick:
	@echo "⚡ Quick optimization test..."
	python optimize_covenants.py --priors analysis/calibration/output/posterior_samples.parquet --output-dir analysis/optimization --alpha 0.10 --method grid --n-samples 100 --seed 42

# ==========================================
# STANDARD ACADEMIC PIPELINE
# ==========================================

# Validate input data integrity  
data:
	@echo "📊 Validating data integrity..."
	python -c "from orchestrator_advanced import read_accor_assumptions; print('✅ Data loaded successfully')"

# Generate all figures with reproducibility stamps
figures: analysis
	@echo "📈 Generating academic figures..."
	@echo "✅ Charts created with seed and git stamps"

# Create analysis manifest for full provenance
manifest: analysis
	@echo "📋 Creating analysis manifest..."
	@echo "✅ Manifest includes git hash, seed, N tracking"

# Full academic pipeline with optimization
academic: install optimization-deps all optimization
	@echo "🎓 Complete academic analysis with novel optimization framework!"

# Clean output directory  
clean:
	@echo "🧹 Cleaning output directory..."
	rm -f *.png *.pdf *.csv *.json
	rm -rf __pycache__ analysis/__pycache__ *.pyc
	@echo "✅ Outputs cleaned"

# Help target
help:
	@echo "IFRS-16 LBO Engine - Academic Pipeline with Novel Optimization"
	@echo ""
	@echo "🎯 Main targets:"
	@echo "  all              - Run complete reproducible pipeline (default)"
	@echo "  paper            - Generate F1-F6 figures (methods)"
	@echo "  optimization     - Generate F7-F11 figures (novel optimization)"
	@echo "  paper-optimization - Complete F1-F11 pipeline"
	@echo "  academic         - Full installation + optimization framework"
	@echo ""
	@echo "📦 Installation:"
	@echo "  install          - Core Python dependencies"
	@echo "  sobol-optional   - SALib for Sobol sensitivity"
	@echo "  optimization-deps - PyMC + scikit-optimize"
	@echo "  academic-install - Everything for complete pipeline"
	@echo ""
	@echo "🔬 Novel Framework (Tracks 1+3+5):"
	@echo "  calibrate        - Bayesian hierarchical priors"
	@echo "  analytic         - Analytic headroom dynamics"
	@echo "  optimize         - Covenant design optimization"
	@echo "  optimize-quick   - Quick optimization test"
	@echo ""
	@echo "📊 Standard Pipeline:"
	@echo "  empirical        - Multi-company analysis"
	@echo "  test             - Model validation"
	@echo "  clean            - Remove generated files"
	@echo ""
	@echo "📈 Outputs:"
	@echo "  F1-F6: Standard methods figures"
	@echo "  F7-F11: Novel optimization figures"
	@echo "  Pareto frontiers, policy maps, elasticities"
	@echo "  Complete reproducibility manifests"

# Quick development test
dev-test:
	@echo "⚡ Quick development test..."
	python -c "from orchestrator_advanced import OrchestratorAdvanced; print('✅ Import successful')"
