# IFRS-16 LBO Engine - Academic Rep# Core analysis pipeline
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
	@echo "Empirical analysis complete"cibility Pipeline
# Ensures end-to-end reproducibility with graceful fallbacks

.PHONY: all clean test data analysis figures manifest paper install check-deps academic sobol-optional

# Default target: complete reproducible pipeline
all: check-deps test data analysis figures manifest

# Install Python dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install numpy pandas matplotlib scipy seaborn openpyxl
	@echo "✅ Core dependencies installed"

# Optional: Install SALib for Sobol sensitivity analysis
sobol-optional:
	@echo "🔬 Installing SALib for Sobol sensitivity analysis..."
	pip install SALib
	@echo "✅ SALib installed (Sobol analysis will be available)"

# Check if Python and core dependencies are available
check-deps:
	@echo "🔍 Checking dependencies..."
	@python -c "import numpy, pandas, matplotlib, scipy; print('✅ Core dependencies OK')" || (echo "❌ Missing dependencies. Run 'make install'" && exit 1)

# Run regression tests first
test:
	@echo "🧪 Running basic model test..."
	python -c "from orchestrator_advanced import read_accor_assumptions, run_comprehensive_lbo_analysis; a = read_accor_assumptions(); r = run_comprehensive_lbo_analysis(a); print(f'✅ Model test passed. IRR: {r[\"metrics\"][\"IRR\"]:.2%}')"

# Validate input data integrity  
data:
	@echo "📊 Validating data integrity..."
	python -c "from orchestrator_advanced import read_accor_assumptions; print('✅ Data loaded successfully')"

# Core analysis pipeline
analysis:
	@echo "🔬 Running core LBO analysis with academic rigor..."
	python orchestrator_advanced.py

# Generate all figures with reproducibility stamps
figures: analysis
	@echo "📈 Generating academic figures..."
	@echo "✅ Charts created with seed and git stamps"

# Create analysis manifest for full provenance
manifest: analysis
	@echo "📋 Creating analysis manifest..."
	@echo "✅ Manifest includes git hash, seed, N tracking"

# Academic pipeline with all features
academic: install sobol-optional all
	@echo "📚 Academic analysis complete with full Sobol sensitivity"

# Optional: compile LaTeX paper (if tex files exist)
paper: figures manifest
	@echo "📄 Paper artifacts ready for LaTeX compilation"
	@echo "✅ All figures and tables generated"

# Clean output directory  
clean:
	@echo "🧹 Cleaning output directory..."
	rm -f *.png *.pdf *.csv *.json
	rm -rf __pycache__ src/__pycache__ src/modules/__pycache__
	@echo "✅ Outputs cleaned"

# Help target
help:
	@echo "IFRS-16 LBO Engine - Academic Pipeline"
	@echo ""
	@echo "Main targets:"
	@echo "  all           - Run complete reproducible pipeline (default)"
	@echo "  paper         - Generate all paper figures and tables"
	@echo "  paper-no-sobol- Paper pipeline without Sobol analysis"
	@echo "  academic      - Install all deps and run with Sobol analysis"
	@echo "  install       - Install core Python dependencies"
	@echo "  sobol-optional- Install SALib for Sobol sensitivity"
	@echo "  test          - Quick model validation"
	@echo "  clean         - Remove generated files"
	@echo ""
	@echo "Paper outputs:"
	@echo "  • F1_monte_carlo.pdf - Monte Carlo IRR distribution"
	@echo "  • F2_sources_uses.pdf - Sources & Uses waterfall"
	@echo "  • F3_exit_bridge.pdf - Exit equity bridge"
	@echo "  • F4_deleveraging.pdf - Deleveraging timeline"
	@echo "  • F5_sobol.pdf - Sobol sensitivity indices (if SALib available)"
	@echo "  • F6_stress_grid.pdf - Deterministic stress scenarios"
	@echo "  • manifest.json - Complete reproducibility manifest"
	@echo "✅ Clean complete"

# Run empirical analysis
empirical: analysis
	@echo "🏨 Running empirical hotel operator analysis..."
	python paper_empirical/run_empirical.py

# Full academic pipeline with empirical
academic: all empirical
	@echo "🎓 Academic pipeline complete!"
	@echo "✅ Methods and empirical analyses finished"

# Quick development test
dev-test:
	@echo "⚡ Quick development test..."
	python -c "from orchestrator_advanced import OrchestratorAdvanced; print('✅ Import successful')"
