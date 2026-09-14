.PHONY: all env figures paper test render-v3 paper-v3 audit-v3

env:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo "conda activate ifrs16-lbo" 

all: figures paper

figures:
	python analysis/scripts/case_study_accor.py

paper:
	latexmk -pdf -cd -interaction=nonstopmode analysis/paper/main.tex

test:
	pytest -q

render-v3:
	python -m analysis.scripts.render_manuscript_v3

paper-v3:
	python -m analysis.scripts.build_manuscript_v3

audit-v3:
	python -m analysis.scripts.audit_manuscript_v3

# Legacy aliases for backward compatibility
install: env
academic: all
clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf *.pdf *.log *.aux *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
	rm -rf output/ analysis/figures/ .coverage
