.PHONY: help install install-plot test lint validate validate-report clean build

help:
	@echo "psgscoring — make targets"
	@echo "  make install          Install package + test extras (editable)"
	@echo "  make install-plot     Install matplotlib for PDF reports"
	@echo "  make test             Run pytest"
	@echo "  make lint             Run ruff"
	@echo "  make validate         Run PSG-IPA validation (needs PSGIPA_DATA_DIR)"
	@echo "  make validate-report  Build PDF report from /tmp/validation_results.json"
	@echo "  make build            Build sdist + wheel"
	@echo "  make clean            Remove build / cache artefacts"

install:
	pip install -e ".[test]"

install-plot:
	pip install -e ".[plot]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check psgscoring tests

validate:
	@if [ -z "$$PSGIPA_DATA_DIR" ]; then \
		echo "PSGIPA_DATA_DIR not set. Download PSG-IPA from PhysioNet and export the path."; \
		exit 1; \
	fi
	python validate_psgipa.py --data-dir "$$PSGIPA_DATA_DIR" --workers 5

validate-report:
	python validation_report.py \
		--input-json /tmp/validation_results.json \
		--output-pdf validation_report.pdf

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
