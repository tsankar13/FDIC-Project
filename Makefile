.PHONY: install report visualize trends clean

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e .

report:
	fdic-ml --generate-synopsis-report --output-dir backend/artifacts

visualize:
	fdic-ml --generate-visualizations --output-dir backend/artifacts

trends:
	fdic-ml --generate-trends --trend-quarters 8 --output-dir backend/artifacts

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
