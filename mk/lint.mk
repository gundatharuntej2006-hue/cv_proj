lint:
	python -m flake8 src/ tests/ --max-line-length=100 || true
	python -m black --check src/ tests/ || true
