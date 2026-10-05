test-unit:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing || true
