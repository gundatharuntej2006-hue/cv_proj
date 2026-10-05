PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

.PHONY: all help install-base install-p1 install-p2 install-p3 install-p4 install-all test test-contracts verify-ownership run-app clean

all: help

help:
	@echo "Explainable TB Screening CAD System (cv_proj) - Build Commands"
	@echo "-------------------------------------------------------------"
	@echo "make install-base       : Install core tbcore requirements"
	@echo "make install-p1         : Install P1 dependencies (Data, App)"
	@echo "make install-p2         : Install P2 dependencies (Gate, RAD-DINO)"
	@echo "make install-p3         : Install P3 dependencies (D-FINE, XAI, PDF)"
	@echo "make install-p4         : Install P4 dependencies (Conformal Triage)"
	@echo "make install-all        : Install all project dependencies"
	@echo "make test               : Run pytest test suite"
	@echo "make test-contracts     : Validate contracts and mock pipeline packets"
	@echo "make verify-ownership   : Check branch and CODEOWNERS boundaries"
	@echo "make run-app            : Launch Streamlit Clinician Dashboard"
	@echo "make clean              : Remove caches and build artifacts"

install-base:
	$(PIP) install -r requirements/base.txt
	$(PIP) install -e . --no-deps

install-p1: install-base
	$(PIP) install -r requirements/p1.txt

install-p2: install-base
	$(PIP) install -r requirements/p2.txt

install-p3: install-base
	$(PIP) install -r requirements/p3.txt

install-p4: install-base
	$(PIP) install -r requirements/p4.txt

install-all: install-base
	$(PIP) install -r requirements/p1.txt -r requirements/p2.txt -r requirements/p3.txt -r requirements/p4.txt

test:
	$(PYTEST) tests/

test-contracts:
	$(PYTEST) tests/test_contracts.py

verify-ownership:
	$(PYTEST) tests/test_ownership.py

run-app:
	streamlit run src/p1_app/dashboard.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info
