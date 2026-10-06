# Makefile fragment for Person 4 (Kunal - Statistics, Calibration, Triage, Pipeline, Governance)

.PHONY: mocks-p4 gate-check triage-fit eval-tables test-p4

mocks-p4:
	$(PYTHON) -m tbcore.mock_p4

gate-check:
	$(PYTHON) -m p4_eval.gate check

triage-fit:
	$(PYTHON) -c "from p4_stats.fit_triage import *; print('Triage fitting module ready')"

eval-tables:
	$(PYTHON) -c "from p4_eval.tables import export_evaluation_tables; export_evaluation_tables()"
	$(PYTHON) -c "from p4_eval.external import run_all_external_evaluations; run_all_external_evaluations()"
	$(PYTHON) -c "from p4_eval.comparisons import generate_table_14_comparisons; generate_table_14_comparisons()"

test-p4:
	$(PYTHON) -m pytest tests/test_tbcore.py tests/test_p4_pipeline_triage.py tests/test_contracts.py tests/test_ownership.py
