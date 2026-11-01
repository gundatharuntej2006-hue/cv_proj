# Person 2 targets. Included by the root Makefile (P1 owns that file).
# HF_HOME points at the shared model cache so no project re-downloads RAD-DINO.
export HF_HOME ?= D:/hf-cache

.PHONY: p2-01 mocks-p2 test-p2 p2-report

p2-01:            ## P2-01 VRAM feasibility, then the report and D15
	py -3 scripts/p2_01_vram_feasibility.py
	py -3 scripts/p2_01_write_report.py

mocks-p2:         ## P2-02 regenerate every P2 mock (C03, C07, C08, C09, C17)
	py -3 scripts/p2_02_make_mocks.py

test-p2:          ## P2 test suite; add -m "not slow" to skip the real-weights test
	py -3 -m pytest tests/p2_cls tests/p2_gate -q

p2-report: mocks-p2 test-p2
