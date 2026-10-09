# P3 targets; included by the root Makefile (P1).
PYTHON ?= python
P3_DFINE := third_party/D-FINE
P3_DFINE_COMMIT := 956d1709314c2c6a4df6f34de232054578a7449f

.PHONY: p3-setup p3-test p3-feasibility p3-feasibility-smoke p3-mocks p3-train-512 p3-train-640 p3-train-smoke p3-d16-criterion p3-export-c10 p3-latency p3-latency-gpu p3-latency-cpu p3-latency-detect p3-verify

## p3-setup: fetch the pinned D-FINE submodule and install P3 dependencies
p3-setup:
	git submodule update --init $(P3_DFINE)
	@test "$$(git -C $(P3_DFINE) rev-parse HEAD)" = "$(P3_DFINE_COMMIT)" || \
		(echo "D-FINE is not at the pinned commit $(P3_DFINE_COMMIT)"; exit 1)
	$(PYTHON) -m pip install -r requirements/p3.txt

## p3-test: P3 tests (gpu-marked tests skipped)
p3-test:
	$(PYTHON) -m pytest tests/p3_* -m "not gpu" -q

## p3-feasibility: D-FINE T4 feasibility, 512 and 640 with AMP (run on Kaggle/Colab T4)
p3-feasibility:
	$(PYTHON) notebooks/p3/p3_01_dfine_feasibility.py $(P3_FEAS_ARGS)

## p3-feasibility-smoke: same script, tiny CPU run (plumbing check only)
p3-feasibility-smoke:
	$(PYTHON) notebooks/p3/p3_01_dfine_feasibility.py --smoke

## p3-mocks: regenerate P3 mock contracts under mock/ (C10, C11, C15) for consumers (TB_MOCK=1)
p3-mocks:
	$(PYTHON) mock/make_c10.py --split development
	$(PYTHON) mock/make_c10.py --split calibration
	$(PYTHON) mock/make_c11.py --split development
	$(PYTHON) mock/make_c15.py

## p3-train-512 / p3-train-640: D-FINE training on a T4 (auto-resume from Drive); P3-06
p3-train-512:
	PYTHONPATH=src $(PYTHON) -m p3_det.train --config dfine_512.yaml
p3-train-640:
	PYTHONPATH=src $(PYTHON) -m p3_det.train --config dfine_640.yaml

## p3-train-smoke: tiny CPU run of the training wrapper on FAKE P1 data
p3-train-smoke:
	PYTHONPATH=src $(PYTHON) -m p3_det.train --config dfine_512.yaml --smoke

## p3-d16-criterion: write the pre-registered D16 criterion (commit it before comparing)
p3-d16-criterion:
	PYTHONPATH=src $(PYTHON) -m p3_det.select_size criterion

## p3-export-c10: C10 detections for development + calibration with the locked detector (P3-08)
p3-export-c10:
	PYTHONPATH=src $(PYTHON) -m p3_det.export_c10

## p3-latency-detect: R3 detect latency on the local GPU (<= 500 ms/image target), real locked model (P3-08)
p3-latency-detect:
	PYTHONPATH=src $(PYTHON) -m pytest tests/p3_det/test_r3_detect.py -m gpu -s -q

## p3-latency: experiment E6 (P3-16), full pipeline, GPU then CPU-only (each in its own process)
p3-latency: p3-latency-gpu p3-latency-cpu

p3-latency-gpu:
	PYTHONPATH=src $(PYTHON) -m p3_ops.latency --mode gpu

p3-latency-cpu:
	PYTHONPATH=src $(PYTHON) -m p3_ops.latency --mode cpu

## p3-verify: audit all P3 work (tests/p3_verify) -> docs/p3_verification_report.md
p3-verify:
	PYTHONPATH=src $(PYTHON) -m p3_ops.verify_all
