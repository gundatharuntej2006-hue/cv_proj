# Local dry run

Everything here runs offline, downloads no weights, and opens no restricted data.

```bash
cd D:\tb-triage

# 1. syntax check every module and notebook builder
py -3 -m py_compile artifacts/kaggle_concept_raydino_20261021_1830/src/*.py
py -3 -m py_compile artifacts/kaggle_concept_raydino_20261021_1830/tests/*.py

# 2. full synthetic test suite: guards, shapes, ID alignment, target semantics,
#    thresholds, metrics, paired bootstrap, patient-level resampling, BH correction,
#    empty and single-class groups, notebook JSON validity, code-only check
py -3 -m pytest artifacts/kaggle_concept_raydino_20261021_1830/tests -q

# 3. regenerate the notebooks and confirm valid JSON
py -3 artifacts/kaggle_concept_raydino_20261021_1830/src/build_notebooks.py

# 4. the Stage 0 audit and the score-only Stage 1 analysis (reads committed artifacts
#    only; no model, no images, no internal_test)
py -3 artifacts/kaggle_concept_raydino_20261021_1830/src/stage0_audit.py
```

**Expected:** 29 tests pass, 4 notebooks regenerate as valid JSON, the audit prints the
input inventory and the TUNE-only subgroup table.

**What this does NOT do.** It does not encode an image, download a weight, run CLEAR or
RayDINO, or produce any benchmark number. **A passing local dry run is not a Kaggle run**
and must never be reported as one.
