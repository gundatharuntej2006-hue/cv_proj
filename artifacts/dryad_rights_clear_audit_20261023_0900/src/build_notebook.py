# -*- coding: utf-8 -*-
"""Generate the Kaggle notebook for the TUNE-only CLEAR concept audit."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "kaggle"
REV = "142f169c8b006b0f28cf55c6a237c3003db5ed8f"


def md(*l):
    return {"cell_type": "markdown", "metadata": {}, "source": [x + "\n" for x in l]}


def code(*l):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [x + "\n" for x in l]}


cells = [
    md("# CLEAR concept audit — Vietnam TUNE only", "",
       "**Diagnostic. Not expected to change AUROC, and no AUROC change may be claimed",
       "from it.** The reported screening AUROC remains 0.7242.", "",
       "| | |", "|---|---|",
       "| data licence | **CC0-1.0**, confirmed from the Dryad API for this exact DOI |",
       "| CLEAR | Apache-2.0, pinned revision `%s` |" % REV[:12],
       "| split | **Vietnam TUNE only.** REPORT and TBX11K `internal_test` are hard-blocked |",
       "| concepts | **9 exact** vocabulary entries; 3 excluded for lacking one |", "",
       "Cells run in order: licence confirmation, input validation, inference,",
       "aggregate analysis, report."),

    md("## 1 — Licence confirmation"),
    code("import json, sys, os, time, hashlib",
         "from pathlib import Path",
         "WORK = Path('/kaggle/working'); INP = Path('/kaggle/input')",
         "CODE = next((p for p in INP.glob('*/src/run_concept_audit.py')), None)",
         "assert CODE, 'attach the code dataset'",
         "sys.path.insert(0, str(CODE.parent))",
         "LIC = {'vietnam_images': 'CC0-1.0 (Dryad API, doi:10.5061/dryad.5tb2rbpcv)',",
         "       'clear_code': 'Apache-2.0', 'clear_weights': 'Apache-2.0',",
         "       'raydino': 'OUT OF SCOPE - gated, FAIR NC licence, not used here'}",
         "print(json.dumps(LIC, indent=2))",
         "assert 'CC0' in LIC['vietnam_images'], 'data licence not confirmed'",
         "print('\\nlicence gate: PASSED')"),

    md("## 2 — Input validation", "",
       "Hard-fails on REPORT or `internal_test`. Checks IDs, counts, missingness."),
    code("import run_concept_audit as A",
         "for p in INP.rglob('*internal*test*'):",
         "    raise SystemExit('attached input exposes the locked partition: %s' % p)",
         "d = A.load_tune_only()          # raises SplitViolation on any REPORT leak",
         "m, primary = A.concept_map()",
         "names = [c['concept'] for c in primary]",
         "import numpy as np",
         "print('TUNE patients      :', len(d['pid']))",
         "print('unique ids         :', len(set(d['pid'])))",
         "print('TB positive        :', int(d['y'].sum()))",
         "print('previous TB        :', int(d['past_tb'].sum()))",
         "print('concepts (exact)   :', len(names), names)",
         "print('excluded           :', [c['concept'] for c in m['mapping']",
         "                               if c['concept_idx'] is None])",
         "for k in ('y','past_tb','rad_system','clf_score'):",
         "    assert np.isfinite(np.asarray(d[k], float)).all(), k",
         "print('\\ninput gate: PASSED')"),

    md("## 3 — Inference", "",
       "Downloads the pinned CLEAR assets (~1.73 GB) and encodes 1,352 images at 448 px.",
       "Only the 9 pinned rows of the concept-embedding tensor are used; it is",
       "memory-mapped so the full 1.13 GB is never materialised. Halves the batch on OOM."),
    code("import torch, psutil",
         "v = psutil.virtual_memory()",
         "print('RAM available %.1f GB | VRAM free %.1f GB'",
         "      % (v.available/1e9,",
         "         torch.cuda.mem_get_info()[0]/1e9 if torch.cuda.is_available() else 0))",
         "if v.available < 3e9:",
         "    print('WARNING: under 3 GB of host RAM. Encoding may swap.')",
         "t0 = time.perf_counter()",
         "S = A.encode_concepts(d['pid'], device='cuda' if torch.cuda.is_available() else 'cpu')",
         "print('similarities', S.shape, ' %.1f min' % ((time.perf_counter()-t0)/60))",
         "assert S.shape == (len(d['pid']), len(names))",
         "assert np.isfinite(S).all()"),

    md("## 4 — Aggregate analysis", "",
       "Spearman rho between each concept similarity and the frozen classifier score, by",
       "group, with bootstrap CIs and **Benjamini-Hochberg** correction across concepts.",
       "Plus incremental R2 over scanner, age, sex, symptom count and past TB.", "",
       "**Association is not causation.** No concept score modifies any TB score."),
    code("rows, incr = A.analyse(d, S, names)",
         "import pandas as pd",
         "R = pd.DataFrame(rows); I = pd.DataFrame(incr)",
         "(WORK/'tables').mkdir(parents=True, exist_ok=True)",
         "R.to_csv(WORK/'tables/concept_associations.csv', index=False)",
         "I.to_csv(WORK/'tables/concept_incremental_r2.csv', index=False)",
         "print(R[R.group=='negatives (all)'][",
         "      ['concept','n','spearman_rho','ci95_lo','ci95_hi','p_bh_adjusted']",
         "      ].to_string(index=False))",
         "print()",
         "print(I.to_string(index=False))"),

    md("## 5 — Report", "",
       "Aggregate tables only. **No per-case concept score leaves the session.**"),
    code("sig = R[(R.group=='negatives (all)') & (R.p_bh_adjusted < 0.05)]",
         "print('concepts surviving BH at 0.05 among negatives: %d of %d'",
         "      % (len(sig), len(names)))",
         "print()",
         "print('Reminders for whoever writes this up:')",
         "print(' - a similarity to the sentence \"fibrosis\" is NOT a fibrosis detection')",
         "print(' - TUNE is exposed; this is exploratory, not validation')",
         "print(' - association is not causation; do not claim the classifier uses a cue')",
         "print(' - cavity, nodule and reticular opacity were EXCLUDED, not measured')",
         "print(' - the screening AUROC is unchanged at 0.7242')",
         "assert not (WORK/'tables/per_case.csv').exists(), 'no per-case table may be written'"),
]

OUT.mkdir(parents=True, exist_ok=True)
doc = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                   "language": "python"},
                    "language_info": {"name": "python"}, "accelerator": "GPU"}}
(OUT / "concept_audit_tune_only.ipynb").write_text(json.dumps(doc, indent=1),
                                                   encoding="utf-8")
print("wrote kaggle/concept_audit_tune_only.ipynb (%d cells)" % len(cells))
