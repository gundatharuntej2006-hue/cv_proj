# -*- coding: utf-8 -*-
"""Phase 2: can our preregistered concepts be mapped onto CLEAR's vocabulary?

CLEAR does not ship a curated finding list. `mimic_concepts.csv` is an ordered vocabulary
of 368,294 free-text observations extracted from radiology reports — whole sentences, not
labels. So "fibrosis" is not one output; it is an unknown number of sentences, an unknown
share of which are NEGATED ("no evidence of fibrosis").

This script quantifies that, per preregistered concept, so the mapping decision rests on
measurement rather than assumption. It downloads nothing and trains nothing: it reads the
vocabulary already cached outside the package.
"""
import csv
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
VOCAB = Path(r"D:\hf-cache\clear_vocab\mimic_concepts.csv")

# Preregistered in kaggle_concept_raydino_20261021_1830/kaggle/01_concept_audit.ipynb,
# fixed before any association could be computed. Not edited here.
PREREGISTERED = [
    "fibrosis", "scarring", "volume loss", "calcified granuloma", "calcification",
    "cavity", "focal opacity", "consolidation", "pleural thickening", "nodule",
    "reticular opacity", "bronchiectasis",
]

# Negation and uncertainty cues. A vocabulary entry carrying one of these asserts the
# ABSENCE or DOUBT of the finding, so it must never be pooled with assertions of it.
NEG = re.compile(
    r"\b(no|not|without|free of|absence of|absent|negative for|resolved|"
    r"clear(?:ed)? of|unremarkable)\b", re.I)
UNC = re.compile(
    r"\b(may|might|could|possible|possibly|probable|probably|suspect\w*|"
    r"question\w*|equivocal|cannot be excluded|concerning for|suggest\w*)\b", re.I)


def main():
    rows = list(csv.DictReader(io.open(VOCAB, encoding="utf-8")))
    vocab = [r["concept"] for r in rows]
    print("vocabulary entries: %d" % len(vocab))

    out = {"vocabulary": {"file": "mimic_concepts.csv",
                          "hf_repo": "peterhan91/CLEAR",
                          "pinned_revision": "142f169c8b006b0f28cf55c6a237c3003db5ed8f",
                          "n_entries": len(vocab),
                          "kind": "free-text report-derived observations (sentences), "
                                  "NOT a curated finding label set",
                          "examples": vocab[:3]},
           "preregistered_concepts": PREREGISTERED, "mapping": []}

    for term in PREREGISTERED:
        pat = re.compile(r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\w*", re.I)
        hits = [v for v in vocab if pat.search(v)]
        neg = [v for v in hits if NEG.search(v)]
        unc = [v for v in hits if UNC.search(v) and not NEG.search(v)]
        aff = [v for v in hits if not NEG.search(v) and not UNC.search(v)]
        exact = [v for v in hits if v.strip().lower() == term.lower()]
        out["mapping"].append({
            "concept": term,
            "n_matching_entries": len(hits),
            "n_affirmative": len(aff),
            "n_negated": len(neg),
            "n_uncertain": len(unc),
            "pct_negated": round(100.0 * len(neg) / len(hits), 1) if hits else None,
            "exact_single_output_exists": bool(exact),
            "mapping_type": ("NOT PRESENT" if not hits else
                             "EXACT" if exact else "PROXY - aggregate over many entries"),
            "example_affirmative": aff[:2],
            "example_negated": neg[:2],
        })

    print("\n%-20s %8s %8s %8s %8s  %-8s %s"
          % ("concept", "matches", "affirm", "negated", "%neg", "exact?", "mapping"))
    for m in out["mapping"]:
        print("%-20s %8d %8d %8d %7s%%  %-8s %s"
              % (m["concept"], m["n_matching_entries"], m["n_affirmative"],
                 m["n_negated"], m["pct_negated"], m["exact_single_output_exists"],
                 m["mapping_type"]))

    n_absent = sum(1 for m in out["mapping"] if m["n_matching_entries"] == 0)
    n_exact = sum(1 for m in out["mapping"] if m["exact_single_output_exists"])
    out["summary"] = {
        "concepts_absent_from_vocabulary": n_absent,
        "concepts_with_an_exact_single_output": n_exact,
        "concepts_requiring_proxy_aggregation": len(PREREGISTERED) - n_absent - n_exact,
        "conclusion": (
            "CLEAR exposes a similarity over 368,294 free-text observations, not a "
            "curated finding vocabulary. No preregistered concept is a single model "
            "output. Every concept would have to be constructed by aggregating over "
            "many matching sentences, a large share of which are NEGATED assertions of "
            "the same finding. That aggregation rule is a modelling choice this project "
            "has not preregistered, it is not specified by CLEAR's documentation, and "
            "choosing it after seeing associations would be exactly the post-hoc "
            "selection the protocol forbids."),
    }
    json.dump(out, io.open(RUN / "tables/concept_mapping.json", "w", encoding="utf-8"),
              indent=2)

    with io.open(RUN / "tables/concept_mapping.csv", "w", encoding="utf-8",
                 newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "n_matching_entries", "n_affirmative", "n_negated",
                    "n_uncertain", "pct_negated", "exact_single_output_exists",
                    "mapping_type"])
        for m in out["mapping"]:
            w.writerow([m["concept"], m["n_matching_entries"], m["n_affirmative"],
                        m["n_negated"], m["n_uncertain"], m["pct_negated"],
                        m["exact_single_output_exists"], m["mapping_type"]])

    print("\nabsent: %d | exact single output: %d | proxy aggregation required: %d"
          % (n_absent, n_exact, len(PREREGISTERED) - n_absent - n_exact))


if __name__ == "__main__":
    main()
