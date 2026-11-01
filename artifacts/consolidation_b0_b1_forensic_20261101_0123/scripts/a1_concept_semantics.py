# -*- coding: utf-8 -*-
"""Workstream A: what does the pinned CLEAR "consolidation" score actually measure?

Text-side only. No images, no classifier scores, no patient data of any kind are read.
Everything here is a property of the pinned concept vocabulary and the pinned concept
embedding tensor, so it is reproducible by anyone holding those two assets.

    py -3 a1_concept_semantics.py
"""
import csv
import json
import random
import re
from pathlib import Path

import torch

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
TAB = RUN / "tables"
REV = "142f169c8b006b0f28cf55c6a237c3003db5ed8f"
EMB = Path(r"D:\hf-cache\hub\models--peterhan91--CLEAR\snapshots") / REV / "concept_embeddings_368294.pt"
VOCAB = Path(r"D:\hf-cache\clear_vocab\mimic_concepts.csv")
SEED = 20261101

# The nine primary concepts plus the two negative controls, exactly as pinned.
AUDIT = {"fibrosis": 835, "scarring": 96952, "volume loss": 368164,
         "calcified granuloma": 209383, "calcification": 77874, "focal opacity": 281312,
         "consolidation": 123742, "pleural thickening": 341471, "bronchiectasis": 57870,
         "scoliosis": 82734, "degenerative changes": 339316}
CONTROLS = ("scoliosis", "degenerative changes")

NEG_RE = re.compile(r"\b(no|without|absent|negative for|free of|clear of|resolved|"
                    r"ruled out|not|never|unremarkable)\b", re.I)


def load():
    rows = list(csv.DictReader(open(VOCAB, encoding="utf-8")))
    by_idx = {int(r["concept_idx"]): r["concept"] for r in rows}
    by_txt = {}
    for r in rows:
        by_txt.setdefault(r["concept"].strip().lower(), []).append(int(r["concept_idx"]))
    return rows, by_idx, by_txt


def main():
    rows, by_idx, by_txt = load()
    C = torch.load(EMB, map_location="cpu", mmap=True, weights_only=True)
    assert C.shape == (368294, 768), C.shape

    def e(i):
        v = C[i].float()
        return v / v.norm().clamp_min(1e-12)

    out = {"vocabulary_rows": len(rows), "embedding_shape": list(C.shape),
           "embedding_dtype": str(C.dtype), "revision": REV, "seed": SEED}

    # ---- A1: exact identity of every audited concept -----------------------------------
    mapping = []
    for name, idx in AUDIT.items():
        txt = by_idx[idx]
        dupes = by_txt[txt.strip().lower()]
        mapping.append({
            "concept_label_used_in_reports": name,
            "exact_vocabulary_phrase": txt,
            "phrase_is_label_verbatim": txt.strip().lower() == name,
            "concept_idx": idx,
            "csv_row_index_0based": idx,
            "n_vocabulary_entries_with_this_exact_text": len(dupes),
            "role": "negative control" if name in CONTROLS else "primary",
            "representation": "single phrase; no pooling, expansion or template",
            "embedding_row": idx,
            "embedding_l2_norm_raw": round(float(C[idx].float().norm()), 6),
        })
    with open(TAB / "concept_mapping.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mapping[0].keys()))
        w.writeheader()
        w.writerows(mapping)

    # ---- how often does the bare word appear inside longer, negated sentences? ---------
    stems = {"consolidation": "consolidat", "scarring": "scar", "fibrosis": "fibro",
             "focal opacity": "opacit", "volume loss": "volume loss",
             "calcification": "calcif", "bronchiectasis": "bronchiectas",
             "pleural thickening": "pleural thicken", "scoliosis": "scolios",
             "calcified granuloma": "granulom", "degenerative changes": "degenerat"}
    ctx = []
    for name, stem in stems.items():
        hits = [r["concept"] for r in rows if stem in r["concept"].lower()]
        neg = [h for h in hits if NEG_RE.search(h)]
        ctx.append({"concept": name, "stem": stem,
                    "vocabulary_entries_containing_stem": len(hits),
                    "of_which_contain_a_negation_cue": len(neg),
                    "negated_fraction": round(len(neg) / max(len(hits), 1), 4)})
    with open(TAB / "concept_context_frequency.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ctx[0].keys()))
        w.writeheader()
        w.writerows(ctx)
    out["context_frequency"] = {c["concept"]: c["negated_fraction"] for c in ctx}

    # ---- A3: concept-to-concept text geometry -------------------------------------------
    names = list(AUDIT)
    E = torch.stack([e(AUDIT[n]) for n in names])
    M = (E @ E.T).numpy()
    with open(TAB / "concept_similarity_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["concept"] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [round(float(M[i, j]), 4) for j in range(len(names))])

    ci = names.index("consolidation")
    near = sorted(((float(M[ci, j]), names[j]) for j in range(len(names)) if j != ci),
                  reverse=True)
    out["consolidation_nearest_audited_concepts"] = [
        {"concept": n, "cosine": round(c, 4)} for c, n in near]

    # ---- A4: the negation test -----------------------------------------------------------
    # Does the text encoder separate "X" from "no X"? Built only from exact vocabulary
    # entries that already exist; no phrase is invented.
    pairs = []
    for name, idx in AUDIT.items():
        for pat in ("no %s", "no focal %s", "without %s", "%s resolved", "%s has resolved"):
            q = pat % name
            if q in by_txt:
                pairs.append({"positive_phrase": name, "positive_idx": idx,
                              "negated_phrase": q, "negated_idx": by_txt[q][0],
                              "cosine_positive_vs_negated":
                                  round(float(e(idx) @ e(by_txt[q][0])), 4)})
    rng = random.Random(SEED)
    samp = [rng.randrange(len(rows)) for _ in range(400)]
    nulls = []
    for name, idx in AUDIT.items():
        v = e(idx)
        cs = sorted(float(v @ e(j)) for j in samp)
        nulls.append({"concept": name,
                      "random_cosine_median": round(cs[len(cs) // 2], 4),
                      "random_cosine_p95": round(cs[int(0.95 * (len(cs) - 1))], 4),
                      "random_cosine_max": round(cs[-1], 4)})
    nullby = {n["concept"]: n for n in nulls}
    for p in pairs:
        n = nullby[p["positive_phrase"]]
        p["random_cosine_p95_for_this_concept"] = n["random_cosine_p95"]
        p["negation_above_random_p95"] = p["cosine_positive_vs_negated"] > n["random_cosine_p95"]
    with open(TAB / "negation_sensitivity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)
    with open(TAB / "random_cosine_reference.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nulls[0].keys()))
        w.writeheader()
        w.writerows(nulls)

    # ---- how the anchor sits among real sentences that mention consolidation -----------
    hits = [(int(r["concept_idx"]), r["concept"]) for r in rows
            if "consolidat" in r["concept"].lower()]
    rng2 = random.Random(SEED + 1)
    sub = rng2.sample(hits, 1500)
    anchor = e(123742)
    neg_c, pos_c = [], []
    for i, t in sub:
        c = float(anchor @ e(i))
        (neg_c if NEG_RE.search(t) else pos_c).append(c)

    def q(a, f):
        a = sorted(a)
        return round(a[int(f * (len(a) - 1))], 4)

    out["anchor_vs_sentences"] = {
        "n_sampled": len(sub),
        "affirmative": {"n": len(pos_c), "median": q(pos_c, .5), "p25": q(pos_c, .25),
                        "p75": q(pos_c, .75)},
        "negated": {"n": len(neg_c), "median": q(neg_c, .5), "p25": q(neg_c, .25),
                    "p75": q(neg_c, .75)},
        "median_gap_affirmative_minus_negated": round(q(pos_c, .5) - q(neg_c, .5), 4),
    }
    out["negation_pairs"] = pairs
    out["random_cosine_reference"] = nulls
    (TAB / "concept_semantics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("exact phrase for the consolidation concept: %r (idx 123742)" % by_idx[123742])
    print("\nnegation pairs that exist in the pinned vocabulary:")
    for p in pairs:
        print("  %-24s vs %-30s cos %+.4f   (random p95 %+.4f)%s"
              % (p["positive_phrase"], p["negated_phrase"],
                 p["cosine_positive_vs_negated"], p["random_cosine_p95_for_this_concept"],
                 "  <- far above chance" if p["negation_above_random_p95"] else ""))
    a = out["anchor_vs_sentences"]
    print("\nanchor vs %d real sentences mentioning consolidation:" % a["n_sampled"])
    print("  affirmative (n=%d) median cos %+.4f" % (a["affirmative"]["n"], a["affirmative"]["median"]))
    print("  negated     (n=%d) median cos %+.4f" % (a["negated"]["n"], a["negated"]["median"]))
    print("  gap: %+.4f" % a["median_gap_affirmative_minus_negated"])
    print("\nconsolidation's nearest audited concepts:")
    for r in out["consolidation_nearest_audited_concepts"][:5]:
        print("  %-24s %+.4f" % (r["concept"], r["cosine"]))


if __name__ == "__main__":
    main()
