# -*- coding: utf-8 -*-
"""Priority 2: scanner/class counts, data quality, split provenance.

Read-only. Writes three CSVs into the run directory and nowhere else.
"""
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]


def rd(p):
    return list(csv.DictReader(io.open(ROOT / p, encoding="utf-8-sig")))


def write(name, header, rows):
    with io.open(RUN / name, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote %s (%d rows)" % (name, len(rows)))


def scanner_counts():
    rows = []
    vn = rd("data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv")
    split = json.load(io.open(ROOT / "artifacts/p2/external/vietnam_tune_report_split.json",
                              encoding="utf-8"))
    tune = set(str(x) for x in split["tune"])
    rep = set(str(x) for x in split["report"])
    for rs in sorted({r["rad_system"] for r in vn}):
        sub = [r for r in vn if r["rad_system"] == rs]
        pos = sum(1 for r in sub if r["xpert_result"] == "1")
        nt = sum(1 for r in sub if str(r["pid"]) in tune)
        nr = sum(1 for r in sub if str(r["pid"]) in rep)
        rows.append(["Vietnam (Dryad)", "rad_system=%s" % rs, len(sub), pos, len(sub) - pos,
                     round(pos / len(sub), 4), 0, 0, nt, nr,
                     "Dryad metadata column rad_system"])
    pos = sum(1 for r in vn if r["xpert_result"] == "1")
    rows.append(["Vietnam (Dryad)", "ALL", len(vn), pos, len(vn) - pos, round(pos / len(vn), 4),
                 0, 0, len(tune), len(rep), "Dryad metadata"])

    tb = rd("artifacts/p2/labels_tbx11k_trainval_v1.csv")
    for sp in ("train", "val"):
        sub = [r for r in tb if r["official_split"] == sp]
        p = sum(1 for r in sub if r["tb_label"] == "1")
        rows.append(["TBX11K", "SOURCE NOT RECORDED (official_split=%s)" % sp, len(sub), p,
                     len(sub) - p, round(p / len(sub), 4), len(sub), 0, 0, 0,
                     "label CSV has no scanner/source column"])

    nlm = rd("artifacts/p2/audit/cross_dataset_case_predictions.csv")
    for coh in sorted({r["cohort"] for r in nlm}):
        sub = [r for r in nlm if r["cohort"] == coh]
        p = sum(1 for r in sub if r["label"] == "1")
        rows.append([coh, "single site (source == cohort)", len(sub), p, len(sub) - p,
                     round(p / len(sub), 4), 0, 0, 0, 0, "NLM distribution, one site per set"])

    dd = json.load(io.open(ROOT / "artifacts/p2/external/dadb_third_site.json", encoding="utf-8"))
    for r in dd["results"]:
        if r["variant"] != "full" or r["site"] == "DA+DB":
            continue
        rows.append(["DA/DB (TBX11K extra)", "site=%s" % r["site"], r["n"], r["n_tb"],
                     r["n"] - r["n_tb"], round(r["n_tb"] / r["n"], 4), 0, 0, 0, 0,
                     "site split recorded in dadb_third_site.json"])

    write("scanner_class_counts.csv",
          ["dataset", "source_group", "n", "n_positive", "n_negative", "prevalence",
           "n_missing_source", "n_missing_label", "n_in_TUNE", "n_in_REPORT", "source_evidence"],
          rows)


def data_quality():
    rows = []

    def chk(dataset, ids, labels, note):
        dup = len(ids) - len(set(ids))
        miss = sum(1 for x in labels if x is None or str(x).strip() == "")
        bad = sum(1 for x in labels if str(x).strip() not in ("0", "1"))
        rows.append([dataset, len(ids), len(set(ids)), dup, miss, bad,
                     "1 image per patient" if len(set(ids)) == len(ids) else "repeats present",
                     note])

    vn = rd("data/raw/vietnam/Lunit_Versions_-_Dryad_Data_Set.csv")
    chk("Vietnam (Dryad)", [r["pid"] for r in vn], [r["xpert_result"] for r in vn],
        "pid is the patient key; xpert_result is the Xpert Ultra reference standard")
    ga = rd("artifacts/p2/audit/vietnam_case_level_gate_audit.csv")
    chk("Vietnam case-level scores", [r["pid"] for r in ga], [r["xpert_result"] for r in ga],
        "case-level score file used for all Vietnam analyses in this run")
    tb = rd("artifacts/p2/labels_tbx11k_trainval_v1.csv")
    chk("TBX11K trainval", [r["image_id"] for r in tb], [r["tb_label"] for r in tb],
        "patient identifiers are not distributed with TBX11K; image_id is the only key")
    nlm = rd("artifacts/p2/audit/cross_dataset_case_predictions.csv")
    chk("Shenzhen+Montgomery", [r["image_id"] for r in nlm], [r["label"] for r in nlm],
        "NLM sets are one image per patient")

    a = {r["pid"]: r["xpert_result"] for r in vn}
    b = {r["pid"]: r["xpert_result"] for r in ga}
    mism = sum(1 for k in a if k in b and a[k] != b[k])
    rows.append(["Vietnam label agreement (Dryad vs score file)", len(a), len(b), 0, 0, mism,
                 "keys in both: %d" % len(set(a) & set(b)),
                 "label mismatches between the two files"])

    c = {r["pid"]: r["rad_system"] for r in vn}
    d = {r["pid"]: r["rad_system"] for r in ga}
    smis = sum(1 for k in c if k in d and c[k] != d[k])
    rows.append(["Vietnam rad_system agreement (Dryad vs score file)", len(c), len(d), 0, 0, smis,
                 "keys in both: %d" % len(set(c) & set(d)),
                 "scanner mismatches between the two files"])

    rows.append(["TBX11K label mapping", 8400, 8400, 0, 0, 0,
                 "tb_label=1 for folder tb (active_tb 631, latent_tb 139, active_latent 30)",
                 "healthy 3800 and sick_non_tb 3800 map to 0; the fitted head in all current "
                 "results uses healthy-only negatives, giving 3,241 training rows"])
    write("data_quality.csv",
          ["dataset_or_check", "n_rows", "n_unique_ids", "n_duplicate_ids", "n_missing_label",
           "n_malformed_or_mismatched", "patient_identifier_note", "notes"], rows)


def split_provenance():
    rows = [
        ["TBX11K", "train (official)", "FITTING ALLOWED", "used for every head fit",
         "yes - all heads", "labels_tbx11k_trainval_v1.csv, official_split=train (6,600 rows)"],
        ["TBX11K", "development", "SELECTION ALLOWED", "early stopping and variant selection",
         "yes - repeatedly", "659 cases used as dev in p2_33 and p2_36"],
        ["TBX11K", "internal_test", "RESERVED", "never read",
         "no", "recorded untouched in the vietnam_eval.json note"],
        ["Vietnam (Dryad)", "TUNE (1,352)", "DEVELOPMENT",
         "pooling grid and aggregator chosen here", "yes - the 18-config sweep",
         "vietnam_tune_report_split.json, seed 20261020"],
        ["Vietnam (Dryad)", "REPORT (1,356)", "RESERVED EVALUATION", "read once at the end",
         "no - read once", "mil_final.json; selection optimism -0.0005"],
        ["Shenzhen", "all 662", "EXPOSED",
         "informed preprocessing and negative-composition selection", "yes",
         "hard_negatives.json and learning_curve.json both score it"],
        ["Montgomery", "all 138", "EXPOSED",
         "informed preprocessing and negative-composition selection", "yes",
         "hard_negatives.json and learning_curve.json both score it"],
        ["DA/DB", "all 176", "EXTERNAL, scored once under global pooling",
         "not used for any choice", "no", "dadb_third_site.json"],
        ["CheXpert OOD pools", "lateral / non-medical", "OOD GATE CALIBRATION",
         "used to set the kNN OOD threshold", "yes",
         "radino_ood_lateral_v1.npz, radino_ood_nonmedical_v1.npz"],
    ]
    write("split_provenance.csv",
          ["dataset", "partition", "allowed_role", "how_it_was_actually_used",
           "previously_exposed_in_model_selection", "evidence"], rows)


if __name__ == "__main__":
    scanner_counts()
    data_quality()
    split_provenance()
