# -*- coding: utf-8 -*-
"""Phase 0/1: partition inventory, target crosswalk, leakage inventory.

Read-only. Writes machine-readable tables into ../tables/.

    py -3 phase01_audit.py
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1]
TAB = RUN / "tables"

C01 = ROOT / "artifacts/p2/dryrun/provisional_c01.csv"
LAB = ROOT / "artifacts/p2/labels_tbx11k_trainval_v1.csv"
DUP = ROOT / "artifacts/p2/dedup/provisional_near_dup_official_v0.csv"

TB_CATS = ("active_tb", "latent_tb", "active_latent_tb")
ACTIVE_CATS = ("active_tb", "active_latent_tb")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    c01, lab = rows(C01), rows(LAB)
    labi = {r["image_id"]: r for r in lab}

    # --- the two target columns, derived independently of the stored columns -----------
    # tb_label      == category in {active, latent, active_latent}
    # active_tb_label == category in {active, active_latent}
    mism = []
    for r in c01:
        l = labi.get(r["image_id"])
        if l is None:
            mism.append((r["image_id"], "absent from the label table"))
            continue
        if l["tbx11k_category"] != r["category"]:
            mism.append((r["image_id"], "category differs: %s vs %s"
                         % (l["tbx11k_category"], r["category"])))
        if l["tb_label"] != r["tb_label"]:
            mism.append((r["image_id"], "tb_label differs"))
        derived_tb = "1" if r["category"] in TB_CATS else "0"
        derived_act = "1" if r["category"] in ACTIVE_CATS else "0"
        if r["tb_label"] != derived_tb:
            mism.append((r["image_id"], "tb_label not derivable from category"))
        if l["active_tb_label"] != derived_act:
            mism.append((r["image_id"], "active_tb_label not derivable from category"))

    # --- target crosswalk --------------------------------------------------------------
    cw = collections.Counter()
    for r in c01:
        l = labi[r["image_id"]]
        cw[(r["split"], r["category"], l["tb_label"], l["active_tb_label"],
            r["active_tb_label"] or "EMPTY")] += 1
    with open(TAB / "target_crosswalk.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "tbx11k_category", "tb_label", "active_tb_label",
                    "active_tb_label_as_stored_in_C01", "n"])
        for k in sorted(cw):
            w.writerow(list(k) + [cw[k]])

    # --- partition table ----------------------------------------------------------------
    part = collections.OrderedDict()
    for r in c01:
        s = r["split"]
        d = part.setdefault(s, collections.Counter())
        d["n"] += 1
        d[r["category"]] += 1
        d["pos_tb_label"] += r["category"] in TB_CATS
        d["pos_active_tb_label"] += r["category"] in ACTIVE_CATS
    with open(TAB / "partition_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["n", "healthy", "sick_non_tb", "active_tb", "latent_tb",
                "active_latent_tb", "pos_tb_label", "pos_active_tb_label"]
        w.writerow(["split"] + cols + ["prevalence_tb_label", "prevalence_active_tb_label"])
        for s, d in part.items():
            w.writerow([s] + [d[c] for c in cols]
                       + ["%.4f" % (d["pos_tb_label"] / d["n"]),
                          "%.4f" % (d["pos_active_tb_label"] / d["n"])])

    # --- near-duplicate leakage, mapped onto the C01 partitions ------------------------
    sp = {r["image_id"]: r["split"] for r in c01}
    cat = {r["image_id"]: r["category"] for r in c01}
    pair_counts = collections.Counter()
    leaks = []
    for r in rows(DUP):
        a, b = r["image_id_a"], r["image_id_b"]
        sa, sb = sp.get(a), sp.get(b)
        if sa is None or sb is None:
            pair_counts["not_in_C01"] += 1
            continue
        key = "|".join(sorted((sa, sb)))
        pair_counts[key] += 1
        if sa != sb:
            leaks.append([a, sa, cat[a], b, sb, cat[b], r["cosine"]])
    with open(TAB / "near_duplicate_cross_split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id_a", "c01_split_a", "category_a",
                    "image_id_b", "c01_split_b", "category_b", "cosine"])
        w.writerows(sorted(leaks))

    affected_cal = sorted({a for a, sa, _, b, sb, _, _ in leaks if sa == "calibration"}
                          | {b for a, sa, _, b, sb, _, _ in leaks if sb == "calibration"})

    out = {
        "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     __import__("time").gmtime()),
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in (C01, LAB, DUP)},
        "label_table_rows": len(lab),
        "c01_rows": len(c01),
        "target_mapping_mismatches": mism,
        "target_mapping_deterministic": not mism,
        "c01_stores_active_tb_label_as_blank_for_latent": sum(
            1 for r in c01 if r["category"] == "latent_tb" and r["active_tb_label"] == ""),
        "near_duplicate_pairs_by_partition_pair": dict(pair_counts),
        "calibration_images_with_a_cross_split_near_duplicate": len(affected_cal),
        "calibration_affected_ids": affected_cal,
        "patient_ids_available": False,
        "patient_id_note": ("TBX11K ships no patient identifier; C01 and the label table "
                            "carry image_id only. Folds are image-level and cannot be "
                            "claimed patient-grouped."),
    }
    (TAB / "phase01_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("mismatches:", len(mism))
    print("partitions:", {s: d["n"] for s, d in part.items()})
    print("near-dup pairs:", dict(pair_counts))
    print("calibration images with a cross-split near-duplicate:", len(affected_cal))


if __name__ == "__main__":
    main()
