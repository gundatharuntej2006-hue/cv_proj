# -*- coding: utf-8 -*-
"""Derive the TBX11K image-level label table from the shipped COCO annotations.

Standalone and stdlib-only, so it can be run against any TBX11K checkout without this
project's code. Writes the label CSV and a validation report, and exits non-zero if any
check fails.

    py -3 p1_make_tbx11k_labels.py --tbx11k /path/to/TBX11K
    py -3 p1_make_tbx11k_labels.py --tbx11k D:/data/TBX11K --out labels.csv

Where the labels come from, so this can be checked rather than trusted:

  * **Three-class diagnosis** is the folder prefix in each COCO `file_name` record:
    `health/` -> healthy, `sick/` -> sick_non_tb, `tb/` -> tuberculosis. Every one of the
    8,400 core trainval images carries one, with no gaps.

  * **TB subtype** is the set of lesion-box categories the dataset itself declares:
    1 = ActiveTuberculosis, 2 = ObsoletePulmonaryTuberculosis (latent), 3 =
    PulmonaryTuberculosis (unspecified). All boxes category 1 gives active, all category 2
    gives latent, a mix gives active_latent. Category 3 is ignored when a more specific
    category is present on the same image.

  * **One image, tb1199, carries no lesion box at all.** Deriving from boxes gives train
    counts of 473 active / 103 latent / 23 mixed; the published counts are 473 / 104 / 23,
    which total 600 - every TB image in the train split. So tb1199 can only be the 104th
    latent case. It is labelled latent_tb with label_basis=deduced_from_published_counts so
    the deduction stays visible instead of blending in with the 8,399 read directly.

Scope: the **core** lists (TBX11K_train/val, 8,400 images). The all_*.txt lists reference a
further 576 images under imgs/extra/ (Shenzhen, Montgomery, DA, DB) which are not part of
the benchmark and are deliberately excluded.
"""
import argparse
import collections
import csv
import json
import os
import sys

BOX_CAT = {1: "active", 2: "latent", 3: "unspecified"}
FOLDER = {"health": "healthy", "sick": "sick_non_tb", "tb": "tb"}
SUBTYPE = {"active": "active_tb", "latent": "latent_tb",
           "active+latent": "active_latent_tb"}
PUBLISHED_TRAIN = {"active_tb": 473, "latent_tb": 104, "active_latent_tb": 23}
EXPECTED_TOTALS = {"healthy": 3800, "sick_non_tb": 3800}
NO_BOX_FALLBACK = "latent_tb"


def derive(tbx11k):
    rows, problems = [], []
    for stem, split in (("TBX11K_train", "train"), ("TBX11K_val", "val")):
        p = os.path.join(tbx11k, "annotations", "json", stem + ".json")
        if not os.path.exists(p):
            problems.append("missing annotation file: %s" % p)
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        boxes = collections.defaultdict(set)
        for a in d.get("annotations", []):
            boxes[a["image_id"]].add(BOX_CAT.get(a["category_id"], "unknown"))
        for im in d["images"]:
            folder, fname = im["file_name"].split("/", 1)
            if folder not in FOLDER:
                problems.append("unexpected folder %r in %s" % (folder, im["file_name"]))
                continue
            base = FOLDER[folder]
            if base != "tb":
                cat, basis = base, "folder"
            else:
                s = boxes.get(im["id"])
                if not s:
                    cat, basis = NO_BOX_FALLBACK, "deduced_from_published_counts"
                else:
                    s = (s - {"unspecified"}) or s
                    key = "+".join(sorted(s))
                    if key not in SUBTYPE:
                        problems.append("unexpected box category set %r on %s"
                                        % (key, im["file_name"]))
                        continue
                    cat, basis = SUBTYPE[key], "lesion_box_categories"
            rows.append({"image_id": fname.rsplit(".", 1)[0], "official_split": split,
                         "file_name": im["file_name"], "category": cat,
                         "label_basis": basis})
    return rows, problems


def validate(rows, tbx11k, problems, check_files=True):
    """Every check is a line in the report; any failure makes the run exit non-zero."""
    out = []
    state = {"ok": True}

    def check(name, passed, detail=""):
        state["ok"] = state["ok"] and bool(passed)
        out.append("[%s] %-46s %s" % ("PASS" if passed else "FAIL", name, detail))

    check("annotation files parsed without problems", not problems,
          "; ".join(problems[:3]) if problems else "")
    check("row count is 8400", len(rows) == 8400, "got %d" % len(rows))

    ids = [r["image_id"] for r in rows]
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    check("image_id is unique", not dupes,
          ("duplicates: %s" % dupes[:5]) if dupes else "")

    by_split = collections.Counter(r["official_split"] for r in rows)
    check("official split sizes are 6600 / 1800",
          by_split.get("train") == 6600 and by_split.get("val") == 1800,
          str(dict(by_split)))

    cats = collections.Counter(r["category"] for r in rows)
    for k, v in EXPECTED_TOTALS.items():
        check("trainval %s == %d" % (k, v), cats.get(k) == v, "got %d" % cats.get(k, 0))
    tb_total = sum(cats.get(k, 0) for k in SUBTYPE.values())
    check("trainval tuberculosis == 800", tb_total == 800, "got %d" % tb_total)

    tr = collections.Counter(r["category"] for r in rows if r["official_split"] == "train")
    for k, v in PUBLISHED_TRAIN.items():
        check("train %s matches the paper (%d)" % (k, v), tr.get(k) == v,
              "got %d" % tr.get(k, 0))

    deduced = [r["image_id"] for r in rows
               if r["label_basis"] == "deduced_from_published_counts"]
    check("exactly one label is deduced, and it is tb1199",
          deduced == ["tb1199"], "got %s" % deduced)

    if check_files:
        missing = [r["file_name"] for r in rows
                   if not os.path.exists(os.path.join(tbx11k, "imgs", r["file_name"]))]
        check("every labelled image exists on disk", not missing,
              ("%d missing, e.g. %s" % (len(missing), missing[:3])) if missing else "")

    # cross-check against the official list files, which are the split definition
    for stem, split in (("TBX11K_train", "train"), ("TBX11K_val", "val")):
        lp = os.path.join(tbx11k, "lists", stem + ".txt")
        if not os.path.exists(lp):
            check("list file %s.txt present" % stem, False, "not found")
            continue
        with open(lp, encoding="utf-8") as fh:
            listed = set()
            for line in fh:
                line = line.strip()
                if line:
                    listed.add(line.split("/")[-1].rsplit(".", 1)[0])
        got = set(r["image_id"] for r in rows if r["official_split"] == split)
        check("%s matches %s.txt exactly" % (split, stem), listed == got,
              "only in list: %d, only in csv: %d" % (len(listed - got), len(got - listed)))

    out.append("")
    out.append("category totals (trainval): %s" % dict(sorted(cats.items())))
    out.append("train totals:               %s" % dict(sorted(tr.items())))
    out.append("label_basis:                %s"
               % dict(collections.Counter(r["label_basis"] for r in rows)))
    return state["ok"], out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tbx11k", required=True, help="path to the TBX11K directory")
    ap.add_argument("--out", default="tbx11k_image_labels_v1.csv")
    ap.add_argument("--report", default="tbx11k_labels_validation.txt")
    ap.add_argument("--skip-file-check", action="store_true",
                    help="do not verify that each labelled image exists on disk")
    args = ap.parse_args()

    rows, problems = derive(args.tbx11k)
    rows.sort(key=lambda r: r["image_id"])
    ok, report = validate(rows, args.tbx11k, problems, check_files=not args.skip_file_check)

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "official_split", "file_name",
                                           "category", "label_basis"])
        w.writeheader()
        w.writerows(rows)

    header = ["TBX11K image-level label validation",
              "source: %s" % os.path.abspath(args.tbx11k),
              "output: %s (%d rows)" % (os.path.abspath(args.out), len(rows)), ""]
    text = "\n".join(header + report) + "\n"
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)
    sys.stdout.write("\n%s\n" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
