# -*- coding: utf-8 -*-
"""Segmenter latency and mask agreement: CPU (training-time path) vs GPU (served path).

The regional feature caches were built with the PSPNet lung mask on CPU. Serving moved the
segmenter to the GPU; this script shows the masks agree and reports the end-to-end latency.

    py -3 scripts/p2_41_segmenter_latency.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd   # before torch: pyarrow and torch DLLs clash on Windows if pandas loads second

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from p2_cls import pipeline_v2 as P2                                 # noqa: E402

OUT = ROOT / "artifacts/p2/segmenter_latency.json"


def pick_images():
    c01 = pd.read_csv(ROOT / "artifacts/p2/dryrun/provisional_c01.csv")
    dev = c01[c01.split == "development"]
    ids = list(dev[dev.category == "active_tb"].image_id[:4]) + list(dev[dev.category == "healthy"].image_id[:4])
    paths = [ROOT / "data/processed/v1/tbx11k/img512" / (i + ".png") for i in ids]
    sp = json.loads((ROOT / "artifacts/p2/external/vietnam_tune_report_split.json").read_text(encoding="utf-8"))
    paths += [ROOT / "data/processed/v1/vietnam_img512" / (str(i) + ".png") for i in sp["tune"][:8]]
    return paths


def main():
    import torch
    from p2_cls.pipeline import standardise
    paths = pick_images()
    imgs = [standardise(p)[0] for p in paths]
    assert all(im is not None for im in imgs)

    cpu = {"meta": {}, "_segmenter": None, "_segmenter_cpu": True}
    gpu = {"meta": {}, "_segmenter": None}
    seg_cpu = P2._segmenter(cpu)
    seg_gpu = P2._segmenter(gpu)
    print("devices: cpu path %s | served path %s" % (cpu["_segmenter_device"], gpu["_segmenter_device"]))

    # warm-up
    P2._lung_mask(seg_cpu, imgs[0], "cpu")
    P2._lung_mask(seg_gpu, imgs[0], gpu["_segmenter_device"])
    agree, iou, t_cpu, t_gpu = [], [], [], []
    for im in imgs:
        t = time.perf_counter(); mc = P2._lung_mask(seg_cpu, im, "cpu").astype(bool); t_cpu.append(time.perf_counter() - t)
        if gpu["_segmenter_device"] == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter(); mg = P2._lung_mask(seg_gpu, im, gpu["_segmenter_device"]).astype(bool)
        if gpu["_segmenter_device"] == "cuda":
            torch.cuda.synchronize()
        t_gpu.append(time.perf_counter() - t)
        agree.append(float((mc == mg).mean()))
        iou.append(float((mc & mg).sum() / max((mc | mg).sum(), 1)))
    print("mask agreement over %d images: min %.5f mean %.5f | IoU min %.5f" % (len(imgs), min(agree), np.mean(agree), min(iou)))
    print("segmenter ms/image: cpu %.0f | served %.0f" % (1000 * np.median(t_cpu), 1000 * np.median(t_gpu)))

    # end-to-end on the v3 candidate bundle (served path), warm
    b = P2.load_bundle_v2(ROOT / "artifacts/p2/bundle_v3_candidate")
    P2.run_case_v2(paths[0], b)
    lat = []
    for p in paths:
        t = time.perf_counter(); r = P2.run_case_v2(p, b); lat.append(time.perf_counter() - t)
        assert r["status"] == "ANALYSED", r
    print("end-to-end ms/image (v3, served path, warm): median %.0f | p90 %.0f" % (
        1000 * np.median(lat), 1000 * np.percentile(lat, 90)))
    rec = {"n_images": len(imgs), "segmenter_device_served": gpu["_segmenter_device"],
           "mask_agreement_min": round(min(agree), 6), "mask_agreement_mean": round(float(np.mean(agree)), 6),
           "mask_iou_min": round(min(iou), 6),
           "segmenter_ms_cpu_median": round(1000 * float(np.median(t_cpu)), 1),
           "segmenter_ms_served_median": round(1000 * float(np.median(t_gpu)), 1),
           "end_to_end_ms_median": round(1000 * float(np.median(lat)), 1),
           "end_to_end_ms_p90": round(1000 * float(np.percentile(lat, 90)), 1),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    OUT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
