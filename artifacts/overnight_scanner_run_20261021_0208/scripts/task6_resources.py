# -*- coding: utf-8 -*-
"""Task 6: matched timing and peak memory for the global and regional read-outs.

Scope, stated so the numbers are not over-read: this measures the cost of the *read-out*
over already-cached frozen features. It does **not** measure RAD-DINO inference, which is
identical for both arms and dominates end-to-end latency. Both arms are timed in the same
process, on the same 1,356 REPORT cases, with warm-up and repeats, so the comparison is
matched even though the absolute numbers are not an end-to-end deployment figure.

No production code path is altered; both read-outs are reimplemented here over the cached
features.
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import common as C                                                   # noqa: E402

REPEATS = 20
WARMUP = 5
SEED = 17


def main():
    import torch
    import psutil

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = psutil.Process()
    out = {"device": dev, "repeats": REPEATS, "warmup": WARMUP,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "scope": "read-out over cached frozen features only; excludes RAD-DINO inference"}

    vn = C.REG / "regional_vietnam_lungs_g12_v1.npz"
    vids = C.read_ids(vn)
    meta, _, report_set = C.vietnam_meta()
    rep_pos = sorted([k for k, p in enumerate(vids) if p in report_set])

    # ---------------- streaming load, timed and measured ----------------
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    host_before = proc.memory_info().rss
    t0 = time.perf_counter()
    V, _ = C.stream_regions(vn, keep_idx=rep_pos, device=dev, verbose=False)
    load_s = time.perf_counter() - t0
    host_peak = max(proc.memory_info().rss, host_before)
    out["streaming_load"] = {
        "n_cases": len(rep_pos), "seconds": round(load_s, 3),
        "vram_after_gb": round(torch.cuda.memory_allocated() / 1e9, 3) if dev == "cuda" else None,
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if dev == "cuda" else None,
        "host_rss_increase_mb": round((host_peak - host_before) / 1e6, 1),
        "npz_on_disk_gb": round(vn.stat().st_size / 1e9, 3),
    }

    torch.manual_seed(SEED)
    W_reg = torch.randn(1, 768, device=dev)
    b_reg = torch.zeros(1, device=dev)

    def time_it(fn):
        for _ in range(WARMUP):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            fn()
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        a = np.array(ts)
        return {"mean_ms": round(float(a.mean() * 1000), 3),
                "sd_ms": round(float(a.std(ddof=1) * 1000), 3),
                "min_ms": round(float(a.min() * 1000), 3),
                "per_case_us": round(float(a.mean() * 1e6 / len(rep_pos)), 2)}

    # regional read-out: 144 region scores per case, then max
    out["regional_readout"] = time_it(lambda: C.score_head(V, W_reg, b_reg, "max"))

    # global read-out over the equivalent pooled vector (mean of the same tokens)
    with torch.no_grad():
        G = torch.cat([V[i:i + 256].float().mean(1) for i in range(0, len(V), 256)]).half()
    out["global_readout"] = time_it(
        lambda: C.score_head(G.unsqueeze(1), W_reg, b_reg, "max"))
    out["regional_over_global_latency_ratio"] = round(
        out["regional_readout"]["mean_ms"] / out["global_readout"]["mean_ms"], 2)

    if dev == "cuda":
        out["vram_peak_total_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        f, t = torch.cuda.mem_get_info()
        out["vram_budget_gb"] = {"total": round(t / 1e9, 2), "cap_80pct": round(0.8 * t / 1e9, 2)}
        out["within_80pct_cap"] = bool(torch.cuda.max_memory_allocated() < 0.8 * t)

    # ---------------- artifact sizes ----------------
    out["artifact_sizes_gb"] = {
        "regional_tbx11k_g12": round((C.REG / "regional_tbx11k_lungs_g12_v1.npz").stat().st_size / 1e9, 3),
        "regional_vietnam_g12": round((C.REG / "regional_vietnam_lungs_g12_v1.npz").stat().st_size / 1e9, 3),
        "global_vietnam": round((ROOT / "artifacts/p2/external/radino_vietnam_lungs_only_v1.npz").stat().st_size / 1e9, 4),
    }
    out["feature_size_ratio_regional_over_global"] = round(
        out["artifact_sizes_gb"]["regional_vietnam_g12"]
        / out["artifact_sizes_gb"]["global_vietnam"], 1)

    v = psutil.virtual_memory()
    out["host_ram_at_measurement"] = {"available_gb": round(v.available / 1e9, 2),
                                      "total_gb": round(v.total / 1e9, 2),
                                      "percent_used": v.percent}

    json.dump(out, io.open(RUN / "tables/resource_audit.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
