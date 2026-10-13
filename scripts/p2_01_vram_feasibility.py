# -*- coding: utf-8 -*-
"""P2-01: does the Person 2 workload fit in 6 GB of laptop VRAM?

Acceptance criterion from the work plan:
  "Peak max_memory_allocated <= 5.5 GB for embedding and Grad-CAM paths on RTX 4050;
   go/no-go recorded; D15 locked"
plus P2-03's criterion: "11,200 images <= 60 min on laptop".

What this measures, in order:
  1. RAD-DINO config: image size, patch size, embedding width. Nothing is assumed.
  2. Embedding extraction in fp16 at several batch sizes: peak VRAM and images/second.
  3. The Grad-CAM path: forward plus backward at batch 1, which Person 3 needs from
     our classifier module, so it has to fit too.
  4. EfficientNet-B3 training steps at 512 pixels: what batch size the laptop holds,
     for reference only, since the baseline actually trains on a Kaggle T4.

Every number is measured, never estimated. Out-of-memory is caught and reported as a
result rather than a crash.
"""
import json, os, platform, sys, time
from datetime import date

import torch

MODEL_ID = "microsoft/rad-dino"
OUT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "artifacts", "p2", "p2_01_vram_feasibility.json")
GB = 1024 ** 3


def banner(msg):
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}", flush=True)


def peak_reset():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def peak_gb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / GB


def gpu_info():
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "total_gb": round(p.total_memory / GB, 2),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version()}


# ----------------------------------------------------------------- 1. load the model
def load_radino():
    from transformers import AutoImageProcessor, AutoModel
    try:
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower() or "awaiting" in msg.lower():
            print("\nRAD-DINO could not be downloaded. It is a gated Microsoft model.")
            print("Fix: open https://huggingface.co/microsoft/rad-dino , accept the licence,")
            print("then run:  hf auth login   and paste a read token. Re-run this script.")
        raise
    cfg = model.config
    size = proc.size if hasattr(proc, "size") else {}
    img = size.get("shortest_edge") or size.get("height") or getattr(cfg, "image_size", None)
    info = {"model_id": MODEL_ID,
            "image_size": img,
            "patch_size": getattr(cfg, "patch_size", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "n_layers": getattr(cfg, "num_hidden_layers", None),
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    if info["image_size"] and info["patch_size"]:
        g = info["image_size"] // info["patch_size"]
        info["tokens"] = g * g + 1
    return model, proc, info


# ----------------------------------------------------------------- 2. embedding pass
def bench_embedding(model, img_size, batches=(1, 2, 4, 8, 16)):
    model = model.half().eval().cuda()
    rows = []
    for bs in batches:
        try:
            peak_reset()
            x = torch.randn(bs, 3, img_size, img_size, device="cuda", dtype=torch.float16)
            with torch.inference_mode():
                for _ in range(2):                      # warm-up
                    model(pixel_values=x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                reps = 5
                for _ in range(reps):
                    out = model(pixel_values=x)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / reps
            emb = out.pooler_output if getattr(out, "pooler_output", None) is not None \
                else out.last_hidden_state[:, 0]
            rows.append({"batch": bs, "peak_gb": round(peak_gb(), 2),
                         "sec_per_batch": round(dt, 3),
                         "img_per_sec": round(bs / dt, 1),
                         "minutes_for_11200": round(11200 / (bs / dt) / 60, 1),
                         "emb_dim": int(emb.shape[-1]), "ok": True})
            print(f"  batch {bs:>3}: peak {rows[-1]['peak_gb']:.2f} GB, "
                  f"{rows[-1]['img_per_sec']:.1f} img/s, "
                  f"{rows[-1]['minutes_for_11200']:.1f} min for 11,200", flush=True)
            del x, out, emb
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": bs, "ok": False, "error": "CUDA out of memory"})
            print(f"  batch {bs:>3}: OUT OF MEMORY", flush=True)
            torch.cuda.empty_cache()
            break
    return rows


# ----------------------------------------------------------------- 3. Grad-CAM path
def bench_gradcam(model, img_size):
    """Grad-CAM needs activations and gradients of the last block, at batch 1.
    Person 3 calls our classifier_module(), so this path must fit as well."""
    model = model.float().cuda()
    model.eval()
    for p in model.parameters():                 # frozen backbone, as in the methodology
        p.requires_grad_(False)
    last_block = model.encoder.layer[-1] if hasattr(model, "encoder") else None
    store = {}

    def hook(_m, _inp, out):
        act = out[0] if isinstance(out, tuple) else out
        act.retain_grad()
        store["act"] = act
        return out

    h = last_block.register_forward_hook(hook) if last_block is not None else None
    try:
        peak_reset()
        # the backbone is frozen, so the graph has to start at the input for the
        # activations of the last block to carry gradients at all
        x = torch.randn(1, 3, img_size, img_size, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=x)
            feat = out.last_hidden_state[:, 0]
            head = torch.nn.Linear(feat.shape[-1], 2).cuda()   # the small trained head
            score = head(feat.float())[0, 1]
        score.backward()
        got_grad = store.get("act") is not None and store["act"].grad is not None
        res = {"peak_gb": round(peak_gb(), 2), "grad_captured": bool(got_grad), "ok": True}
        print(f"  forward+backward at batch 1: peak {res['peak_gb']:.2f} GB, "
              f"activation gradient captured: {res['grad_captured']}", flush=True)
    except torch.cuda.OutOfMemoryError:
        res = {"ok": False, "error": "CUDA out of memory"}
        print("  forward+backward at batch 1: OUT OF MEMORY", flush=True)
    finally:
        if h is not None:
            h.remove()
        torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------- 4. EfficientNet-B3
def bench_effnet(batches=(4, 8, 16), img_size=512):
    import timm
    rows = []
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.CrossEntropyLoss()
    for bs in batches:
        try:
            peak_reset()
            x = torch.randn(bs, 3, img_size, img_size, device="cuda")
            y = torch.randint(0, 3, (bs,), device="cuda")
            for _ in range(2):                                  # warm-up
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lossf(model(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lossf(model(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 3
            rows.append({"batch": bs, "peak_gb": round(peak_gb(), 2),
                         "sec_per_step": round(dt, 3),
                         "img_per_sec": round(bs / dt, 1), "ok": True})
            print(f"  batch {bs:>3}: peak {rows[-1]['peak_gb']:.2f} GB, "
                  f"{rows[-1]['sec_per_step']:.3f} s/step, {rows[-1]['img_per_sec']:.1f} img/s",
                  flush=True)
            del x, y
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": bs, "ok": False, "error": "CUDA out of memory"})
            print(f"  batch {bs:>3}: OUT OF MEMORY", flush=True)
            torch.cuda.empty_cache()
            break
    del model, opt
    torch.cuda.empty_cache()
    return rows


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch. Check the driver and the torch build.")
    res = {"task": "P2-01", "date": str(date.today()), "gpu": gpu_info(), "budget_gb": 5.5}
    print("GPU:", res["gpu"])

    banner("1. RAD-DINO model card")
    model, proc, info = load_radino()
    res["model"] = info
    for k, v in info.items():
        print(f"  {k}: {v}")
    img_size = info["image_size"] or 518

    banner("2. Embedding extraction, fp16 (P2-03 path)")
    res["embedding"] = bench_embedding(model, img_size)

    banner("3. Grad-CAM path, forward + backward at batch 1 (P3-09 needs this)")
    res["gradcam"] = bench_gradcam(model, img_size)
    del model
    torch.cuda.empty_cache()

    banner("4. EfficientNet-B3 training step at 512 px (reference; trains on Kaggle T4)")
    res["effnet_b3"] = bench_effnet()

    ok_emb = [r for r in res["embedding"] if r.get("ok") and r["peak_gb"] <= res["budget_gb"]]
    best = max(ok_emb, key=lambda r: r["batch"]) if ok_emb else None
    res["verdict"] = {
        "embedding_fits": bool(best),
        "best_batch": best["batch"] if best else None,
        "best_peak_gb": best["peak_gb"] if best else None,
        "minutes_for_11200": best["minutes_for_11200"] if best else None,
        "meets_60_min": bool(best and best["minutes_for_11200"] <= 60),
        "gradcam_fits": bool(res["gradcam"].get("ok") and
                             res["gradcam"].get("peak_gb", 99) <= res["budget_gb"]),
        # a batch only counts as usable if it stays inside the budget. On Windows the
        # driver silently spills past VRAM into shared system memory instead of raising
        # out-of-memory, so "it ran" is not proof that it fitted; the speed collapse is.
        "effnet_laptop_batch": max([r["batch"] for r in res["effnet_b3"]
                                    if r.get("ok") and r["peak_gb"] <= res["budget_gb"]],
                                   default=None),
        "effnet_spilled_batches": [r["batch"] for r in res["effnet_b3"]
                                   if r.get("ok") and r["peak_gb"] > res["budget_gb"]],
    }
    res["verdict"]["go"] = bool(res["verdict"]["embedding_fits"] and res["verdict"]["gradcam_fits"]
                                and res["verdict"]["meets_60_min"])

    banner("VERDICT")
    for k, v in res["verdict"].items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("\nwrote", OUT_JSON)


if __name__ == "__main__":
    main()
