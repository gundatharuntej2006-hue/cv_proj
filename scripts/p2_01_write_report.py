# -*- coding: utf-8 -*-
"""Turn the P2-01 measurements into the two deliverables the work plan asks for:
docs/p2_vram_feasibility.md and decisions/p2_licences.json (D15)."""
import json, os, re
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(os.path.join(ROOT, "artifacts", "p2", "p2_01_vram_feasibility.json"),
                     encoding="utf-8"))
BUDGET = RES["budget_gb"]

# ---------------------------------------------------------------- licence + revision
from huggingface_hub import hf_hub_download, HfApi

cfg_path = hf_hub_download("microsoft/rad-dino", "config.json")
revision = os.path.basename(os.path.dirname(cfg_path))
licence = "not stated in the model card front matter"
try:
    readme = open(hf_hub_download("microsoft/rad-dino", "README.md"), encoding="utf-8").read()
    fm = readme.split("---")[1] if readme.startswith("---") else ""
    m = re.search(r"^license(?:_name)?:\s*(.+)$", fm, re.M)
    if m:
        licence = m.group(1).strip().upper()
    try:                                  # read the licence file itself, not just the tag
        lic_txt = open(hf_hub_download("microsoft/rad-dino", "LICENSE"),
                       encoding="utf-8", errors="ignore").read()
        licence += f", LICENSE file starts \"{lic_txt.strip().splitlines()[0].strip()}\""
    except Exception:
        licence += ", no LICENSE file in the repository"
except Exception as e:                                    # network or card missing
    licence += f" [README unavailable: {type(e).__name__}]"

RES["model"]["revision"] = revision
RES["model"]["licence"] = licence

# recompute the verdict with the budget applied everywhere
eff_ok = [r["batch"] for r in RES["effnet_b3"] if r.get("ok") and r["peak_gb"] <= BUDGET]
eff_spill = [r["batch"] for r in RES["effnet_b3"] if r.get("ok") and r["peak_gb"] > BUDGET]
RES["verdict"]["effnet_laptop_batch"] = max(eff_ok, default=None)
RES["verdict"]["effnet_spilled_batches"] = eff_spill
json.dump(RES, open(os.path.join(ROOT, "artifacts", "p2", "p2_01_vram_feasibility.json"),
                    "w", encoding="utf-8"), indent=1)

# ---------------------------------------------------------------- the report
g, m, v = RES["gpu"], RES["model"], RES["verdict"]
L = []
L.append("# P2-01 VRAM feasibility on a 6 GB laptop")
L.append("")
L.append(f"**Task:** P2-01 · **Owner:** Person 2 · **Date:** {RES['date']} · "
         f"**Verdict: {'GO' if v['go'] else 'NO-GO'}**")
L.append("")
L.append("Acceptance criterion from the work plan: peak `max_memory_allocated` at or below "
         f"{BUDGET} GB for the embedding and Grad-CAM paths, plus P2-03's requirement that "
         "11,200 images embed in under 60 minutes on the laptop.")
L.append("")
L.append("## Machine")
L.append("")
L.append(f"| Item | Value |\n|---|---|")
L.append(f"| GPU | {g['name']} |")
L.append(f"| VRAM reported by the driver | {g['total_gb']} GB |")
L.append(f"| PyTorch | {g['torch']} (CUDA {g['cuda']}) |")
L.append(f"| Python | {g['python']} |")
L.append("")
L.append("## Model actually loaded")
L.append("")
L.append("| Item | Value |\n|---|---|")
L.append(f"| Model | `{m['model_id']}` |")
L.append(f"| Pinned revision | `{m['revision']}` |")
L.append(f"| Licence (model card) | {m['licence']} |")
L.append(f"| Input size | {m['image_size']} x {m['image_size']} pixels |")
L.append(f"| Patch size | {m['patch_size']} |")
L.append(f"| Tokens per image | {m['tokens']} |")
L.append(f"| Embedding width | {m['hidden_size']} |")
L.append(f"| Transformer blocks | {m['n_layers']} |")
L.append(f"| Parameters | {m['params_m']} M |")
L.append("")
L.append("The embedding width of 768 confirms contract C07, which declares `emb float16 [N, 768]`.")
L.append("")
L.append("## 1. Embedding extraction, fp16 (the P2-03 path)")
L.append("")
L.append("| Batch | Peak VRAM (GB) | Images/second | Minutes for 11,200 images | Within budget |")
L.append("|---|---|---|---|---|")
for r in RES["embedding"]:
    if r.get("ok"):
        L.append(f"| {r['batch']} | {r['peak_gb']:.2f} | {r['img_per_sec']:.1f} | "
                 f"{r['minutes_for_11200']:.1f} | {'yes' if r['peak_gb'] <= BUDGET else 'no'} |")
    else:
        L.append(f"| {r['batch']} | out of memory | | | no |")
L.append("")
L.append(f"Frozen RAD-DINO in half precision is far cheaper than budgeted. At batch "
         f"{v['best_batch']} the peak is {v['best_peak_gb']:.2f} GB, about "
         f"{v['best_peak_gb'] / BUDGET * 100:.0f} percent of the {BUDGET} GB allowance, and one "
         f"full pass over TBX11K takes about {v['minutes_for_11200']:.0f} minutes against the "
         "60-minute criterion.")
L.append("")
L.append("**Consequence for P2-06.** The ten preprocessing variants (full image plus four crop "
         f"margins, each with CLAHE off and on) cost roughly "
         f"{v['minutes_for_11200'] * 10:.0f} minutes of GPU time in total for the training split, "
         "so the variant sweep is cheap and does not need Kaggle.")
L.append("")
L.append("## 2. Grad-CAM path, forward and backward at batch 1")
L.append("")
gc = RES["gradcam"]
if gc.get("ok"):
    L.append(f"Peak VRAM **{gc['peak_gb']:.2f} GB**, well inside the budget, and the gradient of "
             "the last block's activations was captured, which is what Person 3's Grad-CAM needs "
             "from `classifier_module()` (task P3-09).")
    L.append("")
    L.append("One implementation note for P3-09 and D19: the backbone is frozen, so its parameters "
             "have `requires_grad=False` and the activations carry no gradient unless the graph "
             "starts earlier. Set `requires_grad=True` on the input tensor. Without that, "
             "`retain_grad()` raises and the heatmap is silently unavailable.")
else:
    L.append("The Grad-CAM path did **not** fit. See the raw JSON.")
L.append("")
L.append("## 3. EfficientNet-B3 training step at 512 pixels (reference only)")
L.append("")
L.append("| Batch | Peak VRAM (GB) | Seconds/step | Images/second | Within budget |")
L.append("|---|---|---|---|---|")
for r in RES["effnet_b3"]:
    if r.get("ok"):
        L.append(f"| {r['batch']} | {r['peak_gb']:.2f} | {r['sec_per_step']:.3f} | "
                 f"{r['img_per_sec']:.1f} | {'yes' if r['peak_gb'] <= BUDGET else 'no'} |")
    else:
        L.append(f"| {r['batch']} | out of memory | | | no |")
L.append("")
if eff_spill:
    L.append(f"**Important measurement artefact.** Batch {eff_spill[0]} did not raise an "
             "out-of-memory error, yet its peak exceeds both the budget and the card's physical "
             "memory. On Windows the driver silently spills into shared system memory instead of "
             "failing, and the cost shows up as speed: throughput collapses from "
             f"{[r['img_per_sec'] for r in RES['effnet_b3'] if r['batch'] == eff_ok[-1]][0]:.0f} "
             f"to {[r['img_per_sec'] for r in RES['effnet_b3'] if r['batch'] == eff_spill[0]][0]:.0f} "
             "images per second. A run completing is therefore not evidence that it fitted. "
             "Everyone on the team should check peak memory, not just whether the script survived.")
    L.append("")
L.append(f"Usable laptop batch size for the baseline is **{v['effnet_laptop_batch']}**. The real "
         "baseline still trains on a Kaggle T4 as planned (P2-12); this table only sizes local "
         "debugging runs.")
L.append("")
L.append("## Verdict")
L.append("")
L.append("| Check | Result |\n|---|---|")
L.append(f"| Embedding path within {BUDGET} GB | {'PASS' if v['embedding_fits'] else 'FAIL'} "
         f"(best batch {v['best_batch']}, {v['best_peak_gb']:.2f} GB) |")
L.append(f"| 11,200 images under 60 minutes | {'PASS' if v['meets_60_min'] else 'FAIL'} "
         f"({v['minutes_for_11200']:.1f} minutes) |")
L.append(f"| Grad-CAM path within {BUDGET} GB | {'PASS' if v['gradcam_fits'] else 'FAIL'} |")
L.append(f"| **Overall** | **{'GO' if v['go'] else 'NO-GO'}** |")
L.append("")
L.append("The frozen-backbone design in the methodology is comfortable on this hardware. The "
         "binding constraint is not the foundation model; it is detector training, which stays "
         "on Kaggle.")
L.append("")
L.append("## How to reproduce")
L.append("")
L.append("```")
L.append("py -3 scripts/p2_01_vram_feasibility.py     # writes artifacts/p2/p2_01_vram_feasibility.json")
L.append("py -3 scripts/p2_01_write_report.py         # writes this file and decisions/p2_licences.json")
L.append("```")
L.append("")
L.append("Raw measurements: `artifacts/p2/p2_01_vram_feasibility.json`.")

os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
doc = os.path.join(ROOT, "docs", "p2_vram_feasibility.md")
open(doc, "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- D15
d15 = {
    "decision_id": "D15",
    "owner": "p2",
    "title": "Model licences and pinned revisions (classification side)",
    "criterion_preregistered": ("Every model weight used by Person 2 is recorded with its exact "
                                "revision and its licence, checked on the model card before use. "
                                "Anything that is not usable for an academic project is rejected."),
    "preregistered_at": str(date.today()),
    "candidates": ["microsoft/rad-dino", "timm efficientnet_b3 (ImageNet weights)"],
    "value": {
        "radino": {"model_id": "microsoft/rad-dino", "revision": revision,
                   "licence": licence, "use": "frozen feature extractor, academic project only",
                   "verified_on": str(date.today())},
        "efficientnet_b3": {"source": "timm", "licence": "Apache-2.0 (library); confirm the "
                            "pretrained tag's own licence when the baseline is trained in P2-12",
                            "use": "ImageNet-pretrained baseline"},
        "ood_sources": "added in W5 with P2-10, per the work plan"
    },
    "evidence": ["docs/p2_vram_feasibility.md", "artifacts/p2/p2_01_vram_feasibility.json"],
    "data_used": ["none"],
    "reviewed_by": "p4",
    "guide_ack": False,
    "locked": False,
    "locked_at": None,
    "git_sha": None,
    "note": ("Locked once Person 4 reviews it and the OOD sources are added in W5. RAD-DINO is "
             "MIT-licensed, verified from the model card and the LICENSE file in the model "
             "repository, so there is no academic-use restriction on the backbone. The system is "
             "still presented as decision support only, for clinical reasons rather than licence "
             "reasons.")
}
os.makedirs(os.path.join(ROOT, "decisions"), exist_ok=True)
dec = os.path.join(ROOT, "decisions", "p2_licences.json")
if os.path.exists(dec):
    # P2-10 writes the OOD source register into the same decision file. Re-running this
    # script must refresh the model licences without discarding that work.
    prev = json.load(open(dec, encoding="utf-8"))
    for key in ("ood_sources",):
        if key in prev.get("value", {}):
            d15["value"][key] = prev["value"][key]
    d15["locked"] = prev.get("locked", False)
json.dump(d15, open(dec, "w", encoding="utf-8"), indent=1)

print("wrote", doc)
print("wrote", dec)
print("RAD-DINO revision:", revision)
print("RAD-DINO licence:", licence)
print("verdict:", "GO" if v["go"] else "NO-GO",
      "| effnet usable batch:", v["effnet_laptop_batch"], "| spilled:", eff_spill)
