# P3-16 · Latency experiment E6

Targets (methodology / README): **< 2 s per image on the GPU** (RTX 4050 laptop) and **< 10 s per image CPU-only**
(8 GB RAM deployment). Verdict on the **median**, p95 shown alongside. Measured once per mode, reported as it is:
nothing is tuned to make it pass.

## Protocol (fixed before measuring)

- Full pipeline: P4's `p4_pipeline.api.run_batch` (R4), one `run_batch([path])` call per image, so every
  image's wall-clock time is measured on its own as a cross-check of `CaseResult.timing_ms.total`.
- Inputs: 103 **development** images (seeded sample, seed 0, readable), given as the original files
  `data/raw/tbx11k/<rel_path>`. The first 3 are warm-up and discarded; 100 are measured. No sealed data.
- Per stage (standardise, gate, segment, classify, detect, explain, total) and wall-clock: median, p95
  (numpy linear interpolation), n. Rejected or unreadable images have fewer stages; n says so.
- Verdict: the larger of the median pipeline total and the median wall-clock, compared with the target.
- CPU mode hides CUDA before torch is imported (`CUDA_VISIBLE_DEVICES=""`) and records peak process memory.

## How to run

Close other heavy programs, plug the laptop in, and set Windows to *Best performance* (Settings → System →
Power).

```
make p3-latency-gpu      # = python -m p3_ops.latency --mode gpu
make p3-latency-cpu      # = python -m p3_ops.latency --mode cpu   (separate process: CUDA off from the start)
```

Outputs in `artifacts/p3/latency/`: `e6_gpu.csv`, `e6_cpu.csv` (mode, stage, median_ms, p95_ms, n, machine,
date, git_sha), `e6_<mode>_per_image.csv` (case_id + timings, no file paths), and `e6_<mode>.md` (table to paste).

### Limiting the CPU run to 8 GB of RAM (pick one)

1. **Windows, whole machine (simplest, most faithful):** `Win+R` → `msconfig` → *Boot* → *Advanced options* →
   tick *Maximum memory* → `8192` → OK → restart. Run `make p3-latency-cpu`. Afterwards untick the box and
   restart again. Check *Task Manager → Performance → Memory* shows about 8 GB.
2. **WSL2 (Ubuntu):** in `%UserProfile%\.wslconfig` put
   ```
   [wsl2]
   memory=8GB
   processors=4
   ```
   then `wsl --shutdown`, open Ubuntu, install the repo there and run `make p3-latency-cpu`.
3. **Docker:** `docker run --memory=8g --memory-swap=8g --gpus none ...` with the repo mounted.

Whatever you choose, the CPU report states the peak process memory and flags it if it is above 8 GB. The
machine row records CPU model, total RAM visible to the OS and GPU. With option 1 the RAM shown will be about
8 GB.

## Results (fill in from `e6_gpu.md` / `e6_cpu.md`)

| Mode | Machine | Median total (s) | p95 total (s) | Median wall-clock (s) | Target | Verdict |
|---|---|---|---|---|---|---|
| GPU | | | | | < 2 s | |
| CPU | | | | | < 10 s | |

Per-stage tables: paste `e6_gpu.md` and `e6_cpu.md` here.
