# Operations: report PDF, prediction log and latency (E6)

*Owner: P3. Reviewer: P4.*

## Per-case PDF report

`make_pdf` produces a one-page A4 report for each case [src/p3_ops/pdf_report.py]. It contains:
- the image next to the Grad-CAM overlay, captioned "Attention evidence, not a finding";
- the calibrated TB probability, the final category with a colour tag, both thresholds, the link-rule flag and
  whether it fired;
- the lesion boxes with their scores, and the operating point.

Rejected and unreadable images get a report that shows only the reason and "Send for manual review". The
decision-support disclaimer and scope statement are imported from their single source in `tbcore`, never retyped,
and no PDF is produced without them. The case is identified only by its salted case ID. Free text from the input
is redacted, and an overlay is embedded only if it belongs to the same case.

Tests cover the 7 CaseResult scenario types [tests/p3_ops/test_pdf_report.py]:
- the disclaimer and scope text are extractable from every PDF;
- planted identifiers (DICOM tags and names, user paths, file names, personal names, dates of birth) never appear
  in the text or metadata;
- every file is below 2 MB [tests/p3_ops/test_pdf_report.py].

Those tests used stand-ins for P4's fixtures and for `tbcore`. Re-run on P4's fixtures: TBD: waiting on P4's
CaseResult fixtures (P4-02) and tbcore.

## Prediction log (no PHI)

`log_case` appends exactly one JSON line per case to `logs/predictions.jsonl` (contract C15) [src/p3_ops/logger.py].
- **Content.** Each line is built only from whitelisted, type-checked fields: case ID, run ID, status, final
  category, link-rule flag, gate result, calibrated probability, thresholds, operating point, latency and four
  required version fields. A line without its version fields is refused.
- **No free text.** Free text and file paths never reach the log, and version strings are limited to a safe
  character set.
- **Concurrency.** A file lock makes appends safe across threads and processes; tested with 50 parallel writers
  [tests/p3_ops/test_logger.py].
- **No identifiers.** A grep test over a log built from all scenario types, with identifiers planted in every
  free-text field, finds none [tests/p3_ops/test_logger.py].

## Latency (experiment E6)

**Targets.** Under 2 s per image on a GPU and under 10 s per image CPU-only [README.md, project summary].

**Protocol** [src/p3_ops/latency.py; docs/p3_latency_e6.md]:
- P4's full pipeline (`run_batch`) on 100 development images, after 3 warm-up images [src/p3_ops/latency.py];
- each image timed on its own;
- median and p95 for every stage and for wall-clock;
- the verdict uses the median, taking the larger of pipeline total and wall-clock.

| Mode | Machine | Median total | p95 total | Median wall-clock | n | Target | Verdict | Source |
|---|---|---|---|---|---|---|---|---|
| GPU | TBD | TBD | TBD | TBD | TBD | < 2 s | TBD | TBD: waiting on artifacts/p3/latency/e6_gpu.csv |
| CPU-only (8 GB) | TBD | TBD | TBD | TBD | TBD | < 10 s | TBD | TBD: waiting on artifacts/p3/latency/e6_cpu.csv |

Per-stage medians and p95 (standardise, gate, segment, classify, detect, explain): TBD: waiting on
artifacts/p3/latency/e6_gpu.md and artifacts/p3/latency/e6_cpu.md. Peak memory of the CPU run against the 8 GB
budget: TBD: waiting on artifacts/p3/latency/e6_cpu.md.

Detector alone on the laptop GPU (target ≤ 500 ms per image): TBD: waiting on the `make p3-latency-detect` output.
