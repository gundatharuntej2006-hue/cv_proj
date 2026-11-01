# Suggested next action

Based on what the 2026-10-21 overnight run actually found, not on a generic plan.

## Context the next session needs

Three facts from last night change what is worth doing:

1. **`artifacts/p2/audit/vietnam_case_level_gate_audit.csv` holds the focal-pooling
   per-case scores** (12×12 max, seeds 17/29/43/61/79) for all 2,708 Vietnam participants,
   with `rad_system` and `split_half` alongside. Verified from
   `scripts/p2_35_gate_operational_audit.py:72-101`, and it reproduces TUNE 0.7237 /
   REPORT 0.7242 exactly. Most Vietnam questions need no refitting and no large memory.
2. **TBX11K carries no acquisition-source column**, so no source-grouped training objective
   can be built on permitted training data. Do not re-plan around one.
3. **The scanner disparity is mostly a threshold problem.** Per-scanner AUROC spans
   0.700–0.782; per-scanner thresholds for the same 70% specificity span 1.49–3.73 logits.
   Referral rate is 50.3% on rad_system 3 against 24.6% on rad_system 2, at *lower*
   prevalence.

---

## The prompt

> Run the source-subspace removal experiment pre-registered in
> `artifacts/overnight_run_20261021_2340/PRIMARY_EXPERIMENT_PREREGISTRATION.md` §4. Read
> that section first and do not change the design after seeing any result.
>
> **Protocol, fixed in advance:**
> - The tuberculosis head stays trained on **TBX11K train only** (healthy-only negatives,
>   lungs-only masking, 12×12 regional max, seeds 17/29/43/61/79), early-stopped on TBX11K
>   development. It is **not** refitted on Vietnam.
> - On **Vietnam TUNE only**, fit a linear subspace predicting `rad_system` from the 768-d
>   token representation. Project that subspace out of the tokens before pooling. Rank is
>   fixed at 3 before any result is seen. Scanner labels are not the outcome and TUNE is the
>   declared development half, so no outcome information leaks.
> - Evaluate **once** on Vietnam REPORT. Report pooled and per-scanner AUROC with paired
>   bootstrap against the unprojected head on identical cases.
> - **Decision rule: accept only if the worst-scanner AUROC improves AND the paired pooled
>   difference excludes zero.** A pooled gain with worst-scanner degradation is a rejection.
>   Worst-scanner AUROC is currently **0.7000** (rad_system 3), and per-scanner estimates on
>   this cohort swing by up to **0.055** between random halves, so an improvement below that
>   is noise.
>
> **Memory.** This needs `regional_tbx11k_lungs_g12_v1.npz` (1.35 GB) and
> `regional_vietnam_lungs_g12_v1.npz` (555 MB). Check available RAM first — it was 0.56 GB
> on 2026-10-21. If it is under about 4 GB, stream the `reg` array out of the zip in row
> blocks and accumulate only the pooled result; do not call `np.load` on the whole file.
>
> **Constraints that still hold.** Do not fit, tune, calibrate or select on REPORT, Shenzhen
> or Montgomery. Do not re-run the 18-configuration pooling sweep. Do not re-enable the OOD
> gate as a referral filter. Do not claim the architecture is novel. Write outputs to a new
> run directory; do not overwrite last night's.
>
> Label the result **exploratory**: Vietnam TUNE already chose the pooling configuration, so
> this cannot be an independent generalisation estimate.

---

## If that is too heavy, do this instead

**Per-source threshold calibration — no new features, no fitting on reserved data.**

Fit one referral threshold per `rad_system` on the **TUNE** half at a fixed 70% specificity,
freeze them, apply to **REPORT**, and report per-scanner sensitivity, specificity, referral
rate and precision against the current single global threshold.

Last night's numbers say this removes the twofold referral-load disparity outright. It is a
deployment-relevant result the project can state plainly, it costs minutes, and it needs
only `artifacts/p2/audit/vietnam_case_level_gate_audit.csv`.

The honest caveat to write alongside it: equalising specificity does not equalise
sensitivity (0.636 / 0.709 / 0.588 at local thresholds), so it fixes the workload inequity,
not the detection inequity.

---

## Two smaller items worth clearing

- **Score DA/DB under the regional representation.** It is the last external cohort where
  focal pooling is untested, and it is the only one never used for any selection decision —
  which makes it the cleanest comparison available. 176 images, small memory cost.
- **Apply or reject the three patches in `REPORT_CONSISTENCY_REVIEW.md`.** The 9×9 sweep row
  with no committed artifact is the one that matters; a reader checking the headline table
  against the repository will not find it.

## Do not

- Build a source-balanced training objective on TBX11K. There is no source variable; this
  was checked column by column.
- Treat any Vietnam, Shenzhen or Montgomery number as independent external validation.
- Add architectural complexity. Attention-MIL was already measured and is worse (0.6416
  against 0.7239 on TUNE), and the learning curve is flat, so more data will not help either.
