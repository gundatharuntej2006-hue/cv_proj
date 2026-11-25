# P4 governance review — 2026-11-19

This review covers the state of `origin/main` before any new sealed evaluation. It does not certify any decision or result.

## Findings

- All 23 decision JSON files on `main` were marked locked and carried the identical `git_sha` value `c0ffee1`. That value is a placeholder, not provenance for the recorded decisions.
- D21 cites `docs/methodology.md#section-9` and `#section-12`; the file is absent. Its claimed review and guide acknowledgement cannot be verified from the repository.
- D22 cites `artifacts/p4/calibration/dev_temperature_fit.json` and `reliability_dev.csv`; neither is present. The recorded temperature `1.37` is therefore unverified.
- D23 cites `artifacts/p4/triage/operating_curve_calibration_v1.parquet` and `calibration_split_metrics.json`; neither is present. The recorded thresholds `0.18` and `0.83` are therefore unverified.
- D01/D02 and the other owners' records also require owner review. D02's claimed review and 14 duplicate removals are not supported by its missing cited evidence. D06 conflicts with an unlocked D06 candidate on `origin/atul`. Atul's proposed D06 uses `tau=0.95` and `>=`. In `8013b42`, `reviewed_by` is empty and `data_used` is `images_only_all_splits`; Atul reports that it passes both C16 validators. These two field concerns are resolved. D06 remains unlocked pending P1-07 review and resolution of the competing D06 record.
- Correction after Atul's reply: `origin/atul` branched before the P4 fixes in commits `0768b33` and `41a5ffe`. It has no commits changing `src/tbcore/guard.py` or `contracts/README.md`. A two-tip diff showed the later `main` fixes in reverse; it did not show an Atul-authored regression. The P4 single-use guard and authoritative C16 schema note must still be preserved when branches merge.

## P4 action

D21–D23 are marked draft with the unverified review, guide acknowledgement, lock, and SHA fields cleared. Their proposed values and evidence references remain visible for audit; they are not approved settings. The gate now rejects placeholder provenance, absent evidence, duplicate records, and unlocked decisions before issuing tokens. No sealed token was issued during this review.

Before relocking D21–D23, P4 must supply the real protocol and development/calibration outputs, obtain the named reviews and any required guide acknowledgement, record the actual commit provenance, and rerun the gate. All owners must resolve their own records, and P1 must publish reviewed C01 v1 before the one-time sealed evaluation.

The older P4 report sections contain illustrative or unverified figures. They must be regenerated from traceable artifacts after the decisions are approved; they are not evidence of a completed sealed evaluation.
