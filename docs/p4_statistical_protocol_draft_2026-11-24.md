# P4 statistical protocol draft — 2026-11-24

**Status:** P4-owned prospective draft for team review. It does not certify the legacy D21 record, approve a sealed evaluation, or backdate a decision.

## Detector preprocessing comparison

P4 selects **2,000 paired bootstrap replicates and seed 42** for the four-candidate P3 detector preprocessing study. Use the same reviewed development image IDs and original full-512 lesion annotations for every candidate. In each replicate, sample image IDs with replacement within TB and non-TB strata, preserving each stratum's original size. Use the same sampled IDs and multiplicities for both detectors, recompute both AP50 values, and store the difference `AP50(candidate) - AP50(current)`. The 95% interval is the 2.5th and 97.5th percentiles of these 2,000 paired differences.

Start with full image/CLAHE off. Visit full image/CLAHE on, 0.10 crop/CLAHE off, then 0.10 crop/CLAHE on. Promote a candidate only if its unrounded AP50 point difference is at least 0.01 **and** the paired interval's lower endpoint is strictly above zero. Otherwise keep the current simpler choice. Record every candidate's AP50, all three ordered comparisons, image and lesion counts, replicate count, seed, input hashes, run IDs, and promotion decisions. This rule selects a development configuration; the nominal 95% intervals are not a multiple-comparison-adjusted claim of superiority.

## Governance boundary

The old D21 record cited a missing methodology file and used placeholder commit provenance. This addendum records P4's current proposal before the P3 comparison; it does not validate the old date or other unverified D21 claims. D21 remains unlocked pending review and an actual commit reference. Ritika's P3 protocol should cite this addendum and its commit after review. No P3 comparison should start until P1 supplies a verified C01 v0 file, accessible C04 v0 manifest and images, and original-box coverage/integrity evidence; the P3 runner must hash the agreed C04 path consistently. A D24 registry/gate addition still requires team approval.
