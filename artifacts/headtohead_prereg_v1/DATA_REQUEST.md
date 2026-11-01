# Data request — one untouched screening cohort for a pre-registered head-to-head

**From:** TB-Triage project, Department of AI & ML, BMS Institute of Technology and Management, Bengaluru (P2 contact: Atul Dhull)
**To:** data holders of programmatic chest-X-ray TB screening cohorts (TB Portals / NIAID; the Ho Chi Minh City active-case-finding group; Bengaluru TB research centres)
**Date:** 2026-11-01

## What we are asking for

One retrospective screening cohort, under your governance, with — per participant:

| item | required | notes |
|---|---|---|
| frontal chest radiograph | yes | ≥ 512 px short side; DICOM or PNG; de-identified |
| microbiological TB result | yes | Xpert Ultra or culture; not radiologist reading |
| previous-TB history | yes | self-reported acceptable; source stated |
| radiography system / site identifier | yes | per image; needed for scanner-stratified reporting |
| commercial CAD score and version | yes | produced by the vendor software, independent of us |
| screening questionnaire | preferred | sex, age group, cough, fever, weight loss, night sweats, contact, diabetes |
| screening event or cluster identifier | preferred | lets us resample at the cluster level |

**Size:** at least 400 microbiologically confirmed cases (≈ 4,000 participants at 10% prevalence); the pre-registered power calculation is in `PROTOCOL.md` §7. Smaller cohorts are still evaluated once and reported as underpowered.

**The one condition that matters:** no member of our team has seen the images, scores or labels before the evaluation is locked. We ask you to state this; it is recorded in the evaluation manifest.

## What we will do with it

Score it **once**, with a frozen model and a protocol whose hash is already committed (`PROTOCOL.md` §2, §4): a paired comparison of our frozen candidate against the commercial CAD on identical participants, with non-inferiority at a 0.03 AUROC margin as the pre-specified decision, previous TB as a pre-specified subgroup, and scanner-stratified reporting. No training, tuning, threshold fitting or re-runs on your data. All code is in the project repository before the data arrives; the scoring script refuses a second run.

## What you get

- The full result on your cohort, whatever it shows, in the same honest form as our existing record — intervals first, including the previous-TB subgroup and per-scanner behaviour, within four weeks of receipt.
- A per-image region-level attribution for every radiograph (the model's decision cell), which your readers can check against their own reads.
- Co-authorship on the resulting report for the data holder's named contributors, under your authorship policy.
- Deletion of the data at the end of the agreed period, with written confirmation.

## What we are not asking for

Identifiable data, HIV status, treatment outcomes, or access to the vendor's software. We do not need the data to leave your governance if a secure enclave or a remote-execution arrangement is preferred; the scorer is a single Python script with pinned dependencies and runs on a laptop GPU.

## Why us

Our existing evaluation is on a public cohort (Codlin et al. 2025, Dryad) where our frozen system reached a point estimate level with the commercial comparator — and where every design choice was, by necessity, made on the same patients. We have published that limitation ourselves rather than around it. Your cohort is the test we cannot run on public data.

## Governance

We will sign your data-use agreement; ethics approval reference and institutional sign-off from BMSIT&M to be supplied on request. Point of contact and project record available on request.
