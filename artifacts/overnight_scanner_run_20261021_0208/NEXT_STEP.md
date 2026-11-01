# Next step

## Validate the scanner-offset correction on DA/DB

**Why this and nothing else.** Tonight produced one result worth chasing and three that
close doors. The closed doors are useful: subspace removal is rejected, scanner information
is not linearly removable, and scanner-specific thresholds only redistribute detection. The
one open question is the **offset correction** — pooled REPORT AUROC rose from 0.7242 to
0.7448 (+0.0206) from a transformation that changes no weights, but the interval is
[−0.0030, +0.0436], a scanner-stratified bootstrap over 10,000 resamples gives
[−0.0029, +0.0440], and a permutation test returns p = 0.1575.

It is underpowered, not disproved. **Vietnam cannot settle it**, because TUNE chose the
pooling configuration and the whole cohort is exposed.

**DA/DB can.** 176 images across two sites (DA 72 with 22 TB, DB 104 with 52 TB), scored
once under the global representation in `dadb_third_site.json`, and **never used for any
selection decision** — the only cohort in the project of which that is true. It also still
has no focal-pooling score, which this would supply at the same time.

## The prompt

> Evaluate per-site score centring on DA/DB, the only cohort never used for any selection
> decision. Pre-register the following before looking at any DA/DB result.
>
> - **Representation.** 12×12 regional tokens, max aggregation, head trained on TBX11K
>   train only (healthy-only negatives, lungs-only masking, seeds 17/29/43/61/79). Use
>   `artifacts/overnight_scanner_run_20261021_0208/scripts/common.py` — its streaming
>   loader is proved 100.000000% elementwise identical to the committed path and reproduces
>   TBX11K development 0.9368 and Vietnam TUNE 0.7237 exactly. Assert both numbers before
>   trusting anything downstream.
> - **Offset estimator.** Per site, the median score of that site's non-TB cases. DA/DB has
>   no development half, so use leave-one-out: centre each case by the median of the *other*
>   non-TB cases at its site. Declare this before running it.
> - **Primary comparison.** Pooled AUROC, centred against uncentred, paired bootstrap over
>   4,000 resamples on identical cases, seed 0.
> - **Acceptance rule, with a minimum effect size this time.** Accept only if the paired
>   pooled difference is **at least +0.02**, AND its 95% CI excludes zero, AND neither
>   site's AUROC falls by more than 0.02. Tonight's preregistration stated no minimum
>   effect, and a change of +0.000053 technically satisfied one of its criteria. Do not
>   repeat that.
> - **Also report** per-site AUROC before and after — these must be identical to within
>   floating point, since a constant shift cannot reorder cases within a site, so treat any
>   deviation as a bug — and the referral-rate spread between DA and DB at a single
>   threshold.
>
> **Constraints.** Do not fit anything on DA/DB TB labels. Do not use Shenzhen, Montgomery
> or Vietnam REPORT for any choice. Do not re-run the pooling sweep. Do not re-enable the
> OOD gate in the referral path. Write to a new run directory and do not modify production
> code or tonight's artifacts. Expect about 2 GB of feature memory; check available RAM
> first and stream to GPU if it is tight, as tonight did under 0.33 GB.
>
> **Power warning, to be stated in the report.** 176 cases with 74 positives is *smaller*
> than the Vietnam REPORT half that already failed to resolve this effect. A null result on
> DA/DB is therefore uninformative about whether the effect exists and must be reported as
> inconclusive, not as disconfirmation. If the point estimate is positive and consistent
> with Vietnam's, say that the evidence is suggestive across two cohorts and that a
> properly powered multi-site cohort is what the question actually needs.

## Do not

- Re-run the scanner-subspace experiment at a different rank hoping for a different answer.
  Seeds 17, 29, 43 and 61 all give the same null, and removing 44 directions barely dents
  the scanner probe.
- Treat scanner-specific thresholds as a fix. They equalise workload and cost scanner 3
  eight detections out of 43.
- Deploy anything that needs the scanner ID at inference. The pipeline does not read it and
  has no defined behaviour for an unseen machine.
