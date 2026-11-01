# Our diagnosis was wrong: removing the hard negatives helps

**Owner:** Person 2 · **Date:** 2026-10-20 · **Status:** exploratory, provisional split ·
**Supersedes the proposed fix in:** `p2_vietnam_result.md` §5

## The prediction, and why it was wrong

`p2_vietnam_result.md` localised the screening failure to the negative class: our model
assigns median 0.886 to Xpert-negative screening attendees against 0.953 to cases, and
TB-versus-healthy on the benchmark is AUC 1.0000 against 0.9861 for TB-versus-sick. The
natural inference was that the training negatives are too easy, and that **training against
the 3,800 sick-but-not-tuberculous images alone** would force the model to learn
TB-versus-disease and transfer better.

We tested four negative compositions. The prediction is **wrong, and wrong in the opposite
direction**:

| negatives used in training | n_train | TBX11K dev | Shenzhen | Montgomery | **Vietnam** |
|---|---|---|---|---|---|
| **full-image masking** | | | | | |
| all (healthy + sick) — the baseline | 5,941 | 0.9951 | 0.9404 | 0.8254 | 0.6196 |
| sick only | 3,241 | 0.9432 | 0.8692 | 0.7573 | 0.6126 |
| sick ×3 | 11,341 | 0.9931 | 0.9188 | 0.7756 | **0.5935** |
| **healthy only** | 3,241 | 0.9577 | 0.9354 | **0.9278** | **0.6538** |
| **lungs-only masking** | | | | | |
| all | 5,941 | 0.9891 | 0.9294 | 0.8547 | 0.6428 |
| sick only | 3,241 | 0.9461 | 0.8484 | 0.7769 | 0.6412 |
| sick ×3 | 11,341 | 0.9861 | 0.9013 | 0.8147 | 0.6196 |
| **healthy only** | 3,241 | 0.9569 | **0.9412** | **0.9295** | **0.6743** |

**Adding hard negatives makes external transfer worse. Removing them makes it better.**
Weighting sick negatives ×3 gives the worst Vietnam result of all (0.5935).

The best configuration — lungs-only masking, healthy-only negatives — improves Vietnam from
0.6428 to 0.6743. Paired bootstrap on the same patients: **ΔAUC +0.0315, 95% CI [+0.0163,
+0.0465]** — significant.

The direction is consistent across **six independent comparisons**: healthy-only wins at
Shenzhen, Montgomery and Vietnam, under both maskings. Individual confidence intervals
overlap, but six out of six in the same direction is the stronger evidence.

The largest single effect is at **Montgomery: 0.8254 → 0.9278**, a gain of 0.102.

## Why this might be, and what we do not know

We do not have a demonstrated mechanism, and should not pretend otherwise. Three candidate
explanations, none yet tested:

1. **The sick_non_tb class may carry label noise.** A "sick but not tuberculous" label in a
   high-burden setting is a clinical judgement; if some of those 3,800 images contain
   undiagnosed or healed tuberculosis, training against them teaches the model to suppress
   genuine tuberculosis features. That would explain why *more* of them (×3) is *worse*.
2. **Training against sick controls may suppress transferable features.** To separate TB from
   other disease within TBX11K, the model must discard whatever the two have in common —
   general signs of abnormality. Those shared signs may be exactly what transfers, since
   another hospital's tuberculosis looks different but its abnormality does not.
3. **The sick class may carry its own acquisition signature.** If TBX11K's sick images were
   collected through a third route, training against them adds a second dataset-specific
   axis for the model to exploit, which then fails elsewhere.

Explanation 2 is uncomfortable but important: it would mean a model that is *worse* at the
clinically meaningful discrimination (TB versus other disease) is *better* at transferring,
because it has fallen back on abnormality detection. That is not obviously a good trade, and
it should not be presented as one without saying so.

## What this does and does not change

**It does not rescue the system.** 0.6743 against the commercial comparator's 0.7827 is still
a clear deficit, and at screening operating points the gap remains large. A +0.03 gain does
not make this deployable.

**It does change the recommended configuration**, and it strengthens the D09 case: the best
setup across every external site is **lungs-only masking with healthy-only negatives**, which
is also the configuration that structurally cannot use the border shortcut.

**It is a caution about acting on a plausible diagnosis.** We had a coherent story — curated
negatives, model calls everyone abnormal, therefore use harder negatives — and the data said
the opposite. The story was consistent with every observation we had, and still wrong. Worth
recording as a methodological note rather than quietly dropping.

**A bug worth noting in our own tooling:** the script printed "the negative composition IS the
lever; the diagnosis holds" because its automated verdict tested only the *size* of the best
gain, not *which* intervention produced it. The gain came from the opposite manipulation to
the one predicted. An automated interpretation that cannot distinguish "confirmed" from
"refuted" is worse than none.

## Caveats

- Exploratory; provisional split; `internal_test` untouched.
- Weighting is by row replication, which interacts with the logit-adjusted loss; a proper
  per-sample weighting might behave differently.
- Changing the negative composition changes the class prior, so the calibrated probabilities
  are not comparable across rows — only the AUCs are.
- The `all` baseline here reads 0.6196 against 0.6151 in `p2_vietnam_result.md`. The
  difference is row ordering, since this script concatenates by category; it indicates about
  0.005 of run-to-run noise in a five-seed average, which is a useful thing to know and
  smaller than the effect reported above.
- Development is held fixed across compositions so early stopping stays comparable, which
  means development no longer matches the training distribution for three of the four rows.

Evidence: `artifacts/p2/dryrun/hard_negatives.json`, via `scripts/p2_29_hard_negatives.py`.
