# Sealed scoring: how "the test split is read once" is enforced

**Owner:** Person 2 · **Date:** 2026-10-14 · **Covers:** P2-14 · **Status:** machinery built
and rehearsed; the real one-shot read is still pending (needs P1's C01 and P4's fitted
thresholds)

## Why this exists

The methodology promises that `internal_test` is read exactly once, after every modelling
decision is locked. That promise is the only thing standing between this project and a
reported number that has been quietly optimised against the test set — and a promise in a
document is not a mechanism. Nobody has to be dishonest for it to fail: "let me just check
whether the fix helped" is a completely natural thing to type.

So the guarantee is implemented in code (`src/p2_cls/sealed.py`, CLI
`scripts/p2_14_score_sealed.py`) with the key held by someone who is not running the
scoring.

## How it works

**1. Seal** (Person 2, before the test split is touched). `seal()` freezes what is being
tested and hashes it: model id, variant, the SHA256 of the head weights, the thresholds, and
the SHA256 of the C01 split file. That hash is the `config_sha`. Seals are **write-once** —
re-sealing under the same id is refused, because re-sealing after glimpsing a result would
defeat the entire arrangement. The seal also refuses if the weights on disk no longer match
the registry entry.

**2. Gate token** (Person 4, who holds the signing key). A token is HMAC-SHA256 signed and
names exactly one `config_sha`, with an expiry. Person 2 cannot mint one — that is the
design, not an inconvenience: *the person with the incentive to peek is not the person
holding the key*. The token is also refused if the seal has already been scored, so a
second token cannot be issued by accident.

**3. Unseal.** `score_sealed()` checks, in this order: a token exists → its signature
verifies → it names this seal → its `config_sha` still matches → it has not expired → the
seal is not already in the ledger → the weights on disk still match the seal. Only then is
the test data read. Any failure raises `SealError` and returns nothing; there is deliberately
no mode that hands back partial scores, because a number that escaped the gate is
indistinguishable afterwards from one that passed it.

**4. Ledger.** Every successful unsealing appends a row to
`artifacts/p2/sealed/sealed_ledger.jsonl` with the seal id, config hash, timestamp, token
issuer and a hash of the metrics. A seal in the ledger can never be scored again.

**5. Attempt trail.** A separate `sealed_attempts.jsonl` records every time the test data was
actually read, including reads that crashed. This closes a loophole that a real bug exposed
during the rehearsal: a `score_fn` that raises has already read the test set, and the numbers
could be recovered from a traceback. A crash does **not** spend the seal — a transient bug
must not burn the one permitted read — but it cannot be silent either. The ledger says what
counted; the attempt trail says what happened.

What the mechanism does *not* do: stop someone determined to bypass it by writing their own
script against the embeddings. Nothing in a shared repository can. What it does is make the
honest path the easy one and leave a dated, hashed trace when the path is left — which is the
realistic standard.

## What the rehearsal showed

The whole cycle was rehearsed on the **provisional** split and bundle (seal `rehearsal1`,
`--allow-provisional`), which is why it was safe to spend: that split is not P1's C01, so the
real seal will have a different `split_sha256` and is untouched. Verified on the real CLI:

| attempt | outcome |
|---|---|
| score with no token | REFUSED — "requires authorisation from the holder of the signing key" |
| token signed with a guessed key | REFUSED — signature invalid |
| token edited to extend its expiry | REFUSED — signature invalid |
| token for a different seal | REFUSED — different configuration |
| weights swapped after sealing | REFUSED — no longer match the seal |
| valid token | scored; ledger row written |
| **same token, second time** | **REFUSED — already scored** |
| ask the key holder to re-issue | REFUSED at issue time |

20 tests in `tests/p2_cls/test_sealed.py` cover these as properties. Each one asserts not
only that the attempt fails but that `score_fn` was never called, because a refusal that has
already read the test data is not a refusal.

Numbers from the rehearsal (provisional split, 901 test images, 101 TB, frozen RAD-DINO +
linear head on `full_clahe0`):

| metric | value |
|---|---|
| AUC | 0.9920 |
| sensitivity @ 70% specificity | 1.0000 [0.9634, 1.0] |
| sensitivity @ 95% specificity | 0.9703 [0.9163, 0.9898] |
| at the fitted operating point | sensitivity 0.9208 [0.8514, 0.9593], specificity 0.9888 |
| triage counts | TB 102, REFER 0, NOT_TB 799 |

Three things worth noting, all of which confirm findings already documented:

- **AUC holds up on held-out data** (0.9920 test vs 0.9947 development). This is still an
  in-distribution result and says nothing about another hospital — see
  `p2_ood_gate_finding.md`.
- **Sensitivity at 70% specificity saturates at 1.0000 on the test split too**, exactly as
  `p2_selection_criterion_issue.md` predicted. It remains the right number to *report* (it is
  the WHO screening operating point) and the wrong number to *select* on.
- **The REFER band is empty** (0 of 901 cases): `t_upper` collapsed onto `t_lower` because
  precision at the lower threshold already exceeds the 0.80 target. A three-way triage that
  never uses its middle category is not a three-way triage. This is a real design issue for
  Person 4's D23, not a bug here — it is reproduced in the rehearsal for the second time and
  needs a decision, not a workaround.

## What is still blocked

The real one-shot read needs, in order:

1. **P1's C01** — the provisional split must be replaced, which changes `split_sha256` and so
   produces a different seal.
2. **D08/D09/D11 locked** on that split (currently provisional pending C01).
3. **P4's fitted temperature and triage thresholds (D22/D23)** — the bundle's values are
   computed locally so the demo runs, and the `seal` command refuses a bundle still marked
   `provisional` unless `--allow-provisional` is passed.
4. **The signing key with P4**, never in the repository. `.gitignore` excludes `*.key`,
   `.tb_seal_key*` and `artifacts/p2/sealed/gate_*.json`.

Until all four hold, `seal` without `--allow-provisional` refuses — which is the correct
behaviour: the one-shot read should not be spendable before the thing being tested is final.

## For Person 4

You hold the key. The two commands you run:

```
py -3 scripts/p2_14_score_sealed.py issue --seal-id primary_v1 --ttl 24 --key-file <your key>
py -3 scripts/p2_14_score_sealed.py status
```

Before issuing, check the seal file (`artifacts/p2/sealed/seal_primary_v1.json`) — it is
plain JSON — and confirm the thresholds in it are the ones you fitted and that the split hash
is C01's. Issuing a token is the moment the test read becomes possible, so it is worth the
minute.
