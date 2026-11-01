# Frontend API contract

**For:** whoever builds the UI · **Owner:** Person 2 · **Date:** 2026-10-14

## Starting it

```
cd D:\tb-triage
py -3 scripts/serve.py --preload
```

`http://127.0.0.1:8000`, interactive docs at `/docs`. `--preload` loads the model up front;
without it the **first** request takes ~30 seconds (backbone load) and every one after that
~60 ms. Always use `--preload` before a demo.

CORS is open, so a dev server on another port can call it directly.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | `/health` | is it up, is a model loaded, is it a stub |
| GET | `/model` | what model and thresholds are loaded, with provenance |
| GET | `/schema/decision` | the response shape, machine-readable |
| POST | `/analyse` | one image (multipart `file`) → one decision |
| POST | `/analyse-batch` | up to 40 images (multipart `files`) → decisions, sorted |

```js
const fd = new FormData();
fd.append("file", fileInput.files[0]);
const r = await fetch("http://127.0.0.1:8000/analyse", {method: "POST", body: fd});
const d = await r.json();
```

## The response

```json
{
  "image": "tb0003.png",
  "status": "ANALYSED",
  "stage": "complete",
  "reasons": [],
  "quality": {"pass": true, "reasons": []},
  "ood":  {"score": 0.5781, "threshold": 0.7, "is_ood": false, "abstained": false},
  "analysis": {
    "p_tb_raw": 0.998,
    "p_tb_calibrated": 0.9919,
    "triage": "TB",
    "pre_link": "TB",
    "link_fired": false,
    "screen_positive": true,
    "t_lower": 0.9069, "t_upper": 0.9069, "temperature": 0.4928
  },
  "provisional": true,
  "is_stub": false,
  "latency_ms": 62.5
}
```

### Four rules the UI has to respect

**1. `analysis` is `null` unless `status == "ANALYSED"`. Never show a probability without
checking.** This is not defensiveness about a flaky API — it is the point of the design. A
rejected image means "a human must look at this", *not* "this patient does not have TB", and
there is deliberately no number attached to that outcome. Render rejections as "could not be
assessed — refer", with `reasons`, and no score, no bar, no percentage.

```js
if (d.status !== "ANALYSED") { showRejected(d.reasons); return; }
showScore(d.analysis.p_tb_calibrated, d.analysis.triage);
```

**2. Show a banner whenever `provisional` or `is_stub` is true.** `provisional: true` means
the thresholds are not Person 4's fitted values and the split is not Person 1's final one.
`is_stub: true` means the head is untrained and **nothing on the screen means anything**. A
demo that cannot tell its viewer the model is provisional will mislead them, and this is a
medical screening tool. Both flags are on every response, including rejections.

**3. Use `p_tb_calibrated`, not `p_tb_raw`.** The raw score is pre-calibration and is
included only for debugging.

**4. `triage` is the decision; the probability is supporting detail.** Three values:

| value | meaning | suggested UI |
|---|---|---|
| `TB` | screen positive, high confidence | red, top of the list |
| `REFER` | uncertain, needs a human | amber |
| `NOT_TB` | screen negative | grey/green |

`screen_positive` is `true` for both `TB` and `REFER` — that is the clinically meaningful
binary, because a screening tool's job is deciding who gets a second look.

**Note on `REFER`: it is currently always empty.** `t_upper` has collapsed onto `t_lower`,
so no case lands in the middle band. The UI should still implement all three states — this
is an open design issue for Person 4's D23, not permanent — but do not be surprised when
nothing is amber. If `t_lower == t_upper` in `/model`, say so in the banner.

## Failure modes, all of which return 200 with a result

| what happened | `status` | `stage` | `reasons` |
|---|---|---|---|
| not an image / corrupt | `REJECTED` | `decode` | `["unreadable: ..."]` |
| blank, over/under exposed, wrong shape | `REJECTED` | `quality` | C09 vocabulary (below) |
| not a frontal chest X-ray | `REJECTED` | `ood` | `["out_of_distribution"]` |
| a bug in the pipeline | `ERROR` | `pipeline` | the exception text |

Quality reasons are exactly four strings, never anything else: `blank`, `under_exposed`,
`over_exposed`, `heavily_cropped`. Safe to switch on; safe to translate.

Real HTTP errors are only for malformed *requests*: 400 empty upload, 413 too large (25 MB)
or too many images (40), 415 unsupported content type.

## Batches

`POST /analyse-batch` with repeated `files` fields returns:

```json
{"n": 3, "results": [...], "ordering": "TB first, then REFER, then NOT_TB, rejected last"}
```

Already sorted in reading order — most urgent first, rejected last — so render in the order
given. **One bad file never fails the request**: it comes back as a rejected row among the
good ones, because one unreadable file must not empty a worklist.

## Things the API deliberately does not do

- **No storage.** Uploads are never written to disk and results are never saved. If the UI
  needs history, it has to hold it client-side. Storing patient images needs much more than
  a localhost demo service.
- **No authentication, binds to 127.0.0.1.** Do not expose it to a network. `--host` exists
  but the warning stands.
- **No heatmaps or lesion boxes yet.** `analysis.link_fired` and `pre_link` are where the
  detector plugs in: the link rule can only ever lift `NOT_TB` to `REFER`, never create or
  remove a `TB` call. `POST /analyse?n_lesions=N` exercises it today if you want to build
  the UI before the detector exists.

## If something breaks

- Every response has `latency_ms`; >1000 ms on a warm server means something is wrong.
- `GET /health` reports `n_analysed`, which is useful for confirming requests arrive.
- `GET /model` prints the loaded thresholds, so "why is everything NOT_TB" is usually
  answerable by looking at `t_lower` (currently 0.9069, which is high by design — the
  conformal rule allows a 10% miss rate at that point).
