# -*- coding: utf-8 -*-
"""P2-14: sealed scoring of the held-out internal test split.

The methodology promises that `internal_test` is read exactly once, after every modelling
decision is locked. A promise in a document is not a mechanism, so this module is the
mechanism. Three parts:

  1. **Seal.** Before the test split is touched, `seal()` freezes what is being tested: the
     registry entry, its weight SHA, the variant, the thresholds, and a SHA over all of it.
     The seal is written once and refuses to be overwritten.

  2. **Gate token.** Scoring needs a token issued by Person 4, who holds the signing key.
     A token is HMAC-signed, names exactly one seal, and expires. Person 2 cannot mint one,
     which is the point: the person with the incentive to peek is not the person holding
     the key.

  3. **Ledger.** Every successful unsealing is appended to `sealed_ledger.jsonl`. A seal
     that already appears there cannot be scored again, so "we re-ran it after a tweak"
     leaves a trace instead of being invisible.

Any failure raises `SealError` and returns nothing. There is deliberately no mode that
returns partial scores on a refused seal, because a number that escaped the gate is
indistinguishable afterwards from one that passed it.

The scoring itself is injected (`score_fn`), so this module never imports torch and the
guarantees can be tested without a model.

    from p2_cls.sealed import seal, issue_token, score_sealed
    s = seal("primary_v1", entry, thresholds, out_dir)   # P2, before locking
    t = issue_token(s.seal_id, key, ttl_hours=24)        # P4, after review
    result = score_sealed(s, t, key, score_fn)           # once, ever
"""
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

TOKEN_VERSION = 1
LEDGER_NAME = "sealed_ledger.jsonl"
ATTEMPTS_NAME = "sealed_attempts.jsonl"


class SealError(RuntimeError):
    """Scoring was refused. No scores are produced alongside this."""


def _sha(obj):
    """Stable SHA over a JSON-able object: sorted keys, no whitespace drift."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- the seal
@dataclass
class Seal:
    """What is being tested, frozen. `config_sha` covers every field below it."""
    seal_id: str
    created_utc: str
    model_id: str
    variant: str
    weight_sha256: str
    head_type: str
    backbone: str
    thresholds: dict
    split_sha256: str            # SHA of the C01 split file, so the split cannot change
    config_sha: str = ""
    note: str = ""

    def payload(self):
        d = asdict(self)
        d.pop("config_sha", None)
        d.pop("created_utc", None)        # time is not part of what is being tested
        d.pop("note", None)
        return d

    def compute_sha(self):
        return _sha(self.payload())

    def to_json(self):
        return asdict(self)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(seal_id, entry, thresholds, split_path, out_dir, note=""):
    """Freeze a model + thresholds + split for one-shot test scoring.

    Refuses to overwrite an existing seal: re-sealing under the same id after seeing any
    result would defeat the whole arrangement.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("seal_%s.json" % seal_id)
    if path.exists():
        raise SealError("seal %r already exists at %s; seals are write-once" % (seal_id, path))
    if not Path(entry.weight_path).exists():
        raise SealError("weight file missing: %s" % entry.weight_path)
    live = file_sha256(entry.weight_path)
    if live != entry.weight_sha256:
        raise SealError("weight SHA does not match the registry entry; refusing to seal "
                        "a model whose weights have changed")
    s = Seal(seal_id=seal_id,
             created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             model_id=entry.model_id, variant=entry.variant,
             weight_sha256=entry.weight_sha256, head_type=entry.head_type,
             backbone=entry.backbone, thresholds=dict(thresholds),
             split_sha256=file_sha256(split_path), note=note)
    s.config_sha = s.compute_sha()
    tmp = path.with_suffix(".tmp")
    json.dump(s.to_json(), open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, path)
    return s


def load_seal(path):
    d = json.load(open(path, encoding="utf-8"))
    s = Seal(**d)
    if s.config_sha != s.compute_sha():
        raise SealError("seal file has been edited since it was written (config SHA mismatch)")
    return s


# --------------------------------------------------------------------------- the token
@dataclass
class GateToken:
    """Person 4's authorisation to unseal exactly one seal, once, before `expires_utc`."""
    version: int
    seal_id: str
    config_sha: str
    issued_utc: int
    expires_utc: int
    issuer: str
    signature: str = ""

    def payload(self):
        d = asdict(self)
        d.pop("signature", None)
        return d

    def to_json(self):
        return asdict(self)


def _sign(payload, key):
    if isinstance(key, str):
        key = key.encode("utf-8")
    if not key:
        raise SealError("empty signing key")
    msg = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def issue_token(seal_obj, key, ttl_hours=24, issuer="person4"):
    """Issued by whoever holds the key — not by the person running the scoring."""
    now = int(time.time())
    t = GateToken(version=TOKEN_VERSION, seal_id=seal_obj.seal_id,
                  config_sha=seal_obj.config_sha, issued_utc=now,
                  expires_utc=now + int(ttl_hours * 3600), issuer=issuer)
    t.signature = _sign(t.payload(), key)
    return t


def verify_token(token, seal_obj, key, now=None):
    """Raises SealError unless this token authorises this exact seal, right now."""
    now = int(time.time() if now is None else now)
    if token.version != TOKEN_VERSION:
        raise SealError("unsupported token version %r" % (token.version,))
    expected = _sign(token.payload(), key)
    if not hmac.compare_digest(expected, token.signature or ""):
        raise SealError("token signature invalid (wrong key, or the token was edited)")
    if token.seal_id != seal_obj.seal_id:
        raise SealError("token is for seal %r, not %r" % (token.seal_id, seal_obj.seal_id))
    if token.config_sha != seal_obj.config_sha:
        raise SealError("token was issued for a different configuration; the model, "
                        "thresholds or split changed after the token was granted")
    if now > token.expires_utc:
        raise SealError("token expired at %d (now %d)" % (token.expires_utc, now))
    if now < token.issued_utc - 300:
        raise SealError("token is issued in the future; check the clock")
    return True


def save_token(token, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(token.to_json(), open(p, "w", encoding="utf-8"), indent=1)
    return str(p)


def load_token(path):
    return GateToken(**json.load(open(path, encoding="utf-8")))


# --------------------------------------------------------------------------- the ledger
def ledger_path(out_dir):
    return Path(out_dir) / LEDGER_NAME


def ledger_entries(out_dir):
    p = ledger_path(out_dir)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def already_scored(seal_id, out_dir):
    return any(r["seal_id"] == seal_id for r in ledger_entries(out_dir))


def _append_ledger(out_dir, row):
    p = ledger_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def attempts_path(out_dir):
    return Path(out_dir) / ATTEMPTS_NAME


def attempt_entries(out_dir):
    """Every time the test data was actually read, successful or not. Audit trail: the
    ledger says what counted, this says what happened."""
    p = attempts_path(out_dir)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _append_attempt(out_dir, seal_obj, token, outcome):
    p = attempts_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"seal_id": seal_obj.seal_id, "config_sha": seal_obj.config_sha,
           "attempted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "token_issuer": getattr(token, "issuer", None), "outcome": outcome}
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- the unsealing
def score_sealed(seal_obj, token, key, score_fn, out_dir, weight_path=None, now=None):
    """Score the held-out split, once, if and only if everything still checks out.

    `score_fn()` is called with no arguments and must return a JSON-able dict of metrics.
    It is called *after* every check passes, so a refused seal never reads the test data.
    """
    out = Path(out_dir)
    if token is None:
        raise SealError("no gate token: internal_test scoring requires authorisation from "
                        "the holder of the signing key")
    verify_token(token, seal_obj, key, now=now)
    if already_scored(seal_obj.seal_id, out):
        raise SealError("seal %r has already been scored; the held-out split is read once. "
                        "Score a new seal instead of re-running this one."
                        % seal_obj.seal_id)
    if weight_path is not None:
        live = file_sha256(weight_path)
        if live != seal_obj.weight_sha256:
            raise SealError("the weights on disk no longer match the seal (expected %s, "
                            "found %s)" % (seal_obj.weight_sha256[:12], live[:12]))

    # Past this line the test data is about to be read, so every outcome — including a
    # crash — is recorded. A failing score_fn does not spend the seal (a transient bug must
    # not burn the one read), but it cannot be silent either: 'crash on purpose, read the
    # numbers from the traceback' would otherwise be an invisible way in.
    t0 = time.perf_counter()
    try:
        metrics = score_fn()
        if not isinstance(metrics, dict):
            raise SealError("score_fn must return a dict of metrics")
    except BaseException as e:
        _append_attempt(out, seal_obj, token, "failed: %s: %s" % (type(e).__name__, e))
        raise
    _append_attempt(out, seal_obj, token, "completed")

    row = {"seal_id": seal_obj.seal_id, "config_sha": seal_obj.config_sha,
           "model_id": seal_obj.model_id, "variant": seal_obj.variant,
           "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "token_issuer": token.issuer, "token_issued_utc": token.issued_utc,
           "metrics_sha": _sha(metrics),
           "elapsed_s": round(time.perf_counter() - t0, 2)}
    _append_ledger(out, row)

    res_path = out / ("sealed_result_%s.json" % seal_obj.seal_id)
    json.dump({"seal": seal_obj.to_json(), "ledger": row, "metrics": metrics},
              open(res_path, "w", encoding="utf-8"), indent=1)
    return {"metrics": metrics, "ledger": row, "path": str(res_path)}
