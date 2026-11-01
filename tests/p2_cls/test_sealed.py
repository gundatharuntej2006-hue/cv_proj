# -*- coding: utf-8 -*-
"""P2-14. The property under test is not "does it score correctly" but "is it hard to
cheat". Each test is one way someone could peek at the held-out split, and the assertion is
that the attempt fails *and* that `score_fn` was never called — because a refusal that has
already read the test data is not a refusal.
"""
import json
import time

import pytest

from p2_cls import registry as R
from p2_cls import sealed as S

KEY = "person4-signing-key-not-held-by-p2"
THR = {"t_lower": 0.31, "t_upper": 0.87, "temperature": 1.42}


class Counter:
    """A score_fn that records whether it ran."""
    def __init__(self, metrics=None):
        self.calls = 0
        self.metrics = metrics if metrics is not None else {"auc": 0.97, "sens": 0.91}

    def __call__(self):
        self.calls += 1
        return dict(self.metrics)


@pytest.fixture
def setup(tmp_path):
    """A real head file + registry entry + split file, so SHAs are genuine."""
    import numpy as np
    w = tmp_path / "head.npz"
    with open(w, "wb") as fh:
        np.savez(fh, W=np.zeros((2, 8), np.float32), b=np.zeros(2, np.float32),
                 n_classes=np.array(2))
    split = tmp_path / "c01.csv"
    split.write_text("image_id,split\na,train\nb,internal_test\n", encoding="utf-8")
    entry = R.RegistryEntry(model_id="m1", head_type="binary", variant="full_clahe0",
                            backbone="microsoft/rad-dino@110cbc18", weight_path=str(w),
                            weight_sha256=S.file_sha256(w), n_classes=2, is_primary=True)
    out = tmp_path / "sealed"
    return {"entry": entry, "split": split, "out": out, "weight": w}


# ------------------------------------------------------------------ sealing
def test_seal_is_write_once(setup):
    S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    with pytest.raises(S.SealError, match="write-once"):
        S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])


def test_seal_refuses_if_weights_changed_since_registration(setup):
    setup["weight"].write_bytes(b"different bytes")
    with pytest.raises(S.SealError, match="weight SHA"):
        S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])


def test_seal_file_roundtrips_and_detects_hand_editing(setup, tmp_path):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    path = setup["out"] / "seal_s1.json"
    assert S.load_seal(path).config_sha == s.config_sha

    d = json.load(open(path, encoding="utf-8"))
    d["thresholds"]["t_lower"] = 0.05           # quietly loosen the threshold
    json.dump(d, open(path, "w", encoding="utf-8"))
    with pytest.raises(S.SealError, match="edited"):
        S.load_seal(path)


def test_config_sha_covers_thresholds_and_split(setup):
    a = S.seal("a", setup["entry"], THR, setup["split"], setup["out"])
    b = S.seal("b", setup["entry"], dict(THR, t_lower=0.4), setup["split"], setup["out"])
    assert a.config_sha != b.config_sha, "changing a threshold must change the seal"


# ------------------------------------------------------------------ the token
def test_no_token_refuses_without_reading_the_test_data(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    fn = Counter()
    with pytest.raises(S.SealError, match="no gate token"):
        S.score_sealed(s, None, KEY, fn, setup["out"])
    assert fn.calls == 0


def test_p2_cannot_mint_a_token_with_a_guessed_key(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    forged = S.issue_token(s, "a-key-p2-made-up")
    fn = Counter()
    with pytest.raises(S.SealError, match="signature invalid"):
        S.score_sealed(s, forged, KEY, fn, setup["out"])
    assert fn.calls == 0


def test_editing_a_token_invalidates_it(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    t = S.issue_token(s, KEY, ttl_hours=1)
    t.expires_utc += 10 ** 6                       # extend it by hand
    with pytest.raises(S.SealError, match="signature invalid"):
        S.verify_token(t, s, KEY)


def test_a_token_for_one_seal_does_not_open_another(setup):
    a = S.seal("a", setup["entry"], THR, setup["split"], setup["out"])
    b = S.seal("b", setup["entry"], dict(THR, t_upper=0.9), setup["split"], setup["out"])
    t = S.issue_token(a, KEY)
    fn = Counter()
    with pytest.raises(S.SealError, match="different configuration|token is for seal"):
        S.score_sealed(b, t, KEY, fn, setup["out"])
    assert fn.calls == 0


def test_expired_token_refuses(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    t = S.issue_token(s, KEY, ttl_hours=1)
    fn = Counter()
    with pytest.raises(S.SealError, match="expired"):
        S.score_sealed(s, t, KEY, fn, setup["out"], now=t.expires_utc + 1)
    assert fn.calls == 0


def test_token_survives_a_save_load_roundtrip(setup, tmp_path):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    p = S.save_token(S.issue_token(s, KEY), tmp_path / "tok" / "gate.json")
    assert S.verify_token(S.load_token(p), s, KEY)


# ------------------------------------------------------------------ the unsealing
def test_valid_token_scores_once_and_writes_the_ledger(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    t = S.issue_token(s, KEY)
    fn = Counter({"auc": 0.9712})
    res = S.score_sealed(s, t, KEY, fn, setup["out"], weight_path=setup["weight"])

    assert fn.calls == 1
    assert res["metrics"]["auc"] == 0.9712
    rows = S.ledger_entries(setup["out"])
    assert len(rows) == 1 and rows[0]["seal_id"] == "s1"
    assert rows[0]["config_sha"] == s.config_sha
    assert json.load(open(res["path"], encoding="utf-8"))["metrics"]["auc"] == 0.9712


def test_a_second_run_of_the_same_seal_is_refused(setup):
    """The whole point: 'we tweaked it and re-ran the test' cannot happen quietly."""
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    t = S.issue_token(s, KEY)
    S.score_sealed(s, t, KEY, Counter(), setup["out"])
    fn = Counter()
    with pytest.raises(S.SealError, match="already been scored"):
        S.score_sealed(s, t, KEY, fn, setup["out"])
    assert fn.calls == 0
    assert len(S.ledger_entries(setup["out"])) == 1


def test_swapping_the_weights_after_sealing_is_caught(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    t = S.issue_token(s, KEY)
    setup["weight"].write_bytes(b"a better model, trained after the seal")
    fn = Counter()
    with pytest.raises(S.SealError, match="no longer match the seal"):
        S.score_sealed(s, t, KEY, fn, setup["out"], weight_path=setup["weight"])
    assert fn.calls == 0


def test_a_refused_seal_leaves_no_ledger_row_and_no_result_file(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    with pytest.raises(S.SealError):
        S.score_sealed(s, None, KEY, Counter(), setup["out"])
    assert S.ledger_entries(setup["out"]) == []
    assert not (setup["out"] / "sealed_result_s1.json").exists()


def test_ledger_is_append_only_across_different_seals(setup):
    for sid in ("s1", "s2", "s3"):
        s = S.seal(sid, setup["entry"], dict(THR, t_upper=0.8 + len(sid) / 100),
                   setup["split"], setup["out"])
        S.score_sealed(s, S.issue_token(s, KEY), KEY, Counter(), setup["out"])
    rows = S.ledger_entries(setup["out"])
    assert [r["seal_id"] for r in rows] == ["s1", "s2", "s3"]
    assert all(r["scored_utc"] for r in rows)


def test_already_scored_helper_is_usable_before_asking_for_a_token(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    assert not S.already_scored("s1", setup["out"])
    S.score_sealed(s, S.issue_token(s, KEY), KEY, Counter(), setup["out"])
    assert S.already_scored("s1", setup["out"])


def test_metrics_sha_detects_a_later_edit_of_the_result_file(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    res = S.score_sealed(s, S.issue_token(s, KEY), KEY, Counter({"auc": 0.90}),
                         setup["out"])
    d = json.load(open(res["path"], encoding="utf-8"))
    d["metrics"]["auc"] = 0.99
    assert S._sha(d["metrics"]) != S.ledger_entries(setup["out"])[0]["metrics_sha"]


def test_score_fn_must_return_metrics(setup):
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    with pytest.raises(S.SealError, match="must return a dict"):
        S.score_sealed(s, S.issue_token(s, KEY), KEY, lambda: 0.97, setup["out"])


# ------------------------------------------------------------------ the attempt trail
def test_a_crash_leaves_a_trace_but_does_not_spend_the_seal(setup):
    """'Crash on purpose and read the numbers from the traceback' must not be invisible —
    but a transient bug must not burn the one permitted read either."""
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    t = S.issue_token(s, KEY)

    def boom():
        raise ZeroDivisionError("peeked")

    with pytest.raises(ZeroDivisionError):
        S.score_sealed(s, t, KEY, boom, setup["out"])

    assert S.ledger_entries(setup["out"]) == [], "a crash must not count as a scoring"
    att = S.attempt_entries(setup["out"])
    assert len(att) == 1 and att[0]["outcome"].startswith("failed: ZeroDivisionError")

    # the seal is still usable, and using it now shows both reads in the trail
    S.score_sealed(s, t, KEY, Counter(), setup["out"])
    att = S.attempt_entries(setup["out"])
    assert [a["outcome"] for a in att] == [att[0]["outcome"], "completed"]
    assert len(S.ledger_entries(setup["out"])) == 1


def test_refused_before_reading_leaves_no_attempt_row(setup):
    """A gate failure happens before the data is touched, so it is not an attempt at all."""
    s = S.seal("s1", setup["entry"], THR, setup["split"], setup["out"])
    with pytest.raises(S.SealError):
        S.score_sealed(s, None, KEY, Counter(), setup["out"])
    assert S.attempt_entries(setup["out"]) == []
