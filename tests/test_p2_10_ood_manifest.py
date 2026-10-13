# -*- coding: utf-8 -*-
"""P2-10: the C03 producer. The checks that matter are the two that are easy to get wrong
by hand — licence provenance and role disjointness."""
import json

import pytest

from p2_gate import ood_manifest as M


def make_images(tmp_path, name, n, suffix=".png"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / ("%s_%04d%s" % (name, i, suffix))).write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(16))
    return d


def src(tmp_path, name, ood_type, n, verified=True):
    return M.OodSource(name=name, root=make_images(tmp_path, name, n), ood_type=ood_type,
                       licence="CC-BY-4.0", licence_url="https://example.invalid/licence",
                       licence_verified=verified, verified_by="p2", verified_on="2026-10-06")


@pytest.fixture
def sources(tmp_path):
    return [src(tmp_path, "lat", "lateral_cxr", 600),
            src(tmp_path, "bone", "other_body_part", 250),
            src(tmp_path, "photos", "non_medical", 220)]


# ---------------------------------------------------------------- licence provenance
def test_an_unverified_licence_is_refused_by_default(tmp_path):
    s = [src(tmp_path, "lat", "lateral_cxr", 10, verified=False)]
    with pytest.raises(ValueError, match="licence_verified=False"):
        M.build(s)


def test_an_unverified_source_is_stamped_and_fails_validation(tmp_path):
    s = [src(tmp_path, "lat", "lateral_cxr", 500, verified=False),
         src(tmp_path, "bone", "other_body_part", 250),
         src(tmp_path, "photos", "non_medical", 220)]
    rows = M.build(s, allow_unverified=True)
    assert all(r["licence"].startswith("UNVERIFIED:") for r in rows if r["source"] == "lat")
    problems = M.validate(rows)
    assert any("unverified licence" in p for p in problems), \
        "an unverified source must never pass validation silently"


def test_the_ood_type_vocabulary_is_closed(tmp_path):
    with pytest.raises(ValueError, match="ood_type"):
        M.OodSource(name="x", root=tmp_path, ood_type="xray_of_a_cat", licence="CC0")


# ---------------------------------------------------------------- role disjointness
def test_viewhead_pools_and_ood_eval_never_share_an_image(sources):
    rows = M.build(sources)
    vh = {r["image_id"] for r in rows if r["role"] in M.VIEWHEAD_ROLES}
    ev = {r["image_id"] for r in rows if r["role"] == "ood_eval"}
    assert vh and ev
    assert not (vh & ev), "a view-head training image inside the sealed OOD set"


def test_only_laterals_serve_the_view_head(sources):
    rows = M.build(sources)
    for r in rows:
        if r["role"] in M.VIEWHEAD_ROLES:
            assert r["ood_type"] == "lateral_cxr"


def test_every_viewhead_pool_is_populated(sources):
    rows = M.build(sources)
    for role in M.VIEWHEAD_ROLES:
        assert sum(1 for r in rows if r["role"] == role) > 0, "%s is empty" % role


def test_role_assignment_is_stable_when_a_source_is_added(tmp_path, sources):
    """Adding a non-medical source must not reshuffle which laterals are sealed. A plain
    rng.shuffle would, and the OOD set would silently change between runs."""
    before = {r["image_id"]: r["role"] for r in M.build(sources)}
    sources.append(src(tmp_path, "more_photos", "non_medical", 30))
    after = {r["image_id"]: r["role"] for r in M.build(sources)}
    for iid, role in before.items():
        assert after[iid] == role, "%s moved from %s to %s" % (iid, role, after[iid])


def test_a_different_seed_gives_a_different_split(sources):
    a = {r["image_id"]: r["role"] for r in M.build(sources, seed=1)}
    b = {r["image_id"]: r["role"] for r in M.build(sources, seed=2)}
    assert a != b


# ---------------------------------------------------------------- acceptance criteria
def test_a_valid_collection_passes(sources):
    assert M.validate(M.build(sources)) == []


def test_fewer_than_200_in_a_type_fails(tmp_path):
    s = [src(tmp_path, "lat", "lateral_cxr", 600),
         src(tmp_path, "bone", "other_body_part", 199),
         src(tmp_path, "photos", "non_medical", 220)]
    problems = M.validate(M.build(s))
    assert any("other_body_part" in p and "199" in p for p in problems)


def test_laterals_must_leave_200_sealed_after_the_viewhead_share(tmp_path):
    """300 laterals with half going to the view head leaves ~150 sealed, under the bar."""
    s = [src(tmp_path, "lat", "lateral_cxr", 300),
         src(tmp_path, "bone", "other_body_part", 250),
         src(tmp_path, "photos", "non_medical", 220)]
    assert any("lateral_cxr" in p for p in M.validate(M.build(s)))


def test_duplicate_image_ids_are_caught():
    rows = [{"image_id": "a", "rel_path": "x.png", "source": "s", "ood_type": "non_medical",
             "role": "ood_eval", "licence": "CC0"}] * 2
    assert any("duplicate image_id" in p for p in M.validate(rows, min_per_type=1))


# ---------------------------------------------------------------- the freeze (H5)
def test_freeze_writes_a_sha_that_changes_with_the_content(tmp_path, sources):
    rows = M.build(sources)
    m1 = M.freeze(rows, tmp_path / "c03.csv")
    assert m1["sha256"] and m1["n_rows"] == len(rows)
    assert (tmp_path / "c03.csv.json").exists()
    m2 = M.freeze(rows, tmp_path / "c03b.csv")
    assert m1["sha256"] == m2["sha256"], "same rows must give the same SHA"
    rows2 = rows[:-1]
    m3 = M.freeze(rows2, tmp_path / "c03c.csv")
    assert m3["sha256"] != m1["sha256"], "dropping a row must change the SHA"


def test_freeze_records_the_counts_d13_needs(tmp_path, sources):
    meta = M.freeze(M.build(sources), tmp_path / "c03.csv")
    assert set(meta["counts"]) == set(M.OOD_TYPES)
    assert meta["counts"]["lateral_cxr"]["ood_eval"] >= M.MIN_PER_TYPE
    assert "H5" in meta["note"]
    on_disk = json.load(open(str(tmp_path / "c03.csv") + ".json", encoding="utf-8"))
    assert on_disk["sha256"] == meta["sha256"]


def test_rows_are_written_in_a_stable_order(tmp_path, sources):
    """Otherwise the SHA changes between runs and H5 is unenforceable."""
    rows = M.build(sources)
    a = M.freeze(rows, tmp_path / "a.csv")["sha256"]
    b = M.freeze(list(reversed(rows)), tmp_path / "b.csv")["sha256"]
    assert a == b
