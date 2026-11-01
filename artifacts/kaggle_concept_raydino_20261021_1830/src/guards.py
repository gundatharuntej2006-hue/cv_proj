# -*- coding: utf-8 -*-
"""Hard guards. Import these first in every notebook and script.

The locked partition must never be enumerated, loaded, scored or joined. These guards make
that a runtime error rather than a convention.
"""
from pathlib import Path

LOCKED_PARTITION = "internal_test"
LOCKED_TOKENS = ("internal_test", "internaltest", "internal-test")


class LockedPartitionError(RuntimeError):
    """Raised whenever anything would touch the reserved partition."""


def assert_path_allowed(path):
    """Refuse any path that looks like locked data."""
    s = str(path).lower().replace("\\", "/")
    for tok in LOCKED_TOKENS:
        if tok in s:
            raise LockedPartitionError(
                "refusing to touch a path containing %r: %s" % (tok, path))
    return Path(path)


def safe_listdir(root, pattern="*"):
    """Enumerate a directory, refusing locked subtrees."""
    root = assert_path_allowed(root)
    out = []
    for p in sorted(Path(root).glob(pattern)):
        try:
            assert_path_allowed(p)
        except LockedPartitionError:
            continue                      # silently skip; never surface locked names
        out.append(p)
    return out


def drop_locked_rows(df, split_col="split"):
    """Remove locked rows from a split manifest and prove they are gone.

    Returns (usable_df, n_locked_counted). The count is metadata about how many rows were
    excluded; no locked row's id, label or path is retained.
    """
    if split_col not in df.columns:
        raise ValueError("no %r column to filter on" % split_col)
    mask = df[split_col].astype(str).str.lower().isin(
        [t for t in LOCKED_TOKENS])
    n_locked = int(mask.sum())
    usable = df.loc[~mask].copy()
    remaining = set(usable[split_col].astype(str).str.lower())
    for tok in LOCKED_TOKENS:
        if tok in remaining:
            raise LockedPartitionError("locked rows survived the filter")
    return usable, n_locked


def assert_no_locked_in(obj, where="object"):
    """Structural check that a payload carries no locked-partition content."""
    import json
    s = json.dumps(obj, default=str).lower()
    for tok in LOCKED_TOKENS:
        if tok in s:
            # the bare partition name is allowed in prose and in count fields only
            allowed = ('"%s"' % tok, "n_%s_rows" % tok, "%s_policy" % tok,
                       "locked_partition")
            if not any(a in s for a in allowed):
                raise LockedPartitionError("locked content in %s" % where)
    return True


def assert_shape(arr, expected, name):
    got = tuple(getattr(arr, "shape", ()))
    exp = tuple(expected)
    if len(exp) != len(got) or any(
            e is not None and e != g for e, g in zip(exp, got)):
        raise AssertionError("%s has shape %s, expected %s" % (name, got, exp))
    return True


def assert_ids_aligned(ids_a, ids_b, name_a="a", name_b="b"):
    """Row alignment must be proved by stable IDs, never assumed from order."""
    a, b = list(ids_a), list(ids_b)
    if len(a) != len(b):
        raise AssertionError("%s has %d rows, %s has %d" % (name_a, len(a), name_b, len(b)))
    if len(set(a)) != len(a):
        raise AssertionError("%s contains duplicate ids" % name_a)
    if a != b:
        n = sum(1 for x, y in zip(a, b) if x != y)
        raise AssertionError("%s and %s disagree on %d of %d ids"
                             % (name_a, name_b, n, len(a)))
    return True


def assert_target_mapping(mapping):
    """Refuse an unknown or ambiguous target-label definition."""
    required = {"positive_categories", "negative_categories", "label_column",
                "description"}
    missing = required - set(mapping or {})
    if missing:
        raise ValueError("target mapping is missing %s - refusing to train on an "
                         "undefined target" % sorted(missing))
    pos = set(mapping["positive_categories"])
    neg = set(mapping["negative_categories"])
    if not pos or not neg:
        raise ValueError("target mapping has an empty class")
    overlap = pos & neg
    if overlap:
        raise ValueError("categories in both classes: %s" % sorted(overlap))
    return True
