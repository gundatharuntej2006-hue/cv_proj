"""The ONE place where P3 touches P4's `tbcore` (FIX 4).

The team tbcore package is supported here, along with the earlier test interface.
ASSUMED_API names describe P3's logical operations; ALIASES resolves P4's actual
names. P4 has no C11 validator, so strict producer checks remain explicit.
Behaviour while tbcore is absent:

* seed            -> local fallback seeding (recorded as such by callers)
* sealed guard    -> REFUSE (fail closed)
* validators      -> report "not validated by tbcore" (callers apply their own contract checks)
* disclaimer text -> REFUSE (no PDF without the single-source texts)
"""
from __future__ import annotations

import importlib
import importlib.util
from contextlib import contextmanager
from contextvars import ContextVar

_sealed_scope = ContextVar("p3_sealed_scope", default=None)

ASSUMED_API: dict[str, tuple[str, ...]] = {
    "tbcore.seed": ("set_all",),                      # named in README 4.6
    "tbcore.guard": ("require_unlocked",),            # assumed name (README 4.1 describes the guard, not its API)
    "tbcore.validators": ("validate",),               # assumed: validate(contract_id, path)
    "tbcore.disclaimer": ("DISCLAIMER", "SCOPE_HONESTY"),   # assumed: single-source texts (README 3, P4-01)
}


class TbcoreUnavailable(RuntimeError):
    """tbcore (or the assumed name) is not available."""


def tbcore_available() -> bool:
    import sys
    if "tbcore" in sys.modules:                  # already imported (or explicitly blocked with None)
        return sys.modules["tbcore"] is not None
    try:
        return importlib.util.find_spec("tbcore") is not None
    except (ImportError, ValueError):
        return False


def _get(module: str, name: str):
    if not tbcore_available():
        raise TbcoreUnavailable("tbcore is not available")
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise TbcoreUnavailable(f"{module} is not available") from e
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise TbcoreUnavailable(f"{module}.{name} is not provided by the installed tbcore (update ASSUMED_API)") from e


def check_api() -> list[str]:
    """Names in ASSUMED_API that the installed tbcore does not provide ([] = all present)."""
    missing = []
    for module, names in ASSUMED_API.items():
        for n in names:
            try:
                _resolve(module, n)
            except TbcoreUnavailable:
                missing.append(f"{module}.{n}")
    return missing


ALIASES = {
    ("tbcore.guard", "require_unlocked"): "check_split_access",
    ("tbcore.disclaimer", "DISCLAIMER"): "DISCLAIMER_TEXT",
    ("tbcore.disclaimer", "SCOPE_HONESTY"): "SCOPE_HONESTY_TEXT",
}


def _resolve(module: str, name: str):
    try:
        return _get(module, name)
    except TbcoreUnavailable:
        if (module, name) in ALIASES:
            return _get(module, ALIASES[module, name])
        if (module, name) == ("tbcore.validators", "validate"):
            # P4 publishes individual contract validators, not a generic dispatcher.
            _get(module, "validate_c10_detections")
            _get(module, "validate_c15_prediction_log")
            return _validate_current
        raise


def _validate_current(contract_id, path):
    names = {"C10": "validate_c10_detections", "C15": "validate_c15_prediction_log",
             "C16": "validate_c16_decision", "C14": "validate_c14_case_result"}
    if contract_id == "C11":
        # P4 has no C11 validator. The producer always performs its strict C11
        # checks after this call; do not recurse back into validate_table here.
        return False
    if contract_id not in names:
        raise TbcoreUnavailable(f"No tbcore validator for {contract_id}")
    _get("tbcore.validators", names[contract_id])(path)
    return True


# --- the four uses --------------------------------------------------------------------------------

def seed_all(seed: int) -> bool:
    """tbcore.seed.set_all(seed). Returns False when tbcore is absent (caller seeds locally)."""
    try:
        fn = _get("tbcore.seed", "set_all")
    except TbcoreUnavailable:
        return False
    fn(seed)
    return True


def require_unlocked(split: str, dataset: str, purpose: str) -> None:
    """The sealed-split guard. Raises TbcoreUnavailable if the guard cannot be reached (fail closed);
    whatever tbcore.guard raises (no token) propagates."""
    if _sealed_scope.get() == (dataset, split):
        return
    try:
        fn = _get("tbcore.guard", "require_unlocked")
    except TbcoreUnavailable:
        # Never use the images-only dedup exemption for a P3 scoring operation.
        if purpose == "dedup":
            raise ValueError("P3 scoring cannot use the dedup guard exemption")
        _get("tbcore.guard", "check_split_access")(split_name=split, purpose=purpose)
    else:
        fn(split=split, dataset=dataset, purpose=purpose)


def validate(contract_id: str, path) -> bool:
    """Run tbcore's validator; True if run (raises on invalid), False if tbcore is absent. If tbcore is installed
    but the assumed name is wrong, raises TbcoreUnavailable: never silently skipped."""
    if not tbcore_available():
        return False
    result = _resolve("tbcore.validators", "validate")(contract_id, path)
    return result is not False


def disclaimer_texts() -> tuple[str, str]:
    """(DISCLAIMER, SCOPE_HONESTY), the single-source texts. Raises TbcoreUnavailable (fail closed)."""
    return str(_resolve("tbcore.disclaimer", "DISCLAIMER")), str(_resolve("tbcore.disclaimer", "SCOPE_HONESTY"))


def legacy_schema(name):
    """Shared legacy data types, kept behind the same P4 boundary."""
    return _get("tbcore.schemas", name)


def version_info() -> dict:
    """P4 version stamps for reports, through the shared compatibility boundary."""
    return dict(_get("tbcore.version", "get_version_info")())


@contextmanager
def sealed_scope(dataset, split, token_dir, git_sha):
    """Consume an existing P4 token once. Never issues tokens or unlocks decisions.

    An exclusive claim file survives failure, so concurrent or repeated scoring
    cannot reuse the token. Only this execution context can read after consumption.
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    if split not in ("internal_test", "challenge_test", "external_test"):
        raise ValueError("Not a sealed split")
    root = Path(token_dir)
    token_path = root / f"token_{split}.json"
    data = json.loads(token_path.read_text(encoding="utf-8"))
    if data.get("split") != split or data.get("consumed") or not data.get("token_id"):
        raise PermissionError("Invalid or consumed P4 token")
    if data.get("git_sha") != git_sha:
        raise PermissionError("P4 token does not match the current code revision")
    claim = root / f"p3_claim_{split}.json"
    with claim.open("x", encoding="utf-8") as f:
        json.dump({"token_id": data["token_id"], "git_sha": git_sha, "dataset": dataset}, f)
    # Re-read under our permanent claim; fail closed if another consumer raced.
    current = json.loads(token_path.read_text(encoding="utf-8"))
    if current != data:
        raise PermissionError("P4 token changed during claim")
    data.update(consumed=True, consumed_at=datetime.now(timezone.utc).isoformat())
    temporary = token_path.with_suffix(".p3.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(token_path)
    context = _sealed_scope.set((dataset, split))
    try:
        yield data
    finally:
        _sealed_scope.reset(context)


def validate_gate():
    from p4_eval.gate import check_all_decisions
    return check_all_decisions()
