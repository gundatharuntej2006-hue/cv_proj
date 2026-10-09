"""The ONE place where P3 touches P4's `tbcore` (FIX 4).

P4 has not published tbcore yet. Every tbcore name P3 relies on is listed in ASSUMED_API; when P4's real API
lands, change the names HERE only (and run tests/p3_ops/test_tbcore_adapter.py, which fails loudly if the
installed tbcore does not provide them). Behaviour while tbcore is absent:

* seed            -> local fallback seeding (recorded as such by callers)
* sealed guard    -> REFUSE (fail closed)
* validators      -> report "not validated by tbcore" (callers apply their own contract checks)
* disclaimer text -> REFUSE (no PDF without the single-source texts)
"""
from __future__ import annotations

import importlib
import importlib.util

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
                _get(module, n)
            except TbcoreUnavailable:
                missing.append(f"{module}.{n}")
    return missing


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
    _get("tbcore.guard", "require_unlocked")(split=split, dataset=dataset, purpose=purpose)


def validate(contract_id: str, path) -> bool:
    """Run tbcore's validator; True if run (raises on invalid), False if tbcore is absent. If tbcore is installed
    but the assumed name is wrong, raises TbcoreUnavailable: never silently skipped."""
    if not tbcore_available():
        return False
    _get("tbcore.validators", "validate")(contract_id, path)
    return True


def disclaimer_texts() -> tuple[str, str]:
    """(DISCLAIMER, SCOPE_HONESTY), the single-source texts. Raises TbcoreUnavailable (fail closed)."""
    return str(_get("tbcore.disclaimer", "DISCLAIMER")), str(_get("tbcore.disclaimer", "SCOPE_HONESTY"))
