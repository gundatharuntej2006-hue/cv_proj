"""TEST-ONLY stand-in for P4's tbcore.disclaimer (single source of the disclaimer and scope texts).

The texts below are placeholders for tests, NOT the project's wording; production code imports the
real module from tbcore and refuses to build a PDF without it.
"""
from __future__ import annotations

import sys
import types

DISCLAIMER = ("TEST-DISCLAIMER: decision support only, not a diagnosis (placeholder text for tests; the real "
              "text is owned by P4 in tbcore).")
SCOPE_HONESTY = ("TEST-SCOPE: validated only on public research datasets (placeholder text for tests; the real "
                 "text is owned by P4 in tbcore).")


def install(monkeypatch) -> None:
    """Make ``from tbcore import disclaimer`` return this stand-in for the duration of a test."""
    if "tbcore" in sys.modules and hasattr(sys.modules["tbcore"], "__path__"):
        return                                           # the real tbcore is installed: use it
    pkg = types.ModuleType("tbcore")
    mod = types.ModuleType("tbcore.disclaimer")
    mod.DISCLAIMER, mod.SCOPE_HONESTY = DISCLAIMER, SCOPE_HONESTY
    pkg.disclaimer = mod
    monkeypatch.setitem(sys.modules, "tbcore", pkg)
    monkeypatch.setitem(sys.modules, "tbcore.disclaimer", mod)
