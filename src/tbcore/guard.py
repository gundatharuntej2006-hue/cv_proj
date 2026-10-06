"""Sealed-split guard enforcing no-test-tuning governance on test and OOD datasets."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from tbcore.paths import get_repo_root, resolve_path

SEALED_SPLITS = {"internal_test", "challenge_test", "external_test"}
SEALED_ROLES = {"ood_eval"}


class SealedSplitAccessError(PermissionError):
    """Raised when attempting to access sealed test data without an active unlock token."""
    pass


def check_split_access(
    split_name: str,
    role: Optional[str] = None,
    purpose: Optional[str] = None,
    token_dir: Optional[Path] = None,
    consume_token: bool = False
) -> bool:
    """
    Checks whether access to a given dataset split or role is permitted under governance rules.
    
    Rules:
    - 'train', 'development', 'calibration' splits and unsealed roles are always accessible.
    - If purpose == 'dedup', image loading without labels is permitted for all splits.
    - Sealed splits ('internal_test', 'challenge_test', 'external_test') and role 'ood_eval'
      strictly require a valid, unconsumed single-use token issued by the P4 gate.
    """
    # Exemption for dedup purpose (images only, no labels)
    if purpose == "dedup":
        return True

    is_sealed = (split_name in SEALED_SPLITS) or (role in SEALED_ROLES)
    if not is_sealed:
        return True

    target_key = role if (role in SEALED_ROLES) else split_name

    # Look for gate token
    if token_dir is None:
        token_dir = resolve_path("artifacts/p4/gate")

    token_file = token_dir / f"token_{target_key}.json"
    if not token_file.exists():
        raise SealedSplitAccessError(
            f"Access to sealed split/role '{target_key}' is blocked. "
            f"No gate token found at {token_file}. "
            f"Run 'python -m p4_eval.gate unlock {target_key}' after locking all decisions D01-D23."
        )

    try:
        with open(token_file, "r", encoding="utf-8") as f:
            token_data = json.load(f)
    except Exception as e:
        raise SealedSplitAccessError(f"Failed to read gate token {token_file}: {e}")

    if token_data.get("split") != target_key:
        raise SealedSplitAccessError(
            f"Token mismatch: token is for '{token_data.get('split')}', requested '{target_key}'."
        )

    if token_data.get("consumed", False) and not consume_token:
        raise SealedSplitAccessError(
            f"Gate token for '{target_key}' has already been consumed at {token_data.get('consumed_at')}."
        )

    if consume_token and not token_data.get("consumed", False):
        token_data["consumed"] = True
        token_data["consumed_at"] = datetime.now(timezone.utc).isoformat()
        with open(token_file, "w", encoding="utf-8") as f:
            json.dump(token_data, f, indent=2)

    return True
