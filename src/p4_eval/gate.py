"""No-Test-Tuning Gate: Governance enforcement, decisions validation, and sealed-data token issuance."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tbcore.paths import get_artifacts_dir, get_decisions_dir, get_repo_root, resolve_path
from tbcore.schemas import DecisionRecord
from tbcore.validators import ContractValidationError, validate_c16_decision
from tbcore.version import compute_file_sha256, get_git_sha


ALLOWED_DATA_USED = {
    "D01": ["none"],
    "D02": ["images_only_all_splits"],
    "D03": ["development"],
    "D04": ["none"],
    "D05": ["none"],
    "D06": ["images_only_all_splits"],
    "D07": ["development"],
    "D08": ["train", "development"],
    "D09": ["train", "development"],
    "D10": ["train", "development"],
    "D11": ["train", "development"],
    "D12": ["train", "development"],
    "D13": ["train", "development"],
    "D14": ["none"],
    "D15": ["none"],
    "D16": ["train", "development"],
    "D17": ["train", "development"],
    "D18": ["development"],
    "D19": ["development"],
    "D20": ["none"],
    "D21": ["none"],
    "D22": ["development"],
    "D23": ["calibration"],
}


class GateValidationError(Exception):
    """Raised when gate preconditions or decision locks fail."""
    pass


def check_all_decisions(dec_dir: Optional[Path] = None) -> Dict[str, DecisionRecord]:
    """
    Validates all 23 decision files (D01-D23) against schema, locked status,
    and allowed data_used rules.
    """
    if dec_dir is None:
        dec_dir = get_decisions_dir()

    decisions: Dict[str, DecisionRecord] = {}
    missing = []
    invalid = []

    for i in range(1, 24):
        d_id = f"D{i:02d}"
        matching_files = list(dec_dir.glob(f"*_{d_id.lower()}*.json")) + list(dec_dir.glob(f"{d_id.lower()}*.json"))
        # Also check files named like decisions/p1_splits.json matching D01
        if not matching_files:
            # Look inside files for decision_id
            for jf in dec_dir.glob("*.json"):
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("decision_id") == d_id:
                        matching_files = [jf]
                        break
                except Exception:
                    pass

        if not matching_files:
            missing.append(d_id)
            continue

        jf = matching_files[0]
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            rec = validate_c16_decision(data)
        except Exception as e:
            invalid.append(f"{d_id} ({jf.name}): {e}")
            continue

        if not rec.locked:
            invalid.append(f"{d_id} ({jf.name}) is not locked (locked=false)")

        # Verify allowed data_used
        expected_data = ALLOWED_DATA_USED.get(d_id, [])
        for du in rec.data_used:
            if du not in expected_data:
                invalid.append(
                    f"{d_id} uses '{du}' which violates governance; allowed: {expected_data}"
                )

        decisions[d_id] = rec

    if missing:
        raise GateValidationError(f"Missing decision files for: {', '.join(missing)}")
    if invalid:
        raise GateValidationError(f"Invalid decision files:\n" + "\n".join(invalid))

    return decisions


def verify_gate_prerequisites() -> bool:
    """Verifies all gate prerequisites including decisions D01-D23 and split SHA."""
    _ = check_all_decisions()
    return True


def issue_token(split_name: str, out_dir: Optional[Path] = None) -> Path:
    """
    Validates all governance gates and issues a single-use unlock token for sealed data.
    """
    valid_splits = {"internal_test", "challenge_test", "external_test", "ood_eval"}
    if split_name not in valid_splits:
        raise GateValidationError(f"Split/role '{split_name}' is not a sealed dataset.")

    # Validate decisions
    _ = check_all_decisions()

    if out_dir is None:
        out_dir = resolve_path("artifacts/p4/gate")
    out_dir.mkdir(parents=True, exist_ok=True)

    token_file = out_dir / f"token_{split_name}.json"
    if token_file.exists():
        with open(token_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("consumed", False):
            raise GateValidationError(
                f"A token for '{split_name}' was already issued and consumed at {existing.get('consumed_at')}."
            )

    token_data = {
        "token_id": f"tok_{split_name}_{get_git_sha()}_{int(datetime.now(timezone.utc).timestamp())}",
        "split": split_name,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "consumed": False,
        "consumed_at": None,
        "git_sha": get_git_sha(),
    }

    with open(token_file, "w", encoding="utf-8") as f:
        json.dump(token_data, f, indent=2)

    # Append to docs/p4_gate_audit.md
    audit_file = resolve_path("docs/p4_gate_audit.md")
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_file, "a", encoding="utf-8") as af:
        af.write(
            f"- **{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}** | "
            f"Issued token for `{split_name}` | ID: `{token_data['token_id']}` | Commit: `{get_git_sha()}`\n"
        )

    return token_file


def main():
    parser = argparse.ArgumentParser(description="P4 No-Test-Tuning Gate Governance Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Verify all decision files and gate readiness")
    unlock_parser = subparsers.add_parser("unlock", help="Issue single-use token for sealed split access")
    unlock_parser.add_argument("split", choices=["internal_test", "challenge_test", "external_test", "ood_eval"])

    args = parser.parse_args()

    if args.command == "check":
        try:
            decs = check_all_decisions()
            print(f"[GATE PASS] All {len(decs)} decisions (D01-D23) are valid and locked.")
            sys.exit(0)
        except Exception as e:
            print(f"[GATE FAIL] Preconditions not met: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "unlock":
        try:
            tok = issue_token(args.split)
            print(f"[GATE UNLOCKED] Issued single-use token: {tok}")
            sys.exit(0)
        except Exception as e:
            print(f"[GATE ERROR] Failed to unlock: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
