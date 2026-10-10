"""I3 prep: kill-test analysis, paste parsing and the runbook append helper (no Kaggle needed)."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from p3_ops import kill_test_log as ktl  # noqa: E402

I3_ROW = re.compile(r"\| *(\d{4}-\d\d-\d\d) *\| *Kaggle")          # what the audit's I3 check looks for


def _out(start, last, epochs=20, done=True):
    lines = ["env=kaggle dir=/kaggle/working/checkpoints/r remote=gdrive:tb_checkpoints/r",
             f"start at epoch {start}" + (" (fresh)" if start == 0 else f" (resumed from ckpt_epoch{start - 1:04d}.pt)")]
    lines += [f"epoch {e:3d}  val_loss 1.0" for e in range(start, last + 1)]
    return lines + (["done: epochs ..."] if done else [])


def _before(last):
    return _out(0, last, done=False)


@pytest.mark.parametrize("last, start, lost, ok", [(4, 5, 1, True), (4, 4, 2, False), (4, 3, 3, False)])
def test_epochs_lost_rule(last, start, lost, ok):
    r = ktl.analyse("A", _before(last), _out(start, 19), 20, platform="kaggle", day="2026-10-11")
    assert r["epochs_lost"] == lost and r["pass"] is ok and r["finished"]


def test_fresh_restart_and_finished_before_kill_fail():
    r = ktl.analyse("B", None, _out(0, 19), 20, last_printed=4, platform="kaggle")
    assert not r["pass"] and "nothing was resumed" in r["problems"][0]
    r = ktl.analyse("B", None, ["start at epoch 20 (resumed from ckpt_epoch0019.pt)", "done: x"], 20,
                    last_printed=19, platform="kaggle")
    assert not r["pass"] and "already finished" in r["problems"][0]
    r = ktl.analyse("A", _before(4), _out(5, 12, done=False), 20)            # resumed but did not finish
    assert not r["pass"] and not r["finished"]


def _paste(*results):
    return "\n".join(["| table |"] + [ktl.TAG + json.dumps(r) for r in results])


def test_rows_only_part_b_on_kaggle_counts_for_i3():
    a = ktl.analyse("A", _before(4), _out(5, 19), 20, platform="kaggle", gpu="Tesla T4", run="ka", day="2026-10-11",
                    ckpts_on_drive=["ckpt_epoch0003.pt", "ckpt_epoch0004.pt"])
    b = ktl.analyse("B", None, _out(5, 19), 20, last_printed=4, platform="kaggle", gpu="Tesla T4", run="kb",
                    day="2026-10-11")
    ra, rb = (ktl.runbook_row(r) for r in ktl.parse_paste(_paste(a, b)))
    assert not I3_ROW.search(ra) and "supporting test" in ra                  # Part A never satisfies I3 alone
    assert I3_ROW.search(rb) and "| Stop session |" in rb and "| yes |" in rb
    local_b = ktl.analyse("B", None, _out(5, 19), 20, last_printed=4, platform="local", day="2026-10-11")
    with pytest.raises(ktl.Refused, match="not Kaggle"):
        ktl.runbook_row(local_b)


@pytest.mark.parametrize("text, match", [("", "no KILLTEST_RESULT"), ("hello", "no KILLTEST_RESULT"),
                                         (ktl.TAG + "{bad", "malformed"), (ktl.TAG + '{"part": "A"}', "without")])
def test_paste_refusals(text, match):
    with pytest.raises(ktl.Refused, match=match):
        ktl.parse_paste(text)


def test_append_to_runbook_copy(tmp_path):
    rb = tmp_path / "runbook.md"
    shutil.copy(REPO / "docs/p3_kaggle_runbook.md", rb)
    before = rb.read_text(encoding="utf-8")
    b = ktl.analyse("B", None, _out(5, 19), 20, last_printed=4, platform="kaggle", gpu="Tesla T4", run="kb",
                    day="2026-10-11")
    p = tmp_path / "paste.txt"
    p.write_text(_paste(b))
    real = (REPO / "docs/p3_kaggle_runbook.md").read_text(encoding="utf-8")
    assert ktl.main(["--paste", str(p)], runbook=rb) == 0 and rb.read_text(encoding="utf-8") == before   # show only
    assert ktl.main(["--paste", str(p), "--write"], runbook=rb) == 0
    after = rb.read_text(encoding="utf-8")
    sec = after[after.index("## 10."):]
    table = [l for l in sec.splitlines() if l.startswith("|")]
    assert I3_ROW.search(table[-1]) and "kb" in table[-1]                     # appended inside the log table
    assert (REPO / "docs/p3_kaggle_runbook.md").read_text(encoding="utf-8") == real   # real runbook untouched


def test_notebook_is_plain_python_and_prints_results_last():
    nb = json.loads((REPO / "notebooks/p3/kill_resume_test.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        compile(src, "<cell>", "exec")
    assert "ktl.TAG" in code[-1] and "p.kill()" in code[0]
