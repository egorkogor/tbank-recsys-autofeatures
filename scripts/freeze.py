"""Freeze the pipeline before the single OOT run.

Writes artifacts/freeze.json: commit hash (the working tree must be clean), branches that go to OOT,
decision margins, the hypotheses and their branch pairs. After this file exists run_branch.py --stage oot
is allowed, once per branch.

  python scripts/freeze.py b0 r1_decay r1_window b0_ials b0_perseus full r3_perseus ...
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbrec.config import ART, STAGES  # noqa: E402
from tbrec.metrics import MARGIN  # noqa: E402

HYPOTHESES = {
    "H1 R1 (decay search) is not worse than B0": ("r1_decay", "b0"),
    "H2 Perseus alone is worse than B0": ("r3_perseus", "b0"),
    "H3 B0 + Perseus score is better than B0": ("b0_perseus", "b0"),
    "H4 decay search is not worse than window search at an equal count": ("r1_decay_k56", "r1_window_k56"),
    "H6 per-type history limits beat recency truncation (Perseus)": ("r3_perseus_typed", "r3_perseus"),
}


def main():
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "tests"], cwd=root,
                           capture_output=True, text=True).stdout.strip()
    assert not dirty, f"commit code before the freeze:\n{dirty}"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    f = ART / "freeze.json"
    assert not f.exists(), "already frozen"
    f.write_text(json.dumps({"commit": commit, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "branches": sys.argv[1:],
                             "margins": MARGIN, "stages": STAGES, "hypotheses": HYPOTHESES}, indent=1))
    print("frozen at", commit, "branches:", sys.argv[1:])


if __name__ == "__main__":
    main()
