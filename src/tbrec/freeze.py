"""The freeze before the single OOT run: a manifest of everything an OOT score depends on, and the gate.

freeze.json holds the commit (code must be clean), the branches, the hypotheses, the margins and sha256 of:
rows.parquet, objective.json, every branch's params.json and every feature file the branch reads.
`check` runs before each OOT scoring, for every branch kind (boosters, trivial rankers, Perseus): the branch
must be listed, the code and every hashed file unchanged, and no OOT prediction may exist yet.
"""
import hashlib
import json
import subprocess
from pathlib import Path

from .config import ART, ROOT

FREEZE = ART / "freeze.json"
CODE = ["src", "scripts", "tests"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def branch_files(branch: str) -> list[Path]:
    from .branches import BRANCHES, TRIVIAL, resolve

    files = []
    if branch in TRIVIAL:
        files.append(ART / "features" / f"{TRIVIAL[branch][0]}.parquet")
    elif branch in BRANCHES:
        files.append(ART / "runs" / branch / "params.json")
        files += [ART / "features" / f"{stem}.parquet" for stem in resolve(branch, "oot")]
    return files


def manifest(branches: list[str]) -> dict:
    files = {"rows": ART / "rows.parquet", "objective": ART / "objective.json"}
    hashes = {k: sha256(p) for k, p in files.items() if p.exists()}
    per_branch = {b: {str(p.relative_to(ROOT)): sha256(p) for p in branch_files(b) if p.exists()} for b in branches}
    return {"commit": git("rev-parse", "HEAD"), "common": hashes, "branches_files": per_branch}


def check(branch: str) -> None:
    assert FREEZE.exists(), "OOT is run only after the freeze"
    fz = json.loads(FREEZE.read_text())
    assert branch in fz["branches"], f"{branch} is not in the frozen list"
    assert git("rev-parse", "HEAD") == fz["commit"], "HEAD differs from the frozen commit"
    assert not git("status", "--porcelain", "--", *CODE), "code changed after the freeze"
    for k, h in fz["common"].items():
        p = {"rows": ART / "rows.parquet", "objective": ART / "objective.json"}[k]
        assert sha256(p) == h, f"{p.name} changed after the freeze"
    for rel, h in fz["branches_files"][branch].items():
        assert sha256(ROOT / rel) == h, f"{rel} changed after the freeze"
    assert not (ART / "runs" / branch / "pred_oot.parquet").exists(), f"OOT already scored for {branch}"
