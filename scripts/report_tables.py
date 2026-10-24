"""Markdown tables for the README from artifacts (run once per dataset; TBREC_DATASET selects it).

  python scripts/report_tables.py --stage val --part val_B b0 r1_decay ...
  python scripts/report_tables.py --stage oot b0 r1_decay ...

Prints: the bootstrap comparison against b0 (compare.py logic) and a cost table (feature build seconds,
tuning / fit / predict seconds, predict cost per 1k rows).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tbrec.branches import BRANCHES, resolve  # noqa: E402
from tbrec.config import ART  # noqa: E402

FEAT, RUNS = ART / "features", ART / "runs"


def feature_cost(stem: str) -> float | None:
    f = FEAT / f"{stem}.cost.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    return c.get("seconds") or c.get("total_seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("branches", nargs="+")
    ap.add_argument("--stage", default="val")
    ap.add_argument("--part", default="val_B")
    a = ap.parse_args()
    part = "eval" if a.stage == "oot" else a.part
    out = ART / f"report_{a.stage}_{part}.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare.py"), "--stage", a.stage, "--part", part, "--base", "b0",
                    "--out", str(out)] + a.branches, check=True, capture_output=True)
    lines = [out.read_text(encoding="utf-8"), "", "| branch | features | feature build, s | tune, s | fit, s | predict per 1k rows, s |", "|---|---|---|---|---|---|"]
    for b in a.branches:
        cost = RUNS / b / f"cost_{a.stage}.json"
        c = json.loads(cost.read_text()) if cost.exists() else {}
        cv = json.loads((RUNS / b / "cost_val.json").read_text()) if (RUNS / b / "cost_val.json").exists() else {}
        if b in BRANCHES:
            stems = [s for s in resolve(b, a.stage)]
            fb = sum(x for x in (feature_cost(s) for s in stems) if x)
            imp = RUNS / b / f"importance_{a.stage}.csv"
            n = sum(1 for _ in imp.open()) - 1 if imp.exists() else "—"
        else:
            fb, n = None, 1
        fmt = lambda v: f"{v:.0f}" if isinstance(v, (int, float)) else "—"  # noqa: E731
        lines.append(f"| {b} | {n} | {fmt(fb)} | {fmt(cv.get('tune'))} | {fmt(c.get('fit'))} | "
                     f"{c.get('predict_per_1k_rows', float('nan')):.3f} |" if c else f"| {b} | {n} | {fmt(fb)} | — | — | — |")
    text = "\n".join(lines)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
