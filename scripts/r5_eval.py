"""R5: evaluate one round of agent-proposed features.

Proposal file (written by the agent): [{"name": "...", "why": "..."}, ...]
A name is a grammar name (KEY__FILTER__AGG__WEIGHT, any weighting) or a composite of two grammar names:
ratio(a,b) = a / (b + 1e-6), diff(a,b) = a - b.

Every component passes the truncation-invariance check first. Then, on the selection rows of R1
(training days of the "sel" users, val half A for the decision), each proposal is added alone to the
current set (R1 selection + earlier accepted proposals); accepted if GAUC on val half A rises by at least
max(0.0005, 2 x seed noise of the current set). Everything is logged to artifacts/r5_state.json.

  python scripts/r5_eval.py --round 1 --proposals logs/experiment/r5_round1.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from tbrec.branches import BASE_COLS  # noqa: E402
from tbrec.config import ART  # noqa: E402
from tbrec.data import load_events  # noqa: E402
from tbrec.features import build, parse, valid  # noqa: E402
from tbrec.leakcheck import invariance_errors, small_log  # noqa: E402
from tbrec.model import FEAT, ROWS, fit, gauc, predict  # noqa: E402
from tbrec.config import STAGES  # noqa: E402

STATE = ART / "r5_state.json"
PARAMS = {"learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200}
COMPOSITE = re.compile(r"^(ratio|diff)\(([A-Za-z0-9_]+),([A-Za-z0-9_]+)\)$")


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def components(name: str) -> list[str]:
    m = COMPOSITE.match(name.replace(" ", ""))
    return [m[2], m[3]] if m else [name]


def check_name(n: str) -> str | None:
    try:
        k, f, a, w = parse(n)
    except Exception:
        return "not a grammar name"
    return None if valid(k, f, a, w) else "invalid combination"


def materialise(df: pl.DataFrame, name: str) -> pl.Expr:
    m = COMPOSITE.match(name.replace(" ", ""))
    if not m:
        return pl.col(name)
    a, b = pl.col(m[2]), pl.col(m[3])
    return (a / (b + 1e-6)) if m[1] == "ratio" else (a - b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--proposals", required=True)
    a = ap.parse_args()
    t0 = time.time()
    state = json.loads(STATE.read_text()) if STATE.exists() else {
        "start": json.loads((ART / "r1_decay_selected.json").read_text())["names"], "accepted": [], "journal": []}
    props = json.loads(Path(a.proposals).read_text())
    current = state["start"] + state["accepted"]
    entries, comps = [], set()
    for p in props:
        name = p["name"].replace(" ", "")
        errs = [f"{c}: {e}" for c in components(name) if (e := check_name(c))]
        if name in current:
            errs.append("already in the set")
        entries.append({"round": a.round, "name": name, "why": p.get("why", ""), "error": "; ".join(errs) or None})
        if not errs:
            comps |= set(components(name))
    ev = load_events()
    leaks = invariance_errors(small_log(ev), sorted(comps)) if comps else []
    for e in entries:
        if e["error"] is None and any(c in leaks for c in components(e["name"])):
            e["error"] = "fails truncation invariance"
    ok = [e for e in entries if e["error"] is None]
    log(f"round {a.round}: {len(props)} proposals, {len(ok)} pass checks")
    # selection rows = rows of the R1 grammar table; grammar_decay columns reused, the rest computed
    g = pl.read_parquet(FEAT / "grammar_decay.parquet")
    rows = pl.read_parquet(ROWS).select("row", "uid", "item", "artist", "album", "day", "t", "y", "val_half", "sel")
    rows = rows.join(g.select("row"), on="row", how="semi")
    need_cur = sorted({c for n in current for c in components(n)})
    have = set(g.columns)
    todo = sorted((comps | set(need_cur)) - have)
    df = rows.join(g.select(["row"] + sorted((comps | set(need_cur)) & have)), on="row", how="left")
    if todo:
        x = build(rows.drop("val_half", "sel"), ev, todo, log=lambda s: None).select(["row"] + todo)
        df = df.join(x, on="row", how="left")
    df = df.join(pl.read_parquet(FEAT / "b0.parquet", columns=["row"] + BASE_COLS), on="row", how="left")
    allnames = list(dict.fromkeys(current + [e["name"] for e in ok]))
    df = df.with_columns([materialise(df, n).cast(pl.Float32).alias(n) for n in allnames if n not in df.columns])
    tr = df.filter(pl.col("day").is_in(STAGES["val"]["train"]) & pl.col("sel"))
    va = df.filter(pl.col("day").is_in(STAGES["val"]["eval"]) & (pl.col("val_half") == "A"))
    cols = BASE_COLS + current
    seeds = [gauc(va, predict(b, va, cols)) for b in (fit(tr, cols, {**PARAMS, "seed": s}, va=va) for s in (1, 2, 3))]
    base, noise = float(np.mean(seeds)), float(np.std(seeds))
    thr = max(0.0005, 2 * noise)
    log(f"current set {len(current)}: gauc {base:.4f} (seed sd {noise:.4f}), threshold +{thr:.4f}")
    for e in ok:
        gs = [gauc(va, predict(b, va, cols + [e["name"]])) for b in (fit(tr, cols + [e["name"]], {**PARAMS, "seed": s}, va=va) for s in (1, 2))]
        e["gauc"], e["delta"] = float(np.mean(gs)), float(np.mean(gs) - base)
        e["accepted"] = e["delta"] >= thr
        log(f"  {e['name']}: {e['delta']:+.4f} {'ACCEPT' if e['accepted'] else ''}")
    acc = [e["name"] for e in ok if e.get("accepted")]
    if acc:
        b = fit(tr, cols + acc, {**PARAMS, "seed": 1}, va=va)
        joint = gauc(va, predict(b, va, cols + acc))
        log(f"joint with {len(acc)} accepted: {joint:.4f} ({joint - base:+.4f})")
    else:
        joint = base
    state["accepted"] += acc
    state["journal"] += entries
    state.setdefault("rounds", []).append({"round": a.round, "base_gauc": base, "seed_sd": noise, "threshold": thr,
                                           "accepted": acc, "joint_gauc": joint, "seconds": time.time() - t0})
    STATE.write_text(json.dumps(state, indent=1, default=float))
    names = state["start"] + state["accepted"]
    gnames = sorted({c for n in names for c in components(n)})
    (ART / "r5_accepted.json").write_text(json.dumps({"names": names, "grammar_names": gnames}, indent=1))


if __name__ == "__main__":
    main()
