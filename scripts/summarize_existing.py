"""Post-freeze descriptive inventory and cost accounting; no fitting or scoring.

Only train/val labels are aggregated. OOT contributes row counts and scheduled days,
never labels or predictions. Sources are existing local tables, logs, and JSON files.
"""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: F401,E402
import polars as pl  # noqa: E402
ROOT = Path(__file__).resolve().parents[1]


def main():
    inventory = {"scope": "post-freeze descriptive inventory; no new OOT label statistics, fitting or scoring", "domains": {}}
    for domain in ("yambda", "kion", "x5"):
        art = ROOT / "artifacts" / (domain if domain != "yambda" else "")
        proc = ROOT / "data/processed" / (domain if domain != "yambda" else "")
        ev = pl.scan_parquet(proc / "events.parquet")
        meta = ev.select(pl.len().alias("events"), pl.col("uid").n_unique().alias("users_events"),
                         pl.col("item").n_unique().alias("items_events"), pl.col("day").min().alias("day_min"),
                         pl.col("day").max().alias("day_max")).collect(engine="streaming").to_dicts()[0]
        if domain == "yambda":
            meta["users_listens"] = ev.filter(pl.col("kind") == 0).select(pl.col("uid").n_unique()).collect().item()
            stages = {"train": list(range(273,287)), "val": list(range(287,294)), "oot": list(range(294,301))}
        else:
            stage = json.loads((art / "stages.json").read_text())
            stages = {"train": stage["val"]["train"], "val": stage["val"]["eval"], "oot": stage["oot"]["eval"]}
        rows = pl.scan_parquet(art / "rows.parquet")
        meta["task_rows_total"] = rows.select(pl.len()).collect().item()
        meta["task_users_total"] = rows.select(pl.col("uid").n_unique()).collect().item()
        meta["task_items_total"] = rows.select(pl.col("item").n_unique()).collect().item()
        meta["splits"] = {}
        for name, days in stages.items():
            sub = rows.filter(pl.col("day").is_in(days))
            expr = [pl.len().alias("rows"), pl.col("uid").n_unique().alias("users"), pl.col("item").n_unique().alias("items")]
            if name != "oot":
                expr.append(pl.col("y").mean().alias("positive_share"))
            meta["splits"][name] = {"days": days, **sub.select(expr).collect().to_dicts()[0]}
        meta["sources"] = [str((proc / "events.parquet").relative_to(ROOT)).replace("\\", "/"),
                           str((art / "rows.parquet").relative_to(ROOT)).replace("\\", "/"),
                           "src/tbrec/config.py" if domain == "yambda" else f"artifacts/{domain}/stages.json"]
        inventory["domains"][domain] = meta
    (ROOT / "artifacts/data_summary.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    costs = {"scope": "existing val run; feature build covers 28 prebuilt days; excludes download, unmeasured preparation, objective selection, other branches and OOT run", "stages": {}}
    for name, logfile in [("b0_features", "feat_b0.log"), ("grammar_decay", "feat_grammar_decay.log"), ("r1_features", "feat_r1_decay.log")]:
        lines = (ROOT / "logs" / logfile).read_text(encoding="utf-8").splitlines()
        done = next(s for s in reversed(lines) if s.startswith("done "))
        costs["stages"][name] = {"seconds": int(re.search(r"(\d+)s$", done)[1]), "source": "logs/" + logfile, "source_first": lines[0], "source_last": done}
    lines = (ROOT / "logs/search_r1_decay.log").read_text(encoding="utf-8").splitlines()
    a, b = lines[0], lines[-1]
    def clock(s):
        h,m,sec = map(int,s[:8].split(":"))
        return h*3600+m*60+sec
    costs["stages"]["search"] = {"seconds": (clock(b)-clock(a)) % 86400, "source": "logs/search_r1_decay.log", "source_first": a, "source_last": b, "measurement": "timestamp interval; excludes process startup"}
    for branch in ("b0", "r1_decay"):
        costs[branch] = {"source": f"artifacts/runs/{branch}/cost_val.json", **json.loads((ROOT / f"artifacts/runs/{branch}/cost_val.json").read_text()), "params_source": f"artifacts/runs/{branch}/params.json"}
    costs["preparation_seconds"] = None
    costs["b0_measured_subtotal_seconds"] = costs["stages"]["b0_features"]["seconds"] + sum(costs["b0"][k] for k in ("load","tune","fit","predict"))
    costs["r1_measured_subtotal_seconds"] = sum(s["seconds"] for s in costs["stages"].values()) + sum(costs["r1_decay"][k] for k in ("load","tune","fit","predict"))
    (ROOT / "artifacts/cost_summary.json").write_text(json.dumps(costs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    print(json.dumps(costs, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
