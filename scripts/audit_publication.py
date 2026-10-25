"""Read-only publication audit; report locations and categories, never matching values."""
import json
import re
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "credential_keyword": re.compile(r"token|api_key|apikey|secret|password|sk-|ghp_|hf_", re.I),
    "local_path": re.compile(r"C:[/\\]Users|D:[/\\]PdM_Projects|AppData", re.I),
    "personal_project": re.compile(r"(?!)"),
    "credential_shape": re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})"),
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def scan(data):
    if b"\x00" in data:
        return {}
    lines = data.decode("utf-8", errors="replace").splitlines()
    return {name: [i for i, s in enumerate(lines, 1) if rx.search(s)] for name, rx in PATTERNS.items() if any(rx.search(s) for s in lines)}


def main():
    files = git("ls-files", "-z").decode().split("\x00")[:-1]
    current = {}; oversized = []; raw_candidates = []; csv_headers = {}
    for name in files:
        p = ROOT / name
        found = scan(p.read_bytes())
        if found: current[name] = found
        if p.stat().st_size > 5 * 1024 * 1024: oversized.append(name)
        if p.suffix.lower() in (".parquet", ".jsonl", ".npy", ".pkl"): raw_candidates.append(name)
        if p.suffix.lower() == ".csv":
            header = p.read_text(encoding="utf-8-sig").splitlines()[0]
            csv_headers[name] = header
            if header != "feature,gain": raw_candidates.append(name)
    # Scan actual added/removed patch lines, keeping only filenames, commits and categories.
    patch = git("log", "-p", "--all", "--no-ext-diff", "--format=COMMIT:%H").decode("utf-8", errors="replace")
    history = {}; commit = ""; filename = ""
    for line in patch.splitlines():
        if line.startswith("COMMIT:"): commit = line[7:]
        elif line.startswith("+++ b/"): filename = line[6:]
        elif line.startswith("--- a/"): filename = line[6:]
        elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            matches = [k for k, rx in PATTERNS.items() if rx.search(line[1:])]
            if matches:
                key = commit + ":" + filename
                history.setdefault(key, set()).update(matches)
    objects = git("rev-list", "--objects", "--all").decode().splitlines()
    obj_input = "\n".join(objects) + "\n"
    stats = subprocess.check_output(["git", "cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)"], cwd=ROOT, input=obj_input.encode()).decode().splitlines()
    large_history = []; raw_history = []; largest = (0, "")
    for line in stats:
        typ, size, *rest = line.split(" ", 2)
        if typ != "blob": continue
        name = rest[0] if rest else ""
        size = int(size)
        if size > largest[0]: largest = (size, name)
        if size > 5*1024*1024: large_history.append({"path": name, "bytes": size})
        if Path(name).suffix.lower() in (".parquet", ".jsonl", ".npy", ".pkl"): raw_history.append(name)
    out = {"tracked_count": len(files), "current_locations": current, "current_over_5MiB": oversized,
           "current_raw_candidates": raw_candidates, "csv_headers": csv_headers,
           "history_patch_locations": {k: sorted(v) for k,v in history.items()},
           "history_over_5MiB": large_history, "history_raw_candidates": sorted(set(raw_history)),
           "largest_history_blob": {"bytes": largest[0], "path": largest[1]}}
    target = ROOT / "logs/publication_audit.json"
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k not in ("csv_headers", "history_patch_locations")},ensure_ascii=False,indent=2))
    print("Historical patch finding groups:", len(history))
    print("Full location-only results:", target.relative_to(ROOT))

if __name__ == "__main__":
    main()
