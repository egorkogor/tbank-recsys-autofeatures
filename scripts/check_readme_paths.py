"""Check Markdown and code paths against tracked files; classify local prerequisites."""
import fnmatch
import json
import re
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\x00")[:-1]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    refs = set(re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme))
    # Inline code; fenced commands are also searched for concrete path tokens.
    for code in re.findall(r"(?<!`)`([^`\n]+)`(?!`)", readme):
        refs.update(re.findall(r"[.\w*{}/<>-]+(?:/[.\w*{}/<>-]+)+|[\w*.-]+\.(?:py|json|md|txt|sh|ps1|toml|lock|parquet)", code))
    refs.update(re.findall(r"(?:scripts|src|artifacts|external|configs|docs|logs|data)/[\w*/.-]+", readme))
    results = []
    generated = ("data/", ".venv/", "external/perseus/", "external/perseus/.venv", "logs/queue_")
    for ref in sorted(refs):
        if ref.startswith(("http:", "https:", "#")): continue
        ref = re.sub(r":\d+$", "", ref.split("#")[0]).rstrip("/")
        if not ref: continue
        hits = [f for f in files if fnmatch.fnmatchcase(f, ref) or fnmatch.fnmatchcase(f, ref + "/*")]
        status = "tracked" if hits else "missing"
        if not hits and "/" not in ref:
            hits = [f for f in files if fnmatch.fnmatchcase(Path(f).name, ref)]
            if hits: status = "tracked shorthand"
        if not hits and (ref.startswith(generated) or ref == "external/perseus" or ref == "external" or ref.startswith("logs/<")):
            status = "local prerequisite or generated output"
        results.append({"reference": ref, "status": status, "tracked_matches": hits})
    out = {"references": results, "missing": [r["reference"] for r in results if r["status"] == "missing"]}
    (ROOT / "logs/readme_paths.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"references_checked": len(results), "missing": out["missing"], "local": [r["reference"] for r in results if r["status"].startswith("local")]},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
