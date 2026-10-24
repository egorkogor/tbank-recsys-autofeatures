"""Score Perseus R3 on the OOT days once, after the freeze, through the same gate as every other branch.

The scorer itself is the Perseus adapter (external/perseus_adapter/stage2.py, prepared with LLM assistance);
it writes artifacts/runs/r3_perseus/pred_oot.parquet.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tbrec.freeze import check  # noqa: E402

if __name__ == "__main__":
    check("r3_perseus")
    cmd = (". external/probe_env.ps1; & $ProbePython external/perseus_adapter/stage2.py "
           "score --model r3_oot --start 294 --end 300 --kind r3_oot")
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd], cwd=ROOT)
    assert (ROOT / "artifacts" / "runs" / "r3_perseus" / "pred_oot.parquet").exists(), "Perseus OOT scoring failed"
    sys.exit(r.returncode)
