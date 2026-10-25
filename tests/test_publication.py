"""Publication metadata cannot authorize training/scoring or replace an existing freeze."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tbrec import freeze as fz

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ["artifacts/freeze.json", "artifacts/kion/freeze.json", "artifacts/x5/freeze.json"]


@pytest.mark.parametrize("name", MANIFESTS)
def test_publication_manifest_cannot_authorize_oot_even_at_matching_head(name, monkeypatch):
    path = ROOT / name
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["publication_only"] is True
    assert manifest["manifest_role"] == "published_equivalent"
    monkeypatch.setattr(fz, "FREEZE", path)
    monkeypatch.setattr(fz, "git", lambda *args: manifest["commit"] if args[0] == "rev-parse" else "")

    def must_not_read_data(*args):
        pytest.fail("Publication manifests must be rejected before experimental files are read")

    monkeypatch.setattr(fz, "sha256", must_not_read_data)
    with pytest.raises(AssertionError, match="publication-only manifest cannot authorize OOT"):
        fz.check(manifest["branches"][0])


@pytest.mark.parametrize("name", MANIFESTS)
def test_freeze_command_never_overwrites_publication_manifest(name, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("publication_freeze_command", ROOT / "scripts/freeze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "freeze.json"
    original = (ROOT / name).read_bytes()
    path.write_bytes(original)
    monkeypatch.setattr(fz, "FREEZE", path)
    monkeypatch.setattr(fz, "git", lambda *args: "")
    with pytest.raises(AssertionError, match="already frozen"):
        module.main()
    assert path.read_bytes() == original


@pytest.mark.parametrize("name", MANIFESTS)
def test_direct_perseus_score_rejects_publication_before_checkpoint_access(name, tmp_path):
    # Execute the actual scoring function in isolation: optional GPU imports and model loading
    # must not be needed to establish that a publication manifest is ineligible.
    parsed = ast.parse((ROOT / "external/perseus_adapter/stage2.py").read_text(encoding="utf-8"))
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "score")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/freeze.json").write_bytes((ROOT / name).read_bytes())
    namespace = {"ROOT": tmp_path, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "stage2.score", "exec"), namespace)
    args = SimpleNamespace(start=294, end=300, kind="r3_oot")
    with pytest.raises(AssertionError, match="publication-only manifest cannot authorize OOT"):
        namespace["score"](args)
