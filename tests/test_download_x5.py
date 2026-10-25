"""Conversion checks: preserve raw strings and reject unverified or existing inputs."""
import gzip
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("download_x5", Path(__file__).parents[1] / "scripts/download_x5.py")
x5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x5)
pa, pq = x5.pa, x5.pq


def source_file(tmp_path):
    path = tmp_path / "source.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        stream.write('client_id,value,note\n0007,NA,"a,b"\n0008,,"две строки\nв поле"\n0009,1.00,null\n')
    return path


def test_conversion_preserves_strings_empty_fields_order_and_chunks(tmp_path):
    source = source_file(tmp_path)
    out = tmp_path / "result.parquet"
    assert x5.convert_csv(source, out, chunk_rows=2) == 3
    table = pq.read_table(out)
    assert all(pa.types.is_string(field.type) for field in table.schema)
    assert table.to_pydict() == {"client_id": ["0007", "0008", "0009"], "value": ["NA", "", "1.00"],
                               "note": ["a,b", "две строки\nв поле", "null"]}
    meta = pq.ParquetFile(out).metadata
    assert meta.num_row_groups == 2
    assert all(meta.row_group(0).column(i).statistics is None for i in range(meta.num_columns))
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        x5.convert_csv(source, out)
    assert out.read_bytes() == before


def test_cached_source_must_match_pinned_hash(tmp_path):
    source = source_file(tmp_path)
    cached = tmp_path / "clients.csv.gz"
    cached.write_bytes(source.read_bytes())
    entry = {"name": "clients", "source_csv_gz_sha256": x5.sha256(source)}
    assert x5.verified_source(entry, tmp_path) == cached
    cached.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        x5.verified_source(entry, tmp_path)
    assert cached.read_bytes() == b"corrupt"


def test_download_hash_failure_leaves_no_source_or_partial_file(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    monkeypatch.setattr(x5.urllib.request, "urlopen", lambda *args, **kwargs: source.open("rb"))
    cache = tmp_path / "cache"
    entry = {"name": "clients", "source_url": "https://example.invalid/clients.csv.gz",
             "source_csv_gz_sha256": "0" * 64}
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        x5.verified_source(entry, cache)
    assert list(cache.iterdir()) == []


def test_full_comparison_detects_changed_value_and_row_order(tmp_path):
    source = source_file(tmp_path)
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    x5.convert_csv(source, left, chunk_rows=2)
    x5.convert_csv(source, right, chunk_rows=1)
    assert x5.same_values(left, right, batch_rows=2)
    table = pq.read_table(right)
    pq.write_table(table.take(pa.array([1, 0, 2])), right)
    assert not x5.same_values(left, right, batch_rows=2)
    changed = table.set_column(1, "value", pa.array(["NA", "", "2.00"]))
    pq.write_table(changed, right)
    assert not x5.same_values(left, right, batch_rows=2)
