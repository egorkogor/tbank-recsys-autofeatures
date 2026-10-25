"""Fetch pinned public X5 CSV.gz files and convert them without interpreting values.

Never replaces existing Parquet files. No preparation of labels, training or scoring.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "5"

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(5)
pa.set_io_thread_count(5)
ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "artifacts/x5/source_provenance.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_source(entry, cache_dir):
    """Use a checked cache or atomically download a file from the pinned source."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (entry["name"] + ".csv.gz")
    expected = entry["source_csv_gz_sha256"]
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"{target.name}: cached source SHA256 mismatch; file left unchanged")
        return target
    descriptor, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=cache_dir)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        digest = hashlib.sha256()
        total = 0
        next_notice = 64 * 1024 * 1024
        with urllib.request.urlopen(entry["source_url"], timeout=60) as source, temporary.open("wb") as dest:
            while block := source.read(8 * 1024 * 1024):
                dest.write(block)
                digest.update(block)
                total += len(block)
                if total >= next_notice:
                    print(f"{entry['name']}: downloaded {total // (1024 * 1024)} MiB", flush=True)
                    next_notice += 64 * 1024 * 1024
        if digest.hexdigest() != expected:
            raise ValueError(f"{target.name}: downloaded source SHA256 mismatch")
        temporary.rename(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def convert_csv(source, output, chunk_rows=200_000):
    """Preserve strings, leading zeros, empty fields and row order; bound memory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output.name}: refusing to replace existing Parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".part", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    writer = None
    rows = 0
    try:
        with pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8",
                         chunksize=chunk_rows) as reader:
            for chunk in reader:
                table = pa.Table.from_pandas(
                    chunk, preserve_index=False,
                    schema=pa.schema([pa.field(name, pa.string()) for name in chunk.columns]),
                )
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", write_statistics=False)
                writer.write_table(table)
                rows += table.num_rows
                if rows and rows % 5_000_000 == 0:
                    print(f"{output.stem}: converted {rows} rows", flush=True)
        if writer is None:
            raise ValueError("CSV did not produce a table")
        writer.close()
        writer = None
        temporary.rename(output)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return rows


def same_values(left, right, batch_rows=200_000):
    """Compare every string, null and row position; ignore Parquet encoding metadata."""
    left, right = pq.ParquetFile(left), pq.ParquetFile(right)
    if left.metadata.num_rows != right.metadata.num_rows or not left.schema_arrow.equals(right.schema_arrow):
        return False
    sentinel = object()
    a = iter(left.iter_batches(batch_size=batch_rows, use_threads=False))
    b = iter(right.iter_batches(batch_size=batch_rows, use_threads=False))
    while True:
        first, second = next(a, sentinel), next(b, sentinel)
        if first is sentinel or second is sentinel:
            return first is second
        if not first.equals(second):
            return False


def prepare_file(entry, output_dir, cache_dir, reference_dir=None):
    started = time.perf_counter()
    output = Path(output_dir) / (entry["name"] + ".parquet")
    expected = entry["local_parquet_sha256"]
    if output.exists():
        if sha256(output) != expected:
            raise FileExistsError(f"{output.name}: existing Parquet differs from historical SHA256; use a new output directory")
        status = "existing_historical_file_verified"
    else:
        source = verified_source(entry, cache_dir)
        convert_csv(source, output)
        status = "converted_from_verified_public_csv"
    parquet = pq.ParquetFile(output)
    result = {"name": entry["name"], "status": status,
              "source_csv_gz_sha256": entry["source_csv_gz_sha256"],
              "rows": parquet.metadata.num_rows, "columns": parquet.schema_arrow.names,
              "parquet_sha256": sha256(output), "historical_parquet_sha256": expected}
    result["byte_equal_to_historical"] = result["parquet_sha256"] == expected
    if reference_dir is not None:
        reference = Path(reference_dir) / output.name
        if reference.resolve() == output.resolve():
            raise ValueError("Reference and output must be different files")
        if sha256(reference) != expected:
            raise ValueError(f"{output.name}: reference does not match historical SHA256")
        result["all_values_equal_to_historical"] = same_values(output, reference)
        if not result["all_values_equal_to_historical"]:
            raise ValueError(f"{output.name}: values differ from historical reference")
    result["seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/x5")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/raw/x5/csv")
    parser.add_argument("--files", nargs="+", choices=("clients", "products", "purchases"),
                        default=["clients", "products", "purchases"])
    parser.add_argument("--reference-dir", type=Path, help="Optionally compare every value against historical Parquet")
    args = parser.parse_args()
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in provenance["files"]}
    if args.reference_dir is not None and args.output_dir.resolve() == args.reference_dir.resolve():
        parser.error("--reference-dir must differ from --output-dir")
    results = [prepare_file(entries[name], args.output_dir, args.cache_dir, args.reference_dir)
               for name in dict.fromkeys(args.files)]
    report = {"dataset": provenance["dataset"], "source_commit": provenance["source_commit"],
              "scope": "raw data conversion only; no labels, training or scoring",
              "conversion": {"all_columns": "string", "keep_default_na": False, "chunk_rows": 200_000,
                             "compression": "zstd", "write_statistics": False},
              "versions": {"pandas": pd.__version__, "pyarrow": pa.__version__}, "files": results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
