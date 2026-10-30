import argparse
from pathlib import Path
from scripts.benchmark_common import (ROOT, file_hash, index, make_manifest, read_jsonl,
                                      require, validate_manifest, write_new)


def main():
    p = argparse.ArgumentParser(description="Explicitly create an immutable dev manifest (seed 42).")
    p.add_argument("--dataset", required=True, choices=("qasper", "scifact"))
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    query_path = a.processed_dir / a.dataset / "dev/queries.jsonl"
    queries = read_jsonl(query_path)
    index(queries, "query_id")
    require(bool(queries), "Cannot partition empty queries")
    query_hash = file_hash(query_path)
    manifest = make_manifest(queries, a.dataset, query_hash)
    validate_manifest(manifest, queries, a.dataset, query_hash)
    write_new(a.output, manifest)
    print(manifest["counts"])


if __name__ == "__main__":
    main()
