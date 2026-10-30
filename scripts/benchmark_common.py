"""Shared, gold-free input and immutable artifact helpers for retrieval benchmarks."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("diagnosis", "calibration", "reporting")
KS = (1, 5, 10)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf8").splitlines() if s.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def index(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {key}")
    return result


def load_inputs(processed, dataset, split):
    base = Path(processed) / dataset
    qp = base / split / "queries.jsonl"
    dp = base / split / "documents.jsonl" if dataset == "qasper" else base / "documents.jsonl"
    queries, documents = read_jsonl(qp), read_jsonl(dp)
    index(queries, "query_id")
    index(documents, "evidence_id")
    require(bool(queries) and bool(documents), "Empty benchmark input")
    return queries, sorted(documents, key=lambda d: d["evidence_id"]), {
        "queries_sha256": file_hash(qp), "documents_sha256": file_hash(dp)}


def group_id(query, dataset):
    return query["paper_id"] if dataset == "qasper" else query["query_id"]


def manifest_counts(assignments):
    return {part: {"queries": sum(a["partition"] == part for a in assignments),
                   "groups": len({a["group_id"] for a in assignments if a["partition"] == part})}
            for part in PARTS}


def make_manifest(queries, dataset, query_hash):
    groups = sorted({group_id(q, dataset) for q in queries})
    random.Random(42).shuffle(groups)
    # Largest-remainder allocation; ratios apply to groups, not query counts.
    raw = [len(groups) * r for r in (.4, .2, .4)]
    sizes = [int(n) for n in raw]
    for i in sorted(range(3), key=lambda i: (-(raw[i] - sizes[i]), i))[:len(groups) - sum(sizes)]:
        sizes[i] += 1
    mapping, start = {}, 0
    for part, size in zip(PARTS, sizes):
        mapping.update({g: part for g in groups[start:start + size]})
        start += size
    assignments = [{"query_id": q["query_id"], "group_id": group_id(q, dataset),
                    "partition": mapping[group_id(q, dataset)]}
                   for q in sorted(queries, key=lambda q: q["query_id"])]
    return {"schema_version": 1, "dataset": dataset, "split": "dev", "seed": 42,
            "ratios": [.4, .2, .4], "queries_sha256": query_hash,
            "assignments": assignments, "counts": manifest_counts(assignments)}


def validate_manifest(manifest, queries, dataset, query_hash):
    require(manifest["schema_version"] == 1 and manifest["dataset"] == dataset
            and manifest["split"] == "dev" and manifest["seed"] == 42
            and manifest["ratios"] == [.4, .2, .4], "Manifest protocol mismatch")
    require(manifest["queries_sha256"] == query_hash, "Manifest data hash mismatch")
    rows = index(manifest["assignments"], "query_id")
    require(set(rows) == {q["query_id"] for q in queries}, "Manifest query coverage mismatch")
    groups = {}
    for q in queries:
        row = rows[q["query_id"]]
        g, part = group_id(q, dataset), row["partition"]
        require(row["group_id"] == g and part in PARTS, "Invalid manifest assignment")
        require(groups.setdefault(g, part) == part, "Group leakage between partitions")
    require(manifest["counts"] == manifest_counts(list(rows.values())), "Manifest counts mismatch")
    return rows
