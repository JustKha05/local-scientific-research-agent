import argparse
import csv
import io
import json
import random
from pathlib import Path
from scripts.benchmark_common import require


def new_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="") as stream:
        stream.write(text)


def main():
    p = argparse.ArgumentParser(description="Paired BM25/dense table and diagnosis-only review sample")
    p.add_argument("--dataset", required=True, choices=("qasper", "scifact"))
    p.add_argument("--bm25-dir", required=True, type=Path)
    p.add_argument("--dense-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--include-reporting", action="store_true")
    a = p.parse_args()
    require(a.sample_size > 0, "sample-size must be positive")
    suffix = "final" if a.include_reporting else "development"
    summaries = [json.loads((d / f"{a.dataset}_metrics_{suffix}.json").read_text(encoding="utf8"))
                 for d in (a.bm25_dir, a.dense_dir)]
    b, d = summaries
    require(b["retriever"] == "bm25" and d["retriever"] == "dense", "Wrong retriever pair")
    for field in ("dataset", "split", "source_hashes", "gold_sha256", "manifest_sha256"):
        require(b[field] == d[field], f"Comparison {field} mismatch")
    require(set(b["partitions"]) == set(d["partitions"]), "Partition mismatch")
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["partition", "metric", "bm25", "dense", "dense_minus_bm25"])
    for part in b["partitions"]:
        bp, dp = b["partitions"][part], d["partitions"][part]
        for field in ("selected_queries", "groups", "scored_queries", "skipped_count", "skipped_reasons"):
            require(bp[field] == dp[field], f"Denominator mismatch: {part}")
        require(set(bp["metrics"]) == set(dp["metrics"]), "Metric mismatch")
        for name, value in bp["metrics"].items():
            other = dp["metrics"][name]
            writer.writerow([part, name, value, other, other - value])
    new_text(a.output_dir / f"{a.dataset}_comparison_{suffix}.csv", stream.getvalue())
    if a.include_reporting:
        return
    examples = []
    for method, directory in zip(("bm25", "dense"), (a.bm25_dir, a.dense_dir)):
        rows = json.loads((directory / f"{a.dataset}_diagnosis.json").read_text(encoding="utf8"))
        examples.extend({"retriever": method, **r} for r in rows)
    # Round-robin strata prevent a sample consisting entirely of a common status.
    strata = {}
    for row in examples:
        status = row["metrics"]["status"] if row["metrics"] else "skipped"
        strata.setdefault((row["retriever"], status), []).append(row)
    rng = random.Random(42)
    for key in sorted(strata):
        strata[key].sort(key=lambda r: r["query_id"])
        rng.shuffle(strata[key])
    sample = []
    while len(sample) < a.sample_size and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(sample) < a.sample_size:
                sample.append(strata[key].pop())
    lines = ["# Diagnosis review", "", "Seed=42; stratified round-robin by retriever/status. "
             "This oversampled review is not an estimate of population error rates.", "",
             "Categories: lexical mismatch; semantic distractor; truncation; neighboring context; "
             "dispersed evidence; annotation/representation; unsupported modality; uncertain.", ""]
    for row in sample:
        lines.extend([f"## {row['retriever']} / {row['query_id']}", "",
                      "```json", json.dumps(row, ensure_ascii=False, indent=2), "```", "",
                      "- category:", "- observation / supporting evidence:", "- proposed_fix:", ""])
    new_text(a.output_dir / f"{a.dataset}_diagnosis_review.md", "\n".join(lines))


if __name__ == "__main__":
    main()
