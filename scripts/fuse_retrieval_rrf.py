"""Gold-free RRF over prediction reports; timings cover fusion only."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.benchmark_common import file_hash, index, require, write_new

TIMING_PROTOCOL = "Fusion only (union, RRF scoring, sorting, top-k); excludes source retrieval, report I/O and validation. Not end-to-end pipeline latency."


def validate_sources(bm25, dense, dataset):
    for report, method in ((bm25, "bm25"), (dense, "dense")):
        require(report["schema_version"] == 1, "Unsupported report schema")
        require(report["dataset"] == dataset, "Report dataset mismatch")
        require(report["config"]["retriever"] == method, f"Expected {method} report")
        depth = report["config"]["top_k"]
        require(type(depth) is int and depth > 0, "Invalid source top_k")
        require(report["split"] in ("train", "dev", "test"), "Invalid split")
        require(isinstance(report["source_hashes"], dict) and
                all(report["source_hashes"].get(k) for k in
                    ("queries_sha256", "documents_sha256")), "Missing source hashes")
        require(report["split"] != "dev" or bool(report["manifest_sha256"]),
                "Dev report requires manifest hash")
    for key in ("dataset", "split", "source_hashes", "manifest_sha256"):
        require(bm25[key] == dense[key], f"Report {key} mismatch")
    left, right = (index(r["results"], "query_id") for r in (bm25, dense))
    require(bool(left) and set(left) == set(right), "Query IDs mismatch or empty reports")
    # Compare every evidence occurrence, including across queries. Rank and score
    # are retrieval metadata; all other fields belong to the evidence payload.
    evidence = {}
    for qid in left:
        for key in ("query", "scope", "candidate_count"):
            require(left[qid][key] == right[qid][key], f"Query {qid}: {key} mismatch")
        for report, rows in ((bm25, left), (dense, right)):
            row = rows[qid]
            count = row["candidate_count"]
            require(type(count) is int and count > 0, "Invalid candidate_count")
            require(dataset != "scifact" or row["scope"] == "corpus", "Wrong SciFact scope")
            hits = row["hits"]
            require(len(hits) <= min(count, report["config"]["top_k"]), "Too many source hits")
            require(all(type(h["rank"]) is int for h in hits) and
                    [h["rank"] for h in hits] == list(range(1, len(hits) + 1)), "Invalid source ranks")
            index(hits, "evidence_id")
            for hit in hits:
                eid = hit["evidence_id"]
                require(isinstance(eid, str) and bool(eid), "Invalid evidence_id")
                payload = {k: v for k, v in hit.items() if k not in ("rank", "score")}
                require("text" in payload and "paper_id" in payload, "Missing evidence content")
                require(evidence.setdefault(eid, payload) == payload,
                        f"Conflicting evidence content: {eid}")
                require(dataset != "qasper" or payload["paper_id"] == row["scope"],
                        "Wrong-paper evidence")
    return left, right


def fuse_hits(left, right, rrf_k, top_k):
    scores, evidence = {}, {}
    for hits in (left, right):
        for hit in hits:
            eid = hit["evidence_id"]
            evidence[eid] = {k: v for k, v in hit.items() if k not in ("rank", "score")}
            scores[eid] = scores.get(eid, 0.) + 1 / (rrf_k + hit["rank"])
    order = sorted(scores, key=lambda eid: (-scores[eid], eid))[:top_k]
    return [{**evidence[eid], "rank": rank, "score": scores[eid]}
            for rank, eid in enumerate(order, 1)]


def run(a):
    require(type(a.rrf_k) is int and a.rrf_k >= 0, "rrf-k must be nonnegative")
    require(type(a.top_k) is int and a.top_k > 0, "top-k must be positive")
    output = a.report_dir / f"{a.dataset}_retrieval.json"
    require(not output.exists(), f"Refusing to overwrite {output}")
    reports, sources = [], {}
    for method, directory in (("bm25", a.bm25_dir), ("dense", a.dense_dir)):
        path = directory / f"{a.dataset}_retrieval.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        reports.append(report)
        sources[method] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                           "top_k": report["config"]["top_k"],
                           "returned_depth_by_query": {r["query_id"]: len(r["hits"])
                                                       for r in report["results"]}}
    left, right = validate_sources(*reports, a.dataset)
    results = []
    for qid, row in left.items():
        tick = time.perf_counter()
        hits = fuse_hits(row["hits"], right[qid]["hits"], a.rrf_k, a.top_k)
        seconds = time.perf_counter() - tick
        results.append({k: row[k] for k in ("query_id", "query", "scope", "candidate_count")})
        results[-1].update(hits=hits, search_seconds=seconds)
    report = {k: reports[0][k] for k in
              ("schema_version", "dataset", "split", "source_hashes", "manifest_sha256")}
    report.update(created_at=datetime.now(timezone.utc).isoformat(),
                  config={"retriever": "hybrid_rrf", "rrf_k": a.rrf_k, "top_k": a.top_k},
                  fusion_sources=sources, script_sha256=file_hash(__file__),
                  timing_protocol=TIMING_PROTOCOL,
                  fusion_seconds=sum(r["search_seconds"] for r in results), results=results)
    write_new(output, report)
    print(f"Saved {output}; queries={len(results)}; fusion only, excludes retrieval")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("qasper", "scifact"), required=True)
    for name in ("bm25-dir", "dense-dir", "report-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--top-k", type=int, default=10)
    run(p.parse_args())


if __name__ == "__main__":
    main()
