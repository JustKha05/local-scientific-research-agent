"""Strict retrieval evaluation; reporting is held out unless explicitly unlocked."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts.benchmark_common import (ROOT, KS, PARTS, file_hash, index, load_inputs,
                                      read_jsonl, require, validate_manifest, write_new)
from scripts.evaluate_bm25_smoke import normalize


def score_query(dataset, reference, ranked_ids, documents):
    def key(eid):
        require(eid in documents, f"Unknown gold evidence: {eid}")
        d = documents[eid]
        return (d["paper_id"], normalize(d["text"])) if dataset == "qasper" else eid

    if dataset == "qasper":
        targets = [{key(e) for e in a["evidence_ids"]} for a in reference["annotations"]
                   if not a["answer"].get("unanswerable") and not a["figure_table_evidence"]
                   and not a["unmatched_text_evidence"] and a["evidence_ids"]]
        reason = "no_eligible_text_annotation"
    else:
        targets = []
        for evidence_set in reference["evidence_sets"]:
            require(bool(evidence_set["evidence_ids"]), "Empty rationale")
            target = {key(e) for e in evidence_set["evidence_ids"]}
            require(all(documents[e]["paper_id"] == evidence_set["paper_id"] for e in target),
                    "Wrong gold paper")
            targets.append(target)
        reason = "no_annotated_evidence"
    if not targets:
        return None, reason
    union = set().union(*targets)
    ranked = [key(e) for e in ranked_ids]
    recall_targets = targets if dataset == "qasper" else [union]
    metrics = {}
    complete_name = "complete_evidence_hit" if dataset == "qasper" else "complete_rationale_hit"
    for k in KS:
        retrieved = set(ranked[:k])
        metrics[f"recall@{k}"] = max(len(retrieved & t) / len(t) for t in recall_targets)
        metrics[f"{complete_name}@{k}"] = int(any(t <= retrieved for t in targets))
        if dataset == "scifact":
            metrics[f"union_sentence_recall@{k}"] = metrics[f"recall@{k}"]
            papers = {documents[e]["paper_id"] for e in union}
            hit_papers = {documents[e]["paper_id"] for e in ranked_ids[:k]}
            metrics[f"gold_document_recall_in_top_{k}_sentences"] = len(papers & hit_papers) / len(papers)
    metrics["mrr@10"] = next((1 / rank for rank, e in enumerate(ranked[:10], 1) if e in union), 0.)
    complete = metrics[f"{complete_name}@10"]
    status = ("complete_alternative" if complete and not union <= set(ranked[:10]) else
              "complete" if complete else "partial" if metrics["recall@10"] else "miss")
    return {**metrics, "status": status}, None


def validate_report(report, dataset, split, queries, documents, hashes):
    require(report["dataset"] == dataset and report["split"] == split, "Report dataset/split mismatch")
    require(report["source_hashes"] == hashes, "Report data hashes mismatch")
    top_k = report["config"]["top_k"]
    require(type(top_k) is int and top_k >= 10, "Evaluation requires top-k >= 10")
    rows = index(report["results"], "query_id")
    require(set(rows) == set(queries), "Retrieval query coverage mismatch")
    for qid, row in rows.items():
        q = queries[qid]
        scope = q["paper_id"] if dataset == "qasper" else "corpus"
        require(row["query"] == q["text"] and row["scope"] == scope, "Stale query/wrong scope")
        candidates = sum(dataset != "qasper" or d["paper_id"] == scope for d in documents.values())
        require(row["candidate_count"] == candidates, "Candidate count mismatch")
        require(math.isfinite(row["search_seconds"]) and row["search_seconds"] >= 0, "Invalid latency")
        hits = row["hits"]
        require(len(hits) <= min(top_k, candidates), "Too many hits")
        require(all(type(h["rank"]) is int for h in hits) and
                [h["rank"] for h in hits] == list(range(1, len(hits) + 1)), "Invalid ranks")
        index(hits, "evidence_id")
        for hit in hits:
            eid = hit["evidence_id"]
            require(eid in documents, f"Unknown retrieved evidence {eid}")
            require(all(hit.get(k) == v for k, v in documents[eid].items()), "Stale evidence content")
            require(dataset != "qasper" or documents[eid]["paper_id"] == scope, "Wrong-paper retrieval")
            require(math.isfinite(hit["score"]), "Non-finite score")
        require(all(a["score"] >= b["score"] for a, b in zip(hits, hits[1:])), "Scores out of order")
    return rows


def evaluate(a):
    queries, docs, hashes = load_inputs(a.processed_dir, a.dataset, a.split)
    queries, documents = index(queries, "query_id"), index(docs, "evidence_id")
    report_path = a.report_dir / f"{a.dataset}_retrieval.json"
    report = json.loads(report_path.read_text(encoding="utf8"))
    rows = validate_report(report, a.dataset, a.split, queries, documents, hashes)
    if a.split == "dev":
        require(a.query_manifest is not None, "Dev evaluation requires --query-manifest")
        manifest = json.loads(a.query_manifest.read_text(encoding="utf8"))
        assignments = validate_manifest(manifest, list(queries.values()), a.dataset, hashes["queries_sha256"])
        require(report["manifest_sha256"] == file_hash(a.query_manifest), "Report manifest mismatch")
        parts = PARTS if a.include_reporting else PARTS[:2]
    else:
        assignments = {qid: {"partition": a.split} for qid in queries}
        parts = (a.split,)
    gold_path = a.processed_dir / a.dataset / a.split / "gold.jsonl"
    gold = index(read_jsonl(gold_path), "query_id")
    require(set(gold) == set(queries), "Gold query coverage mismatch")
    summaries, diagnosis = {}, []
    for part in parts:
        scored, skipped = [], Counter()
        for qid, row in rows.items():
            if assignments[qid]["partition"] != part:
                continue
            metric, reason = score_query(a.dataset, gold[qid], [h["evidence_id"] for h in row["hits"]], documents)
            if reason:
                skipped[reason] += 1
            else:
                scored.append(metric)
            if part == "diagnosis":
                reference_ids = {eid for evidence_set in gold[qid].get("evidence_sets", [])
                                 for eid in evidence_set["evidence_ids"]}
                reference_ids.update(eid for annotation in gold[qid].get("annotations", [])
                                     for eid in annotation["evidence_ids"])
                require(reference_ids <= set(documents), "Unknown diagnosis gold evidence")
                diagnosis.append({"query_id": qid, "query": queries[qid]["text"],
                                  "paper_id": queries[qid].get("paper_id"),
                                  "metrics": metric, "skip_reason": reason,
                                  "hits": row["hits"], "gold": gold[qid],
                                  "gold_documents": [documents[eid] for eid in sorted(reference_ids)]})
        names = [k for k in scored[0] if k != "status"] if scored else []
        summaries[part] = {"selected_queries": len(scored) + sum(skipped.values()),
                           "groups": len({assignments[qid].get("group_id", qid) for qid in queries
                                          if assignments[qid]["partition"] == part}),
                           "scored_queries": len(scored), "skipped_count": sum(skipped.values()),
                           "skipped_reasons": dict(skipped),
                           "statuses": dict(Counter(r["status"] for r in scored)),
                           "metrics": {n: sum(r[n] for r in scored) / len(scored) for n in names}}
    output = {"dataset": a.dataset, "split": a.split, "retriever": report["config"]["retriever"],
              "reporting_unlocked": a.include_reporting, "partitions": summaries,
              "manifest_sha256": report.get("manifest_sha256"), "source_hashes": hashes,
              "partition_counts": manifest["counts"] if a.split == "dev" else None,
              "retrieval_sha256": file_hash(report_path), "gold_sha256": file_hash(gold_path),
              "protocol": "Macro over scored queries; smoke recall/MRR semantics; duplicate text consumes rank. "
                          "Complete hit means any one eligible annotation/rationale. Internal, not official scoring."}
    suffix = "final" if a.include_reporting else "development"
    write_new(a.report_dir / f"{a.dataset}_metrics_{suffix}.json", output)
    if diagnosis and not a.include_reporting:
        write_new(a.report_dir / f"{a.dataset}_diagnosis.json", diagnosis)
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("qasper", "scifact"), required=True)
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--query-manifest", type=Path)
    p.add_argument("--include-reporting", action="store_true", help="Only after freezing the method")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
