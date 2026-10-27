import hashlib
import json
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evaluation" / "reports" / "bm25_smoke"
DATA = ROOT / "data" / "processed" / "scifact"
KS = (1, 5, 10)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_index(rows, key):
    index = {row[key]: row for row in rows}
    require(len(index) == len(rows), f"Duplicate {key}")
    return index


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global REPORTS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir",
        default="evaluation/reports/bm25_smoke",
        help="Directory containing retrieval reports.",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)

    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir

    REPORTS = report_dir.resolve()

    if not REPORTS.is_dir():
        raise FileNotFoundError(
            f"Report directory does not exist: {REPORTS}"
        )
    retrieval_path = REPORTS / "scifact_retrieval.json"
    gold_path = DATA / "train" / "gold.jsonl"

    retrieval = json.loads(
        retrieval_path.read_text(encoding="utf-8")
    )
    gold = unique_index(read_jsonl(gold_path), "query_id")
    documents = unique_index(
        read_jsonl(DATA / "documents.jsonl"),
        "evidence_id",
    )

    require(retrieval["dataset"] == "scifact", "Wrong dataset")
    require(retrieval["split"] == "train", "Expected train split")
    require(
        retrieval["config"]["top_k"] >= max(KS),
        "Need at least top_k=10 retrieval",
    )

    selection = json.loads(
        (ROOT / "configs" / "smoke_queries.json")
        .read_text(encoding="utf-8")
    )
    results = retrieval["results"]
    result_index = unique_index(results, "query_id")

    require(
        set(result_index) == set(selection["scifact_query_ids"]),
        "Retrieval does not match smoke query selection",
    )

    scored = []
    skipped = []

    for result in results:
        query_id = result["query_id"]
        require(query_id in gold, f"Missing gold: {query_id}")

        evidence_sets = gold[query_id]["evidence_sets"]
        rationale_sets = []
        gold_papers = set()

        for evidence_set in evidence_sets:
            ids = set(evidence_set["evidence_ids"])
            require(bool(ids), f"Empty rationale: {query_id}")

            for evidence_id in ids:
                require(
                    evidence_id in documents,
                    f"Unknown gold evidence: {evidence_id}",
                )
                require(
                    documents[evidence_id]["paper_id"]
                    == evidence_set["paper_id"],
                    f"Wrong gold paper: {evidence_id}",
                )

            rationale_sets.append(ids)
            gold_papers.add(evidence_set["paper_id"])

        if not rationale_sets:
            skipped.append(query_id)
            continue

        hits = result["hits"]
        require(
            [hit["rank"] for hit in hits]
            == list(range(1, len(hits) + 1)),
            f"Invalid ranking: {query_id}",
        )
        require(
            len(hits) <= retrieval["config"]["top_k"],
            f"Too many hits: {query_id}",
        )

        hit_ids = [hit["evidence_id"] for hit in hits]
        require(
            len(hit_ids) == len(set(hit_ids)),
            f"Duplicate retrieved IDs: {query_id}",
        )

        for hit in hits:
            evidence_id = hit["evidence_id"]
            require(
                evidence_id in documents,
                f"Unknown retrieved ID: {evidence_id}",
            )
            require(
                hit["text"] == documents[evidence_id]["text"],
                f"Stale retrieval text: {evidence_id}",
            )

        union_gold = set().union(*rationale_sets)
        row = {"query_id": query_id}

        for k in KS:
            selected_hits = hits[:k]
            retrieved_ids = {
                hit["evidence_id"] for hit in selected_hits
            }
            retrieved_papers = {
                documents[hit["evidence_id"]]["paper_id"]
                for hit in selected_hits
            }

            row[f"union_sentence_recall@{k}"] = (
                len(retrieved_ids & union_gold) / len(union_gold)
            )

            row[f"complete_rationale_hit@{k}"] = int(
                any(
                    rationale <= retrieved_ids
                    for rationale in rationale_sets
                )
            )

            row[f"gold_document_recall_in_top_{k}_sentences"] = (
                len(retrieved_papers & gold_papers)
                / len(gold_papers)
            )

        scored.append(row)

    metric_names = [
        name for name in scored[0] if name != "query_id"
    ] if scored else []

    averages = {
        name: sum(row[name] for row in scored) / len(scored)
        for name in metric_names
    }

    output = {
        "dataset": "scifact",
        "split": "train",
        "purpose": "Internal retrieval diagnostics, not official scoring",
        "selected_queries": len(results),
        "scored_queries": len(scored),
        "skipped_no_annotated_evidence": len(skipped),
        "source_hashes": {
            "retrieval_sha256": file_hash(retrieval_path),
            "gold_sha256": file_hash(gold_path),
        },
        "protocol": {
            "union_sentence_recall": (
                "Recall against union of all annotated rationale sentences"
            ),
            "complete_rationale_hit": (
                "At least one complete annotated rationale is retrieved"
            ),
            "gold_document_recall": (
                "Gold documents represented in top-k sentence hits; "
                "not top-k document retrieval"
            ),
            "averaging": "Macro over queries with annotated evidence",
        },
        "metrics": averages,
        "per_query": scored,
        "skipped_query_ids": skipped,
    }

    output_path = REPORTS / "scifact_rationale_metrics.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Selected: {len(results)}")
    print(f"Scored: {len(scored)}")
    print(f"Skipped: {len(skipped)}")

    for k in KS:
        print(f"\nTop {k} sentences:")
        for name in (
            f"union_sentence_recall@{k}",
            f"complete_rationale_hit@{k}",
            f"gold_document_recall_in_top_{k}_sentences",
        ):
            value = averages.get(name)
            print(f"  {name}: {value:.4f}" if value is not None
                  else f"  {name}: N/A")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()