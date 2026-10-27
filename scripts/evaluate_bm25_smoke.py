import json
import unicodedata
from collections import Counter
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "evaluation" / "reports" / "bm25_smoke"
KS = (1, 5, 10)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def normalize(text):
    return " ".join(
        unicodedata.normalize("NFC", text).split()
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluate(dataset):
    report_path = REPORTS / f"{dataset}_retrieval.json"
    retrieval = json.loads(report_path.read_text(encoding="utf-8"))

    require(retrieval["dataset"] == dataset, "Wrong dataset report")
    require(retrieval["split"] == "train", "Expected train smoke report")
    require(
        retrieval["config"]["top_k"] >= max(KS),
        "Retrieval report must contain top_k >= 10",
    )

    directory = PROCESSED / dataset
    documents_path = (
        directory / "train" / "documents.jsonl"
        if dataset == "qasper"
        else directory / "documents.jsonl"
    )

    documents = {
        row["evidence_id"]: row
        for row in read_jsonl(documents_path)
    }

    gold = {
        row["query_id"]: row
        for row in read_jsonl(directory / "train" / "gold.jsonl")
    }

    selection = json.loads(
        (ROOT / "configs" / "smoke_queries.json")
        .read_text(encoding="utf-8")
    )
    expected_ids = set(selection[f"{dataset}_query_ids"])
    results = retrieval["results"]
    actual_ids = [row["query_id"] for row in results]

    require(
        len(actual_ids) == len(set(actual_ids)),
        "Duplicate retrieval query IDs",
    )
    require(
        set(actual_ids) == expected_ids,
        "Retrieval queries do not match smoke_queries.json",
    )

    def evidence_key(evidence_id):
        require(
            evidence_id in documents,
            f"Unknown evidence ID: {evidence_id}",
        )
        document = documents[evidence_id]

        if dataset == "qasper":
            return (
                document["paper_id"],
                normalize(document["text"]),
            )

        return evidence_id

    skipped = Counter()
    scored = []

    for result in results:
        query_id = result["query_id"]
        require(query_id in gold, f"Missing gold: {query_id}")

        reference = gold[query_id]
        targets = []

        if dataset == "qasper":
            for annotation in reference["annotations"]:
                answer = annotation["answer"]

                if answer.get("unanswerable"):
                    continue

                if annotation["figure_table_evidence"]:
                    continue

                if annotation["unmatched_text_evidence"]:
                    continue

                ids = annotation["evidence_ids"]

                if ids:
                    targets.append({
                        evidence_key(evidence_id)
                        for evidence_id in ids
                    })

            if not targets:
                skipped["no_eligible_text_annotation"] += 1
                continue
        else:
            ids = {
                evidence_id
                for evidence_set in reference["evidence_sets"]
                for evidence_id in evidence_set["evidence_ids"]
            }

            if not ids:
                skipped["no_annotated_evidence"] += 1
                continue

            targets = [{
                evidence_key(evidence_id)
                for evidence_id in ids
            }]

        hits = result["hits"]

        require(
            [hit["rank"] for hit in hits]
            == list(range(1, len(hits) + 1)),
            f"Invalid hit ranks: {query_id}",
        )
        require(
            len(hits) <= retrieval["config"]["top_k"],
            f"Too many hits: {query_id}",
        )

        hit_ids = [hit["evidence_id"] for hit in hits]
        require(
            len(hit_ids) == len(set(hit_ids)),
            f"Duplicate hit IDs: {query_id}",
        )

        # Validate the report against current processed documents.
        for hit in hits:
            evidence_id = hit["evidence_id"]
            require(
                evidence_id in documents,
                f"Unknown retrieved evidence: {evidence_id}",
            )
            require(
                hit["text"] == documents[evidence_id]["text"],
                f"Stale retrieval text: {evidence_id}",
            )

            if dataset == "qasper":
                require(
                    documents[evidence_id]["paper_id"] == result["scope"],
                    f"Wrong-paper retrieval: {query_id}",
                )

        ranked_keys = [
            evidence_key(hit["evidence_id"])
            for hit in hits
        ]

        metrics = {}

        for k in KS:
            retrieved = set(ranked_keys[:k])

            metrics[f"recall@{k}"] = max(
                len(retrieved & target) / len(target)
                for target in targets
            )

        # Relevant if present in any eligible reference annotation.
        relevant = set().union(*targets)
        reciprocal_rank = 0.0

        for rank, key in enumerate(ranked_keys[:10], start=1):
            if key in relevant:
                reciprocal_rank = 1.0 / rank
                break

        metrics["mrr@10"] = reciprocal_rank

        scored.append({
            "query_id": query_id,
            "query": result["query"],
            "eligible_reference_sets": len(targets),
            **metrics,
        })

    metric_names = [
        "recall@1",
        "recall@5",
        "recall@10",
        "mrr@10",
    ]

    averages = {
        name: (
            sum(row[name] for row in scored) / len(scored)
            if scored else None
        )
        for name in metric_names
    }

    summary = {
        "dataset": dataset,
        "split": "train",
        "purpose": "Internal retrieval smoke evaluation",
        "retrieval_report": report_path.name,
        "selected_queries": len(results),
        "scored_queries": len(scored),
        "skipped_queries": dict(skipped),
        "protocol": {
            "qasper": (
                "Best recall over eligible text-only annotations; "
                "MRR uses their union; normalized duplicate text "
                "within a paper is equivalent."
            ),
            "scifact": (
                "Sentence retrieval against union of annotated evidence; "
                "not complete-rationale or label evaluation."
            ),
            "averaging": "Macro average over scored queries only",
            "duplicate_hits": (
                "Each returned hit consumes one rank slot, "
                "including equivalent text."
            ),
        },
        "metrics": averages,
        "per_query": scored,
    }

    output = REPORTS / f"{dataset}_metrics.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== {dataset.upper()} ===")
    print(f"Selected: {len(results)}")
    print(f"Scored: {len(scored)}")
    print(f"Skipped: {dict(skipped)}")

    for name, value in averages.items():
        display = "N/A" if value is None else f"{value:.4f}"
        print(f"{name}: {display}")

    print(f"Saved: {output}")


def main():
    evaluate("qasper")
    evaluate("scifact")


if __name__ == "__main__":
    main()