import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "evaluation" / "reports" / "bm25_smoke"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def inspect(dataset):
    metrics = json.loads(
        (REPORTS / f"{dataset}_metrics.json")
        .read_text(encoding="utf-8")
    )
    retrieval = json.loads(
        (REPORTS / f"{dataset}_retrieval.json")
        .read_text(encoding="utf-8")
    )

    results = {
        row["query_id"]: row
        for row in retrieval["results"]
    }

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

    # Inspect only scored queries. Skipped queries are not retrieval failures.
    failures = sorted(
        [
            row
            for row in metrics["per_query"]
            if row["recall@10"] < 1.0
        ],
        key=lambda row: (row["recall@10"], row["query_id"]),
    )

    lines = [
        f"# {dataset.upper()} BM25 error analysis",
        "",
        "Source: train smoke queries.",
        "Cases are ordered by Recall@10, lowest first.",
        "Gold evidence is shown for diagnosis only.",
        "",
        f"- Scored queries: {metrics['scored_queries']}",
        f"- Queries with Recall@10 below 1: {len(failures)}",
        f"- Queries with Recall@10 equal to 0: "
        f"{sum(row['recall@10'] == 0 for row in failures)}",
        "",
    ]

    for number, row in enumerate(failures, start=1):
        query_id = row["query_id"]
        result = results[query_id]
        reference = gold[query_id]

        lines.extend([
            f"## Case {number}",
            "",
            f"Query ID: `{query_id}`",
            "",
            f"Question/claim: {result['query']}",
            "",
            f"Recall@10: {row['recall@10']:.4f}",
            "",
            "### Eligible gold evidence",
            "",
        ])

        if dataset == "qasper":
            groups = []

            for annotation in reference["annotations"]:
                if annotation["answer"].get("unanswerable"):
                    continue
                if annotation["figure_table_evidence"]:
                    continue
                if annotation["unmatched_text_evidence"]:
                    continue
                if not annotation["evidence_ids"]:
                    continue

                groups.append({
                    "label": "QASPER annotation",
                    "evidence_ids": annotation["evidence_ids"],
                })
        else:
            groups = reference["evidence_sets"]

        for group_index, group in enumerate(groups, start=1):
            lines.append(
                f"Reference set {group_index}: {group['label']}"
            )
            lines.append("")

            for evidence_id in group["evidence_ids"]:
                document = documents[evidence_id]

                lines.extend([
                    f"- ID: `{evidence_id}`",
                    f"  - Unit: {document.get('unit_type', 'sentence')}",
                    f"  - Text: {document['text']}",
                    "",
                ])

        lines.extend(["### Top 5 retrieved results", ""])

        for hit in result["hits"][:5]:
            lines.extend([
                f"Rank {hit['rank']} — score {hit['score']:.4f}",
                "",
                f"ID: `{hit['evidence_id']}`",
                "",
                hit["text"],
                "",
            ])

        lines.extend([
            "### Manual diagnosis",
            "",
            "- Category: TODO",
            "- Observation: TODO",
            "- Possible improvement: TODO",
            "",
        ])

    output = REPORTS / f"{dataset}_errors.md"
    output.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"{dataset.upper()}: "
        f"{len(failures)} incomplete-retrieval cases"
    )
    print(f"Saved: {output}")


def main():
    inspect("qasper")
    inspect("scifact")


if __name__ == "__main__":
    main()