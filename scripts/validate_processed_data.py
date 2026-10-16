import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SEED = 42
SAMPLE_SIZE = 30


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


def unique_index(rows, key, description):
    index = {row[key]: row for row in rows}
    require(
        len(index) == len(rows),
        f"Duplicate {key}: {description}",
    )
    return index


def validate_qasper(split):
    directory = PROCESSED / "qasper" / split

    documents = read_jsonl(directory / "documents.jsonl")
    queries = read_jsonl(directory / "queries.jsonl")
    gold = read_jsonl(directory / "gold.jsonl")

    document_index = unique_index(
        documents, "evidence_id", f"QASPER {split} documents"
    )
    query_index = unique_index(
        queries, "query_id", f"QASPER {split} queries"
    )
    gold_index = unique_index(
        gold, "query_id", f"QASPER {split} gold"
    )

    require(
        set(query_index) == set(gold_index),
        f"QASPER {split}: queries/gold mismatch",
    )

    paper_ids = {doc["paper_id"] for doc in documents}

    for query in queries:
        require(
            query["paper_id"] in paper_ids,
            f"Missing paper for {query['query_id']}",
        )

    for row in gold:
        paper_id = query_index[row["query_id"]]["paper_id"]

        for annotation in row["annotations"]:
            require(
                not annotation["unmatched_text_evidence"],
                f"Unmatched evidence: {row['query_id']}",
            )

            for evidence_id in annotation["evidence_ids"]:
                require(
                    evidence_id in document_index,
                    f"Missing evidence: {evidence_id}",
                )
                require(
                    document_index[evidence_id]["paper_id"] == paper_id,
                    f"Evidence from wrong paper: {evidence_id}",
                )

    print(f"QASPER {split}: PASSED")
    return queries


def validate_scifact():
    directory = PROCESSED / "scifact"

    documents = read_jsonl(directory / "documents.jsonl")
    document_index = unique_index(
        documents, "evidence_id", "SciFact documents"
    )

    train_queries = None

    for split in ("train", "dev", "test"):
        queries = read_jsonl(directory / split / "queries.jsonl")
        query_index = unique_index(
            queries, "query_id", f"SciFact {split} queries"
        )

        if split == "train":
            train_queries = queries

        if split == "test":
            require(
                not (directory / split / "gold.jsonl").exists(),
                "SciFact test must not have a generated gold.jsonl",
            )
            print("SciFact test: PASSED (queries only)")
            continue

        gold = read_jsonl(directory / split / "gold.jsonl")
        gold_index = unique_index(
            gold, "query_id", f"SciFact {split} gold"
        )

        require(
            set(query_index) == set(gold_index),
            f"SciFact {split}: queries/gold mismatch",
        )

        for row in gold:
            for evidence_set in row["evidence_sets"]:
                require(
                    evidence_set["label"] in {"SUPPORT", "CONTRADICT"},
                    f"Invalid label: {row['query_id']}",
                )

                for evidence_id in evidence_set["evidence_ids"]:
                    require(
                        evidence_id in document_index,
                        f"Missing evidence: {evidence_id}",
                    )
                    require(
                        document_index[evidence_id]["paper_id"]
                        == evidence_set["paper_id"],
                        f"Evidence from wrong paper: {evidence_id}",
                    )

        print(f"SciFact {split}: PASSED")

    return train_queries


def make_sample(queries):
    # Sort before sampling so selection is independent of file order.
    candidates = sorted(queries, key=lambda row: row["query_id"])
    rng = random.Random(SEED)
    selected = rng.sample(
        candidates,
        min(SAMPLE_SIZE, len(candidates)),
    )
    return sorted(row["query_id"] for row in selected)


def main():
    qasper_train = None

    for split in ("train", "dev", "test"):
        queries = validate_qasper(split)

        if split == "train":
            qasper_train = queries

    scifact_train = validate_scifact()

    smoke_config = {
        "purpose": "Pipeline smoke tests, not benchmark reporting",
        "seed": SEED,
        "source_split": "train",
        "qasper_query_ids": make_sample(qasper_train),
        "scifact_query_ids": make_sample(scifact_train),
    }

    output = ROOT / "configs" / "smoke_queries.json"
    output.write_text(
        json.dumps(smoke_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nSmoke queries: "
        f"QASPER={len(smoke_config['qasper_query_ids'])}, "
        f"SciFact={len(smoke_config['scifact_query_ids'])}"
    )
    print(f"Saved: {output}")
    print("Processed data validation: PASSED")


if __name__ == "__main__":
    main()