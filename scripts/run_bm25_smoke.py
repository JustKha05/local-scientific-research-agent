import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from app.retrieval import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TOP_K = 10


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def run_dataset(dataset, selected_ids):
    directory = PROCESSED / dataset

    if dataset == "qasper":
        documents_path = directory / "train" / "documents.jsonl"
    else:
        documents_path = directory / "documents.jsonl"

    documents = read_jsonl(documents_path)
    all_queries = read_jsonl(directory / "train" / "queries.jsonl")
    query_index = {row["query_id"]: row for row in all_queries}

    missing = set(selected_ids) - set(query_index)

    if missing:
        raise ValueError(f"Unknown query IDs: {sorted(missing)}")

    queries = [query_index[query_id] for query_id in selected_ids]

    build_started = time.perf_counter()
    paper_documents = defaultdict(list)
    retrievers = {}

    if dataset == "qasper":
        for document in documents:
            paper_documents[document["paper_id"]].append(document)

        # Build only the papers needed by the smoke queries.
        for paper_id in sorted({q["paper_id"] for q in queries}):
            retrievers[paper_id] = BM25Retriever(
                paper_documents[paper_id]
            )
    else:
        # SciFact: index all corpus sentences, not only gold documents.
        retrievers["corpus"] = BM25Retriever(documents)

    build_seconds = time.perf_counter() - build_started
    results = []

    for query in queries:
        key = query["paper_id"] if dataset == "qasper" else "corpus"
        retriever = retrievers[key]

        started = time.perf_counter()
        hits = retriever.search(query["text"], top_k=TOP_K)
        search_seconds = time.perf_counter() - started

        results.append({
            "query_id": query["query_id"],
            "query": query["text"],
            "scope": key,
            "candidate_count": len(retriever.documents),
            "search_seconds": search_seconds,
            "hits": hits,
        })

    output_dir = ROOT / "evaluation" / "reports" / "bm25_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_retrieval.json"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": "train",
        "purpose": "Smoke test; not final benchmark",
        "python_version": platform.python_version(),
        "rank_bm25_version": version("rank-bm25"),
        "numpy_version": version("numpy"),
        "config": {
            "top_k": TOP_K,
            "k1": 1.5,
            "b": 0.75,
            "epsilon": 0.25,
            "indexed_field": "text",
            "tokenization": "NFC + lowercase + regex word tokens",
        },
        "index_build_seconds": build_seconds,
        "results": results,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mean_seconds = (
        sum(row["search_seconds"] for row in results) / len(results)
        if results else 0
    )

    print(f"\n=== {dataset.upper()} ===")
    print(f"Queries: {len(results)}")
    print(f"Index build: {build_seconds:.3f} seconds")
    print(f"Mean search: {mean_seconds:.4f} seconds")
    print(f"Saved: {output_path}")

    if results:
        example = results[0]
        print(f"\nExample query: {example['query']}")
        print(f"Candidates: {example['candidate_count']}")

        for hit in example["hits"][:3]:
            print(
                f"\n#{hit['rank']} score={hit['score']:.4f} "
                f"id={hit['evidence_id']}"
            )
            print(hit["text"][:400])


def main():
    selection = json.loads(
        (ROOT / "configs" / "smoke_queries.json")
        .read_text(encoding="utf-8")
    )

    run_dataset("qasper", selection["qasper_query_ids"])
    run_dataset("scifact", selection["scifact_query_ids"])

    print("\nBM25 smoke retrieval completed.")


if __name__ == "__main__":
    main()