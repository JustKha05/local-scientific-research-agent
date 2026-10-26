import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "data" / "embeddings"
REPORT_DIR = ROOT / "evaluation" / "reports" / "dense_smoke"
CONFIG_PATH = ROOT / "configs" / "dense_model.json"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
TOP_K = 10


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


def json_hash(value):
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_model_config():
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        # Resolve once, then persist the exact model revision.
        revision = model_info(MODEL_ID).sha
        require(bool(revision), "Cannot resolve model revision.")

        config = {
            "model_id": MODEL_ID,
            "revision": revision,
            "device": "cpu",
            "normalize_embeddings": True,
            "batch_size": BATCH_SIZE,
            "top_k": TOP_K,
        }

        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    require(config["model_id"] == MODEL_ID, "Unexpected model ID.")
    require(bool(config["revision"]), "Missing model revision.")
    require(config["device"] == "cpu", "Expected CPU configuration.")
    require(
        config["normalize_embeddings"] is True,
        "Normalized embeddings are required.",
    )
    require(config["top_k"] == TOP_K, "Unexpected top_k.")

    return config


def count_truncated_documents(model, documents):
    lengths = []

    for start in range(0, len(documents), 256):
        texts = [
            document["text"]
            for document in documents[start:start + 256]
        ]

        encoded = model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
            verbose=False,
        )
        lengths.extend(encoded["length"])

    return {
        "document_count": len(documents),
        "max_seq_length": model.max_seq_length,
        "documents_over_limit": sum(
            length > model.max_seq_length for length in lengths
        ),
        "longest_document_tokens": max(lengths, default=0),
        "policy": "Encode the beginning of each original unit; truncate overflow.",
    }


def load_or_encode(model, documents, config, dataset):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_id": config["model_id"],
        "revision": config["revision"],
        "max_seq_length": model.max_seq_length,
        "normalize_embeddings": True,
        "dtype": "float32",
        "batch_size": config["batch_size"],
        "sentence_transformers_version": version("sentence-transformers"),
        "transformers_version": version("transformers"),
        "torch_version": version("torch"),
        "numpy_version": version("numpy"),
        "documents_sha256": json_hash([
            {
                "evidence_id": document["evidence_id"],
                "text": document["text"],
            }
            for document in documents
        ]),
    }

    cache_key = json_hash(metadata)
    vector_path = CACHE_DIR / f"{dataset}_{cache_key}.npy"
    metadata_path = CACHE_DIR / f"{dataset}_{cache_key}.json"

    started = time.perf_counter()

    if vector_path.exists() and metadata_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        require(saved["metadata"] == metadata, "Cache metadata mismatch.")

        embeddings = np.load(vector_path, allow_pickle=False)
        truncation = saved["truncation"]
        cache_hit = True
        print(f"{dataset}: using cached embeddings")
    else:
        print(f"{dataset}: checking token lengths...")
        truncation = count_truncated_documents(model, documents)

        print(f"{dataset}: encoding {len(documents)} documents...")
        embeddings = model.encode(
            [document["text"] for document in documents],
            batch_size=config["batch_size"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        require(
            np.isfinite(embeddings).all(),
            "Embeddings contain invalid values.",
        )

        np.save(vector_path, embeddings, allow_pickle=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "truncation": truncation,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        cache_hit = False

    require(
        embeddings.shape
        == (len(documents), model.get_sentence_embedding_dimension()),
        f"Invalid cached shape: {embeddings.shape}",
    )
    require(np.isfinite(embeddings).all(), "Invalid embedding values.")
    require(
        np.allclose(
            np.linalg.norm(embeddings, axis=1),
            1.0,
            atol=1e-4,
        ),
        "Document embeddings are not normalized.",
    )

    return embeddings, {
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "load_or_encode_seconds": time.perf_counter() - started,
        "metadata": metadata,
        "truncation": truncation,
    }


def run_dataset(model, config, dataset, selected_ids):
    directory = PROCESSED / dataset
    query_path = directory / "train" / "queries.jsonl"

    query_index = {
        row["query_id"]: row
        for row in read_jsonl(query_path)
    }

    require(
        len(selected_ids) == len(set(selected_ids)),
        "Duplicate selected query IDs.",
    )
    require(
        set(selected_ids) <= set(query_index),
        f"Unknown query IDs for {dataset}.",
    )

    queries = [query_index[query_id] for query_id in selected_ids]

    document_path = (
        directory / "train" / "documents.jsonl"
        if dataset == "qasper"
        else directory / "documents.jsonl"
    )
    documents = read_jsonl(document_path)

    if dataset == "qasper":
        selected_papers = {query["paper_id"] for query in queries}
        documents = [
            document
            for document in documents
            if document["paper_id"] in selected_papers
        ]

    documents.sort(key=lambda document: document["evidence_id"])

    require(bool(documents), f"No documents for {dataset}.")
    require(
        len({doc["evidence_id"] for doc in documents}) == len(documents),
        "Duplicate document IDs.",
    )

    embeddings, cache_report = load_or_encode(
        model, documents, config, dataset
    )

    paper_indices = {}

    if dataset == "qasper":
        for index, document in enumerate(documents):
            paper_indices.setdefault(document["paper_id"], []).append(index)

    results = []

    for query in queries:
        started = time.perf_counter()

        query_vector = model.encode(
            [query["text"]],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        require(
            np.isfinite(query_vector).all(),
            f"Invalid query embedding: {query['query_id']}",
        )

        if dataset == "qasper":
            scope = query["paper_id"]
            candidates = np.asarray(paper_indices[scope], dtype=np.int64)
        else:
            scope = "corpus"
            candidates = np.arange(len(documents))

        scores = embeddings[candidates] @ query_vector
        order = np.argsort(-scores, kind="stable")[:TOP_K]

        hits = []

        for rank, local_index in enumerate(order, start=1):
            document_index = int(candidates[local_index])
            hits.append({
                "rank": rank,
                "score": float(scores[local_index]),
                **documents[document_index],
            })

        results.append({
            "query_id": query["query_id"],
            "query": query["text"],
            "scope": scope,
            "candidate_count": len(candidates),
            # Includes query encoding, scoring and top-k selection.
            "search_seconds": time.perf_counter() - started,
            "hits": hits,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / f"{dataset}_retrieval.json"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split": "train",
        "purpose": "Dense retrieval smoke test; not final benchmark",
        "python_version": platform.python_version(),
        "config": {
            **config,
            "max_seq_length": model.max_seq_length,
            "indexed_field": "text",
            "similarity": "cosine via normalized dot product",
        },
        "embedding_cache": cache_report,
        "results": results,
    }

    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mean_seconds = (
        sum(row["search_seconds"] for row in results) / len(results)
        if results else 0
    )

    print(f"\n=== {dataset.upper()} ===")
    print(f"Queries: {len(results)}")
    print(f"Encoded documents: {len(documents)}")
    print(f"Cache hit: {cache_report['cache_hit']}")
    print(
        "Documents over token limit: "
        f"{cache_report['truncation']['documents_over_limit']}"
    )
    print(f"Mean search including query encoding: {mean_seconds:.4f}s")
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("qasper", "scifact", "both"),
        default="both",
    )
    args = parser.parse_args()

    selection = json.loads(
        (ROOT / "configs" / "smoke_queries.json")
        .read_text(encoding="utf-8")
    )
    config = load_model_config()

    print(f"Model: {config['model_id']}")
    print(f"Revision: {config['revision']}")
    print("Device: CPU")

    model = SentenceTransformer(
        config["model_id"],
        revision=config["revision"],
        device="cpu",
    )

    datasets = (
        ("qasper", "scifact")
        if args.dataset == "both"
        else (args.dataset,)
    )

    for dataset in datasets:
        run_dataset(
            model,
            config,
            dataset,
            selection[f"{dataset}_query_ids"],
        )

    print("\nDense smoke retrieval completed.")


if __name__ == "__main__":
    main()