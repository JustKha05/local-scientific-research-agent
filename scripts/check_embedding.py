import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print(f"Loading: {MODEL_ID}")
    print("Device: CPU")

    model = SentenceTransformer(MODEL_ID, device="cpu")

    documents = [
        "The method combines rule-based and dictionary-based approaches.",
        "The experiment evaluates translation quality on three datasets.",
        "The authors thank the reviewers for their helpful comments.",
    ]
    query = "What approaches does the system combine?"

    started = time.perf_counter()

    document_embeddings = model.encode(
        documents,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.perf_counter() - started

    if document_embeddings.shape != (3, 384):
        raise ValueError(
            f"Unexpected embedding shape: {document_embeddings.shape}"
        )

    if not np.isfinite(document_embeddings).all():
        raise ValueError("Document embeddings contain invalid values.")

    if not np.isfinite(query_embedding).all():
        raise ValueError("Query embedding contains invalid values.")

    if not np.allclose(
        np.linalg.norm(document_embeddings, axis=1),
        1.0,
        atol=1e-5,
    ):
        raise ValueError("Document embeddings are not normalized.")

    # Dot product equals cosine similarity for normalized vectors.
    scores = document_embeddings @ query_embedding[0]
    ranking = np.argsort(-scores, kind="stable")

    print(f"\nDocument shape: {document_embeddings.shape}")
    print(f"Query shape: {query_embedding.shape}")
    print(f"Max sequence length: {model.max_seq_length}")
    print(f"Encoding time: {elapsed:.3f} seconds")

    print("\nRanking:")
    for rank, index in enumerate(ranking, start=1):
        print(
            f"{rank}. score={scores[index]:.4f} "
            f"text={documents[index]}"
        )

    report = {
        "model_id": MODEL_ID,
        "device": "cpu",
        "sentence_transformers_version": version("sentence-transformers"),
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
        "numpy_version": version("numpy"),
        "embedding_dimension": int(document_embeddings.shape[1]),
        "max_seq_length": model.max_seq_length,
        "normalize_embeddings": True,
        "encoding_seconds": elapsed,
        "query": query,
        "ranked_documents": [
            {
                "text": documents[index],
                "score": float(scores[index]),
            }
            for index in ranking
        ],
    }

    output = ROOT / "evaluation" / "reports" / "embedding_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output}")
    print("Embedding execution check: PASSED")
    print("This synthetic example is not a retrieval benchmark.")


if __name__ == "__main__":
    main()