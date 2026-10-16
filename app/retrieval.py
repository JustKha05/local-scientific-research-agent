import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    normalized = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"\b\w+\b", normalized)


class BM25Retriever:
    def __init__(self, documents):
        # Stable ordering also provides deterministic tie-breaking.
        self.documents = sorted(
            documents,
            key=lambda document: document["evidence_id"],
        )

        if not self.documents:
            raise ValueError("Cannot index an empty document collection.")

        tokens = [
            tokenize(document["text"])
            for document in self.documents
        ]

        if not any(tokens):
            raise ValueError("Document collection has no searchable tokens.")

        self.index = BM25Okapi(
            tokens,
            k1=1.5,
            b=0.75,
            epsilon=0.25,
        )

    def search(self, query, top_k=10):
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Avoid arbitrary results when no query term exists in the index.
        if not any(token in self.index.idf for token in query_tokens):
            return []

        scores = self.index.get_scores(query_tokens)
        indices = np.argsort(-scores, kind="stable")[:top_k]

        return [
            {
                "rank": rank,
                "score": float(scores[index]),
                **self.documents[int(index)],
            }
            for rank, index in enumerate(indices, start=1)
        ]