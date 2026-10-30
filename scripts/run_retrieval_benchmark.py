"""Full-split retrieval. This module never opens a gold file."""
import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from scripts.benchmark_common import (ROOT, digest, file_hash, load_inputs,
                                      require, validate_manifest, write_new)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_SEQ_LENGTH = 256  # Verified against the existing dense smoke reports.


def versions():
    result = {"python": platform.python_version()}
    for name in ("numpy", "rank-bm25", "torch", "transformers", "sentence-transformers", "huggingface-hub"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def truncation(model, texts):
    lengths = []
    for start in range(0, len(texts), 256):
        lengths.extend(model.tokenizer(texts[start:start + 256], truncation=False,
                                      padding=False, add_special_tokens=True,
                                      return_length=True, verbose=False)["length"])
    return {"count": len(texts), "over_limit": sum(n > MAX_SEQ_LENGTH for n in lengths),
            "longest_tokens": max(lengths, default=0), "max_seq_length": MAX_SEQ_LENGTH,
            "policy": "Beginning of original unit, overflow truncated; no chunking"}


def cached_embeddings(model, documents, config, cache_dir):
    import numpy as np
    ids = [d["evidence_id"] for d in documents]
    metadata = {"schema_version": 1, "config": config, "evidence_ids": ids,
                "documents_sha256": digest(documents), "versions": versions()}
    key = digest(metadata)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path, vector_path = directory / f"{key}.json", directory / f"{key}.npy"
    require(meta_path.exists() == vector_path.exists(), "Incomplete cache; move it aside explicitly")
    started = time.perf_counter()
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf8"))
        require(saved["metadata"] == metadata, "Cache metadata or evidence ID order mismatch")
        require(saved["vectors_sha256"] == file_hash(vector_path), "Cache vector hash mismatch")
        vectors = np.load(vector_path, allow_pickle=False)
        stats, hit = saved["truncation"], True
    else:
        stats = truncation(model, [d["text"] for d in documents])
        vectors = model.encode([d["text"] for d in documents], batch_size=config["batch_size"],
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=True).astype(np.float32)
        hit = False
    require(vectors.dtype == np.float32 and vectors.shape ==
            (len(documents), model.get_sentence_embedding_dimension()), "Invalid cache shape/dtype")
    require(np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4),
            "Invalid or unnormalized embeddings")
    if not hit:
        with vector_path.open("xb") as stream:
            np.save(stream, vectors, allow_pickle=False)
        write_new(meta_path, {"metadata": metadata, "truncation": stats,
                              "vectors_sha256": file_hash(vector_path)})
    return vectors, {"key": key, "cache_hit": hit, "metadata": metadata,
                     "truncation": stats, "seconds": time.perf_counter() - started}


def candidate_indices(documents, query, dataset):
    return [i for i, d in enumerate(documents)
            if dataset != "qasper" or d["paper_id"] == query["paper_id"]]


def run(a):
    import numpy as np
    started = time.perf_counter()
    output = a.report_dir / f"{a.dataset}_retrieval.json"
    require(not output.exists(), f"Refusing to overwrite {output}")
    queries, documents, hashes = load_inputs(a.processed_dir, a.dataset, a.split)
    manifest_hash = None
    if a.split == "dev":
        require(a.query_manifest is not None, "Dev requires an existing --query-manifest")
        manifest = json.loads(a.query_manifest.read_text(encoding="utf8"))
        validate_manifest(manifest, queries, a.dataset, hashes["queries_sha256"])
        manifest_hash = file_hash(a.query_manifest)
    config = {"retriever": a.retriever, "device": a.device, "top_k": a.top_k,
              "indexed_field": "text", "batch_size": a.batch_size}
    hardware = {"platform": platform.platform(), "processor": platform.processor(),
                "cpu_count": os.cpu_count(), "gpu": None}
    model, cache, query_truncation = None, None, {"policy": "BM25: no truncation"}
    build_started = time.perf_counter()
    retrievers, scopes = {}, {}
    for q in queries:
        scope = q["paper_id"] if a.dataset == "qasper" else "corpus"
        if scope not in scopes:
            scopes[scope] = candidate_indices(documents, q, a.dataset)
            require(bool(scopes[scope]), f"Empty candidate scope {scope}")
    if a.retriever == "bm25":
        from app.retrieval import BM25Retriever
        require(a.device == "cpu", "BM25 supports CPU only")
        config.update(k1=1.5, b=.75, epsilon=.25,
                      tokenization="NFC + lowercase + regex word tokens")
        retrievers = {s: BM25Retriever([documents[i] for i in indices]) for s, indices in scopes.items()}
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        baseline = json.loads((ROOT / "configs/dense_model.json").read_text(encoding="utf8"))
        require(baseline["model_id"] == MODEL and baseline["revision"] == REVISION
                and baseline["normalize_embeddings"] is True, "Baseline config changed; review protocol")
        require(a.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable: enable GPU or explicitly use --device cpu")
        hardware.update(cuda_version=torch.version.cuda, torch_threads=torch.get_num_threads())
        if a.device == "cuda":
            hardware["gpu"] = torch.cuda.get_device_name(0)
            hardware["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        config.update(model_id=MODEL, revision=REVISION, max_seq_length=MAX_SEQ_LENGTH,
                      normalize_embeddings=True, dtype="float32", similarity="normalized dot product")
        model = SentenceTransformer(MODEL, revision=REVISION, device=a.device)
        require(model.max_seq_length == MAX_SEQ_LENGTH, "Model max_seq_length differs from baseline")
        vectors, cache = cached_embeddings(model, documents,
            {k: v for k, v in config.items() if k not in ("top_k",)}, a.cache_dir)
        query_truncation = truncation(model, [q["text"] for q in queries])
    build_seconds = time.perf_counter() - build_started
    results = []
    for q in queries:
        scope = q["paper_id"] if a.dataset == "qasper" else "corpus"
        tick = time.perf_counter()
        if model is None:
            hits = retrievers[scope].search(q["text"], top_k=a.top_k)
        else:
            vector = model.encode([q["text"]], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            require(np.isfinite(vector).all(), "Invalid query vector")
            candidates = np.asarray(scopes[scope], dtype=np.int64)
            scores = vectors[candidates] @ vector
            order = np.argsort(-scores, kind="stable")[:a.top_k]
            hits = [{"rank": rank, "score": float(scores[i]), **documents[int(candidates[i])]}
                    for rank, i in enumerate(order, 1)]
        results.append({"query_id": q["query_id"], "query": q["text"], "scope": scope,
                        "candidate_count": len(scopes[scope]), "search_seconds": time.perf_counter() - tick,
                        "hits": hits})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    latencies = [r["search_seconds"] for r in results]
    report = {"schema_version": 1, "dataset": a.dataset, "split": a.split,
              "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
              "source_hashes": hashes, "manifest_sha256": manifest_hash, "versions": versions(),
              "hardware": hardware, "git": {"commit": commit, "dirty": dirty},
              "script_sha256": file_hash(__file__), "embedding_cache": cache,
              "query_truncation": query_truncation, "index_build_seconds": build_seconds,
              "total_seconds": time.perf_counter() - started,
              "latency_seconds": {"p50": float(np.percentile(latencies, 50)),
                                  "p95": float(np.percentile(latencies, 95)),
                                  "protocol": "No warmup; query encoding + CPU scoring + top-k; cache build separate"},
              "results": results}
    if a.device == "cuda":
        report["hardware"]["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    write_new(output, report)
    print(f"Saved {output}; queries={len(results)}, seconds={report['total_seconds']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("qasper", "scifact"), required=True)
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--retriever", choices=("bm25", "dense"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/embeddings/benchmark")
    p.add_argument("--query-manifest", type=Path)
    p.add_argument("--report-dir", type=Path, required=True)
    a = p.parse_args()
    require(a.top_k > 0 and a.batch_size > 0, "top-k/batch-size must be positive")
    run(a)


if __name__ == "__main__":
    main()
