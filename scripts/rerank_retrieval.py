"""Gold-free cross-encoder reranking of the BM25/dense top-50 union."""
import argparse
import hashlib
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.benchmark_common import file_hash, require, write_new
from scripts.fuse_retrieval_rrf import validate_sources
from scripts.run_retrieval_benchmark import versions

SOURCE_DEPTH = 50
TIMING = ("Reranking only: pair construction, tokenization, cross-encoder inference, "
          "sorting and top-k; CUDA synchronized; no warmup. Excludes source retrieval, "
          "union/validation, truncation audit, model loading and report I/O. "
          "Not end-to-end pipeline latency.")
TRUNCATION = ("Original query + original evidence text, no chunking or manual text edits; "
              "tokenizer pair longest_first truncation from the right, including special tokens.")


def candidate_union(left, right):
    """Sources must have passed validate_sources before calling this helper."""
    candidates = {}
    for row in (left, right):
        require(len(row["hits"]) >= min(SOURCE_DEPTH, row["candidate_count"]),
                "Source has insufficient hits for top-50 protocol; generate new depth-50 reports")
        for hit in row["hits"][:SOURCE_DEPTH]:
            candidates[hit["evidence_id"]] = {
                k: v for k, v in hit.items() if k not in ("rank", "score")}
    require(len(candidates) <= min(100, left["candidate_count"]), "Invalid candidate union size")
    return [candidates[eid] for eid in sorted(candidates)]


def truncation_stats(tokenizer, query, candidates, max_length, batch_size):
    lengths = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        encoded = tokenizer([query] * len(batch), [d["text"] for d in batch],
                            truncation=False, padding=False, add_special_tokens=True,
                            return_length=True, verbose=False)
        lengths.extend(encoded["length"])
    return {"pairs": len(lengths), "over_limit": sum(n > max_length for n in lengths),
            "longest_pair_tokens": max(lengths, default=0),
            "overflow_tokens": sum(max(0, n - max_length) for n in lengths)}


def score_candidates(model, query, candidates, batch_size, max_length, top_k, activation_fn):
    import numpy as np
    if not candidates:
        return []
    scores = np.asarray(model.predict(
        [(query, d["text"]) for d in candidates], batch_size=batch_size,
        show_progress_bar=False, convert_to_numpy=True, activation_fn=activation_fn,
        apply_softmax=False, processing_kwargs={"text": {
            "max_length": max_length, "truncation": "longest_first", "padding": True}}))
    require(scores.shape in ((len(candidates),), (len(candidates), 1)),
            f"Expected one score per candidate, got {scores.shape}")
    scores = scores.reshape(-1)
    require(np.isfinite(scores).all(), "Non-finite cross-encoder scores")
    order = sorted(range(len(candidates)),
                   key=lambda i: (-float(scores[i]), candidates[i]["evidence_id"]))[:top_k]
    return [{**candidates[i], "rank": rank, "score": float(scores[i])}
            for rank, i in enumerate(order, 1)]


def load_model(a):
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder
    require(re.fullmatch(r"[0-9a-f]{40}", a.revision) is not None,
            "--revision must be a resolved 40-character commit SHA, never main/tag")
    require(a.device != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    snapshot = Path(snapshot_download(repo_id=a.model_id, revision=a.revision))
    require(snapshot.name == a.revision, "Downloaded snapshot revision mismatch")
    # Do not override num_labels: that could replace the pretrained classification head.
    model = CrossEncoder(str(snapshot), local_files_only=True, device=a.device,
                         max_length=a.max_length, trust_remote_code=False,
                         model_kwargs={"dtype": torch.float32})
    require(model.num_labels == 1, "Reranker must have exactly one output")
    model.tokenizer.truncation_side = "right"
    return model


def run(a):
    require(re.fullmatch(r"[0-9a-f]{40}", a.revision) is not None,
            "--revision must be a resolved 40-character commit SHA, never main/tag")
    require(all(type(n) is int and n > 0 for n in (a.batch_size, a.max_length, a.top_k)),
            "batch-size, max-length and top-k must be positive")
    output = a.report_dir / f"{a.dataset}_retrieval.json"
    require(not output.exists(), f"Refusing to overwrite {output}")
    reports, sources = [], {}
    for method, directory in (("bm25", a.bm25_dir), ("dense", a.dense_dir)):
        path = directory / f"{a.dataset}_retrieval.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        reports.append(report)
        sources[method] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                           "configured_top_k": report["config"]["top_k"],
                           "used_depth": SOURCE_DEPTH,
                           "returned_depth_by_query": {r["query_id"]: len(r["hits"])
                                                       for r in report["results"]}}
    left, right = validate_sources(*reports, a.dataset)
    unions = {qid: candidate_union(row, right[qid]) for qid, row in left.items()}
    # All source/protocol checks finish before loading a potentially large model.
    import torch
    load_tick = time.perf_counter()
    model = load_model(a)
    require(model.num_labels == 1, "Reranker must have exactly one output")
    load_seconds = time.perf_counter() - load_tick
    hardware = {"platform": platform.platform(), "processor": platform.processor(),
                "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                "device": a.device, "cuda_version": torch.version.cuda, "gpu": None}
    if a.device == "cuda":
        hardware.update(gpu=torch.cuda.get_device_name(0),
                        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
    results = []
    for qid, row in left.items():
        candidates = unions[qid]
        stats = truncation_stats(model.tokenizer, row["query"], candidates, a.max_length, a.batch_size)
        if a.device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        hits = score_candidates(model, row["query"], candidates, a.batch_size,
                                a.max_length, a.top_k, torch.nn.Identity())
        if a.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        result = {k: row[k] for k in ("query_id", "query", "scope", "candidate_count")}
        result.update(hits=hits, search_seconds=seconds, rerank_seconds=seconds,
                      rerank_candidate_count=len(candidates), truncation=stats,
                      source_candidate_counts={"bm25": min(50, len(row["hits"])),
                                               "dense": min(50, len(right[qid]["hits"]))})
        results.append(result)
    config = {"retriever": "cross_encoder", "model_id": a.model_id, "revision": a.revision,
              "tokenizer_id": a.model_id, "tokenizer_revision": a.revision,
              "tokenizer_class": type(model.tokenizer).__name__, "max_length": a.max_length,
              "batch_size": a.batch_size, "device": a.device, "top_k": a.top_k,
              "source_depth": SOURCE_DEPTH, "candidate_policy": "union by evidence_id; no RRF pruning",
              "score": "single raw relevance logit; Identity activation; no softmax",
              "dtype": "float32", "truncation_policy": TRUNCATION,
              "truncation_side": "right", "trust_remote_code": False}
    report = {k: reports[0][k] for k in
              ("schema_version", "dataset", "split", "source_hashes", "manifest_sha256")}
    report.update(created_at=datetime.now(timezone.utc).isoformat(), config=config,
                  rerank_sources=sources, versions=versions(), hardware=hardware,
                  script_sha256=file_hash(__file__),
                  validation_script_sha256=file_hash(Path(__file__).with_name("fuse_retrieval_rrf.py")),
                  timing_protocol=TIMING, model_load_seconds=load_seconds,
                  rerank_seconds=sum(r["rerank_seconds"] for r in results),
                  truncation={"policy": TRUNCATION, "max_length": a.max_length,
                              **{k: sum(r["truncation"][k] for r in results)
                                 for k in ("pairs", "over_limit", "overflow_tokens")},
                              "longest_pair_tokens": max(r["truncation"]["longest_pair_tokens"] for r in results)},
                  results=results)
    if a.device == "cuda":
        hardware["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    write_new(output, report)
    print(f"Saved {output}; queries={len(results)}; reranking only, excludes retrieval")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("qasper", "scifact"), required=True)
    for name in ("bm25-dir", "dense-dir", "report-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True, help="Resolved immutable 40-character commit SHA")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    run(p.parse_args())


if __name__ == "__main__":
    main()
