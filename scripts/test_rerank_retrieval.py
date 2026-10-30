"""Small offline fixtures; never download or instantiate a real model."""
import argparse
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts.benchmark_common import file_hash, write_new
from scripts.evaluate_retrieval_benchmark import validate_report
from scripts.fuse_retrieval_rrf import validate_sources
from scripts.rerank_retrieval import candidate_union, load_model, run, score_candidates

SHA = "a" * 40


def hit(eid, rank):
    return dict(evidence_id=eid, paper_id="p1", text=f"original {eid}",
                title="title", rank=rank, score=-rank)


def source(method, dataset="scifact", count=3):
    return dict(schema_version=1, dataset=dataset, split="dev",
                config=dict(retriever=method, top_k=50), manifest_sha256="manifest",
                source_hashes=dict(queries_sha256="queries", documents_sha256="documents"),
                results=[dict(query_id="q1", query="original question", candidate_count=count,
                              scope="p1" if dataset == "qasper" else "corpus",
                              hits=[hit(str(i), i + 1) for i in range(min(count, 50))])])


class FakeTokenizer:
    truncation_side = "right"

    def __call__(self, queries, texts, **kwargs):
        assert kwargs["truncation"] is False
        assert queries == ["original question"] * len(texts)
        return {"length": [20] * len(texts)}


class FakeModel:
    num_labels = 1
    tokenizer = FakeTokenizer()

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        assert kwargs["processing_kwargs"]["text"]["truncation"] == "longest_first"
        assert kwargs["apply_softmax"] is False
        # All scores tie, intentionally exercises lexical evidence-ID ordering.
        return np.ones((len(pairs), 1), dtype=np.float32)


class RerankerTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures independent of installed torch/CUDA binaries as well as weights.
        fake_torch = types.SimpleNamespace(
            float32="float32", nn=types.SimpleNamespace(Identity=lambda: lambda x: x),
            get_num_threads=lambda: 1, version=types.SimpleNamespace(cuda=None),
            cuda=types.SimpleNamespace(is_available=lambda: False))
        modules = patch.dict("sys.modules", {"torch": fake_torch})
        modules.start()
        self.addCleanup(modules.stop)

    def test_union_depth_dedup_and_no_rrf_pruning(self):
        left, right = source("bm25", count=150), source("dense", count=150)
        right["results"][0]["hits"] = [hit(str(i), rank) for rank, i in enumerate(range(50, 100), 1)]
        lrows, rrows = validate_sources(left, right, "scifact")
        union = candidate_union(lrows["q1"], rrows["q1"])
        self.assertEqual(len(union), 100)
        self.assertEqual(len(candidate_union(lrows["q1"], lrows["q1"])), 50)
        # Source reports deeper than 50 still contribute only their first 50.
        extra = copy.deepcopy(lrows["q1"])
        extra["hits"].append(hit("excluded", 51))
        self.assertNotIn("excluded", {d["evidence_id"] for d in candidate_union(extra, rrows["q1"])})
        short = copy.deepcopy(lrows["q1"])
        short["hits"] = short["hits"][:10]
        with self.assertRaisesRegex(ValueError, "insufficient hits"):
            candidate_union(short, rrows["q1"])

    def test_conflict_and_source_mismatch(self):
        left, right = source("bm25"), source("dense")
        for mutate in (lambda r: r["results"][0]["hits"][0].update(text="conflict"),
                       lambda r: r.update(manifest_sha256="other"),
                       lambda r: r["results"][0].update(query="stale"),
                       lambda r: r["results"][0]["hits"][1].update(evidence_id="0")):
            bad = copy.deepcopy(right)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_sources(left, bad, "scifact")

    def test_scoring_top_k_ties_payload_and_invalid_scores(self):
        candidates = [{k: v for k, v in hit(e, 1).items() if k not in ("rank", "score")}
                      for e in ("z", "a", "b")]
        model = FakeModel()
        with patch.object(model, "predict", return_value=np.array([1., 2., 2.])):
            hits = score_candidates(model, "original question", candidates, 2, 16, 2, None)
        self.assertEqual([h["evidence_id"] for h in hits], ["a", "b"])
        self.assertEqual([h["rank"] for h in hits], [1, 2])
        self.assertEqual(hits[0]["text"], "original a")
        for scores in (np.ones((3, 2)), np.ones(2), np.array([1., np.nan, 0.]),
                       np.array([np.inf, 1., 0.])):
            with patch.object(model, "predict", return_value=scores), self.assertRaises(ValueError):
                score_candidates(model, "question", candidates, 2, 16, 2, None)

    def test_roundtrip_provenance_evaluator_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ("qasper", "scifact"):
                for method in ("bm25", "dense"):
                    write_new(root / method / f"{dataset}_retrieval.json", source(method, dataset))
                args = argparse.Namespace(dataset=dataset, bm25_dir=root / "bm25", dense_dir=root / "dense",
                    report_dir=root / "out", model_id="fake/model", revision=SHA,
                    device="cpu", batch_size=2, max_length=16, top_k=10)
                model = FakeModel()
                with patch("scripts.rerank_retrieval.load_model", return_value=model):
                    run(args)
                output = root / "out" / f"{dataset}_retrieval.json"
                report = json.loads(output.read_text())
                docs = {str(i): {k: v for k, v in hit(str(i), 1).items()
                                 if k not in ("rank", "score")} for i in range(3)}
                validate_report(report, dataset, "dev", {"q1": dict(text="original question", paper_id="p1")},
                                docs, source("bm25")["source_hashes"])
                self.assertEqual(report["config"]["retriever"], "cross_encoder")
                self.assertEqual(report["config"]["tokenizer_revision"], SHA)
                self.assertEqual(report["results"][0]["rerank_candidate_count"], 3)
                self.assertEqual(report["truncation"]["over_limit"], 3)
                self.assertEqual(report["truncation"]["overflow_tokens"], 12)
                self.assertEqual(model.pairs, [("original question", f"original {i}") for i in range(3)])
                for method in ("bm25", "dense"):
                    self.assertEqual(report["rerank_sources"][method]["sha256"],
                                     file_hash(root / method / f"{dataset}_retrieval.json"))
                self.assertIn("torch", report["versions"])
                self.assertEqual(report["hardware"]["device"], "cpu")
                before = output.read_bytes()
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    run(args)
                self.assertEqual(output.read_bytes(), before)
                args.revision = "main"
                with self.assertRaisesRegex(ValueError, "commit SHA"):
                    run(args)

    def test_model_constructor_pins_revision_without_replacing_head(self):
        args = argparse.Namespace(model_id="fake/model", revision=SHA, device="cpu", max_length=16)
        model = FakeModel()
        snapshot = Path("cache") / "snapshots" / SHA
        download = Mock(return_value=str(snapshot))
        def constructor(*a, **kw):
            self.assertEqual(a, (str(snapshot),))
            self.assertIs(kw["local_files_only"], True)
            self.assertIs(kw["trust_remote_code"], False)
            self.assertEqual(kw["model_kwargs"], {"dtype": "float32"})
            self.assertEqual(kw["device"], "cpu")
            self.assertEqual(kw["max_length"], 16)
            self.assertNotIn("revision", kw)
            self.assertNotIn("num_labels", kw)
            return model
        encoder = Mock(side_effect=constructor)
        with patch.dict("sys.modules", {
                "sentence_transformers": types.SimpleNamespace(CrossEncoder=encoder),
                "huggingface_hub": types.SimpleNamespace(snapshot_download=download)}):
            self.assertIs(load_model(args), model)
            download.assert_called_once_with(repo_id="fake/model", revision=SHA)
            self.assertEqual(model.tokenizer.truncation_side, "right")
            model.num_labels = 2
            with self.assertRaisesRegex(ValueError, "one output"):
                load_model(args)
            encoder.reset_mock()
            download.return_value = str(snapshot.with_name("b" * 40))
            with self.assertRaisesRegex(ValueError, "snapshot revision mismatch"):
                load_model(args)
            encoder.assert_not_called()
            download.reset_mock()
            args.revision = "main"
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                load_model(args)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
