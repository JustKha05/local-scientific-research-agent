"""Small offline fixtures only. No model construction or network access."""
import argparse
import contextlib
import copy
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_bm25_smoke as smoke
from scripts import evaluate_scifact_rationales as rationale_smoke
from scripts import compare_retrieval_benchmarks as comparison
from scripts.benchmark_common import (digest, file_hash, load_inputs, make_manifest,
                                      validate_manifest, write_new)
from scripts.evaluate_retrieval_benchmark import evaluate, score_query, validate_report
from scripts.run_retrieval_benchmark import cached_embeddings, candidate_indices, run


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")


def fixture(dataset):
    docs = [{"evidence_id": f"e{i}", "paper_id": "p1" if i < 4 else "p2",
             "text": f"unique token{i} evidence"} for i in range(1, 6)]
    q = {"query_id": "q1", "text": "token1", "paper_id": "p1"}
    gold = ({"query_id": "q1", "annotations": [
        {"answer": {"unanswerable": False}, "figure_table_evidence": [],
         "unmatched_text_evidence": [], "evidence_ids": ids}
        for ids in (["e1", "e2"], ["e3"])]} if dataset == "qasper" else
        {"query_id": "q1", "evidence_sets": [
            {"paper_id": "p1", "evidence_ids": ids} for ids in (["e1", "e2"], ["e3"])]})
    return docs, q, gold


class BenchmarkTests(unittest.TestCase):
    def test_scope_and_gold_free_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ("qasper", "scifact"):
                docs, q, _ = fixture(dataset)
                base = root / "processed" / dataset
                jsonl(base / "dev/queries.jsonl", [q])
                jsonl(base / ("dev/documents.jsonl" if dataset == "qasper" else "documents.jsonl"), docs)
                queries, documents, hashes = load_inputs(root / "processed", dataset, "dev")
                manifest_path = root / f"{dataset}.json"
                write_new(manifest_path, make_manifest(queries, dataset, hashes["queries_sha256"]))
                a = argparse.Namespace(dataset=dataset, split="dev", processed_dir=root / "processed",
                    query_manifest=manifest_path, report_dir=root / dataset, retriever="bm25",
                    device="cpu", top_k=10, batch_size=16, cache_dir=root / "cache")
                # No gold file exists: retrieval must still work.
                run(a)
                report = json.loads((a.report_dir / f"{dataset}_retrieval.json").read_text())
                validate_report(report, dataset, "dev", {"q1": q},
                                {d["evidence_id"]: d for d in docs}, hashes)
                self.assertEqual(report["results"][0]["candidate_count"], 3 if dataset == "qasper" else 5)
                self.assertEqual(len(candidate_indices(documents, q, dataset)), 3 if dataset == "qasper" else 5)

    def test_manifest_immutability_and_groups(self):
        queries = [{"query_id": str(i), "paper_id": str(i // 3)} for i in range(30)]
        m = make_manifest(queries, "qasper", "hash")
        self.assertEqual(m, make_manifest(list(reversed(queries)), "qasper", "hash"))
        validate_manifest(m, queries, "qasper", "hash")
        self.assertEqual([m["counts"][p]["groups"] for p in ("diagnosis", "calibration", "reporting")], [4, 2, 4])
        broken = copy.deepcopy(m)
        broken["assignments"][0]["partition"] = next(p for p in ("diagnosis", "calibration", "reporting")
                                                      if p != broken["assignments"][0]["partition"])
        with self.assertRaises(ValueError):
            validate_manifest(broken, queries, "qasper", "hash")
        with self.assertRaises(ValueError):
            validate_manifest(m, queries, "qasper", "changed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_new(path, m)
            with self.assertRaises(FileExistsError):
                write_new(path, m)
        sf = make_manifest(queries, "scifact", "hash")
        self.assertEqual(sum(c["groups"] for c in sf["counts"].values()), 30)

    def test_alternative_and_metric_parity_with_real_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/smoke_queries.json").write_text(json.dumps(
                {"qasper_query_ids": ["q1"], "scifact_query_ids": ["q1"]}))
            for dataset in ("qasper", "scifact"):
                docs, q, gold = fixture(dataset)
                # First retrieve e3: sufficient alternative, but only 1/3 of SciFact union.
                hits = [{"rank": 1, "score": 1., **docs[2]}]
                report = {"dataset": dataset, "split": "train", "config": {"top_k": 10},
                          "results": [{"query_id": "q1", "query": q["text"],
                                       "scope": "p1" if dataset == "qasper" else "corpus", "hits": hits}]}
                directory = root / "processed" / dataset
                jsonl(directory / ("train/documents.jsonl" if dataset == "qasper" else "documents.jsonl"), docs)
                jsonl(directory / "train/gold.jsonl", [gold])
                write_new(root / f"{dataset}_retrieval.json", report)
                with patch.multiple(smoke, ROOT=root, PROCESSED=root / "processed", REPORTS=root):
                    with contextlib.redirect_stdout(io.StringIO()):
                        smoke.evaluate(dataset)
                expected = json.loads((root / f"{dataset}_metrics.json").read_text())["metrics"]
                actual, reason = score_query(dataset, gold, ["e3"], {d["evidence_id"]: d for d in docs})
                self.assertIsNone(reason)
                self.assertEqual(actual["status"], "complete_alternative")
                for metric, value in expected.items():
                    self.assertEqual(actual[metric], value)
                if dataset == "scifact":
                    with patch.multiple(rationale_smoke, ROOT=root, DATA=directory, REPORTS=root), \
                         patch("sys.argv", ["fixture", "--report-dir", str(root)]):
                        with contextlib.redirect_stdout(io.StringIO()):
                            rationale_smoke.main()
                    rationale = json.loads((root / "scifact_rationale_metrics.json").read_text())["metrics"]
                    for metric, value in rationale.items():
                        self.assertEqual(actual[metric], value)
                    self.assertAlmostEqual(actual["recall@10"], 1 / 3)
                    self.assertEqual(actual["complete_rationale_hit@10"], 1)

    def test_validation_even_for_skipped_queries_and_reporting_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs, _, gold = fixture("qasper")
            queries = [{"query_id": f"q{i}", "paper_id": "p1", "text": "token1"} for i in range(5)]
            # Use separate papers to ensure all three partitions are represented.
            documents = [{**docs[0], "paper_id": f"p{i}", "evidence_id": f"e{i}"} for i in range(5)]
            for i, q in enumerate(queries):
                q["paper_id"] = f"p{i}"
            base = root / "processed/qasper/dev"
            jsonl(base / "queries.jsonl", queries)
            jsonl(base / "documents.jsonl", documents)
            golds = [{"query_id": q["query_id"], "annotations": []} for q in queries]
            jsonl(base / "gold.jsonl", golds)
            _, _, hashes = load_inputs(root / "processed", "qasper", "dev")
            manifest = make_manifest(queries, "qasper", hashes["queries_sha256"])
            mp = root / "manifest.json"
            write_new(mp, manifest)
            results = [{"query_id": q["query_id"], "query": q["text"], "scope": q["paper_id"],
                        "candidate_count": 1, "search_seconds": .01,
                        "hits": [{"rank": 1, "score": 1., **d}]} for q, d in zip(queries, documents)]
            report = {"dataset": "qasper", "split": "dev", "source_hashes": hashes,
                      "config": {"top_k": 10, "retriever": "bm25"},
                      "manifest_sha256": file_hash(mp), "results": results}
            qindex, dindex = {q["query_id"]: q for q in queries}, {d["evidence_id"]: d for d in documents}
            for mutation in (lambda r: r["results"][0]["hits"][0].update(rank=2),
                             lambda r: r["results"][0]["hits"][0].update(text="stale"),
                             lambda r: r["results"][0].update(scope="p9"),
                             lambda r: r["results"].pop(),
                             lambda r: r["results"][0]["hits"][0].update(evidence_id="unknown")):
                bad = copy.deepcopy(report)
                mutation(bad)
                with self.assertRaises(ValueError):
                    validate_report(bad, "qasper", "dev", qindex, dindex, hashes)
            write_new(root / "qasper_retrieval.json", report)
            a = argparse.Namespace(processed_dir=root / "processed", dataset="qasper", split="dev",
                                   report_dir=root, query_manifest=mp, include_reporting=False)
            evaluate(a)
            summary = json.loads((root / "qasper_metrics_development.json").read_text())
            self.assertNotIn("reporting", summary["partitions"])
            diagnosis = json.loads((root / "qasper_diagnosis.json").read_text())
            expected = {r["query_id"] for r in manifest["assignments"] if r["partition"] == "diagnosis"}
            self.assertEqual({r["query_id"] for r in diagnosis}, expected)
            for part in summary["partitions"].values():
                self.assertEqual(part["scored_queries"], 0)
                self.assertEqual(part["selected_queries"], part["skipped_count"])

    def test_cache_checks_without_model_download(self):
        import numpy as np

        class FakeModel:
            max_seq_length = 256
            calls = 0

            def tokenizer(self, texts, **kwargs):
                return {"length": [300 for _ in texts]}

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, **kwargs):
                self.calls += 1
                return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            docs, _, _ = fixture("scifact")
            model = FakeModel()
            _, info = cached_embeddings(model, docs, {"revision": "fixture", "batch_size": 16}, tmp)
            self.assertFalse(info["cache_hit"])
            self.assertEqual(info["truncation"]["over_limit"], len(docs))
            _, info = cached_embeddings(model, docs, {"revision": "fixture", "batch_size": 16}, tmp)
            self.assertTrue(info["cache_hit"])
            self.assertEqual(model.calls, 1)
            path = next(Path(tmp).glob("*.json"))
            metadata = json.loads(path.read_text())
            metadata["metadata"]["evidence_ids"].reverse()
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                cached_embeddings(model, docs, {"revision": "fixture", "batch_size": 16}, tmp)

    def test_dense_runner_and_comparison_offline(self):
        import numpy as np

        class FakeModel:
            max_seq_length = 256

            def __init__(self, *args, **kwargs):
                pass

            def tokenizer(self, texts, **kwargs):
                return {"length": [2 for _ in texts]}

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, **kwargs):
                return np.asarray([[1., 0.] for _ in texts], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs, _, _ = fixture("qasper")
            queries = [{"query_id": f"q{i}", "paper_id": f"p{i}", "text": "token1"} for i in range(5)]
            documents = [{**docs[0], "evidence_id": f"e{i}", "paper_id": f"p{i}"} for i in range(5)]
            gold = [{"query_id": f"q{i}", "annotations": [{"answer": {"unanswerable": False},
                     "figure_table_evidence": [], "unmatched_text_evidence": [],
                     "evidence_ids": [f"e{i}"]}]} for i in range(5)]
            base = root / "processed/qasper/dev"
            for name, rows in (("queries", queries), ("documents", documents), ("gold", gold)):
                jsonl(base / f"{name}.jsonl", rows)
            _, _, hashes = load_inputs(root / "processed", "qasper", "dev")
            mp = root / "manifest.json"
            write_new(mp, make_manifest(queries, "qasper", hashes["queries_sha256"]))
            a = argparse.Namespace(dataset="qasper", split="dev", processed_dir=root / "processed",
                query_manifest=mp, device="cpu", top_k=10, batch_size=16, cache_dir=root / "cache",
                include_reporting=False)
            fake_torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=None), get_num_threads=lambda: 1)
            with patch.dict("sys.modules", {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=FakeModel),
                                            "torch": fake_torch}):
                for method in ("bm25", "dense"):
                    a.retriever, a.report_dir = method, root / method
                    run(a)
                    evaluate(a)
            args = ["fixture", "--dataset", "qasper", "--bm25-dir", str(root / "bm25"),
                    "--dense-dir", str(root / "dense"), "--output-dir", str(root / "comparison"),
                    "--sample-size", "100"]
            with patch("sys.argv", args):
                comparison.main()
            table = (root / "comparison/qasper_comparison_development.csv").read_text()
            self.assertIn("complete_evidence_hit@10", table)
            self.assertNotIn("reporting", table)
            review = (root / "comparison/qasper_diagnosis_review.md").read_text()
            manifest = json.loads(mp.read_text())
            for row in manifest["assignments"]:
                if row["partition"] != "diagnosis":
                    self.assertNotIn('"query_id": "' + row["query_id"] + '"', review)
            # Final evaluation is explicit and does not emit reporting content.
            a.include_reporting = True
            evaluate(a)
            final = json.loads((a.report_dir / "qasper_metrics_final.json").read_text())
            self.assertIn("reporting", final["partitions"])


if __name__ == "__main__":
    unittest.main()
