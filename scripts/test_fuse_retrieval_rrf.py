"""Offline RRF fixtures; no benchmark, models or gold files."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_common import file_hash, write_new
from scripts.evaluate_retrieval_benchmark import validate_report
from scripts.fuse_retrieval_rrf import fuse_hits, run, validate_sources


def hit(eid, rank):
    return dict(evidence_id=eid, rank=rank, score=99., text=eid, paper_id="p1")


def source(method):
    return dict(schema_version=1, dataset="scifact", split="dev",
                source_hashes=dict(queries_sha256="queries", documents_sha256="documents"),
                manifest_sha256="manifest", config=dict(retriever=method, top_k=10),
                results=[dict(query_id="q1", query="question", scope="corpus",
                              candidate_count=3, search_seconds=1.,
                              hits=[hit("a", 1), hit("b", 2)])])


class RRFTests(unittest.TestCase):
    def test_formula_union_and_final_top_k(self):
        hits = fuse_hits([hit("a", 1), hit("b", 2)],
                         [hit("b", 1), hit("c", 2)], 60, 10)
        self.assertEqual([h["evidence_id"] for h in hits], ["b", "a", "c"])
        self.assertAlmostEqual(hits[0]["score"], 1 / 62 + 1 / 61)
        self.assertEqual(hits[1]["score"], 1 / 61)
        self.assertEqual(hits[2]["score"], 1 / 62)
        self.assertEqual(fuse_hits([hit("a", 1)], [hit("a", 1)], 0, 1)[0]["score"], 2.)
        self.assertEqual(len(fuse_hits([hit("a", 1)], [hit("b", 1)], 60, 1)), 1)

    def test_ties(self):
        hits = fuse_hits([hit("z", 1)], [hit("a", 1)], 60, 10)
        self.assertEqual([h["evidence_id"] for h in hits], ["a", "z"])
        self.assertEqual([h["rank"] for h in hits], [1, 2])

    def test_conflicts_duplicates_and_mismatches(self):
        left, right = source("bm25"), source("dense")
        validate_sources(left, right, "scifact")
        mutations = [lambda r, k=k: r.update({k: "changed"}) for k in
                     ("dataset", "split", "source_hashes", "manifest_sha256")]
        mutations += [lambda r, k=k: r["results"][0].update({k: "changed"}) for k in
                      ("query_id", "query", "scope", "candidate_count")]
        mutations += [lambda r: r["results"][0]["hits"][0].update(text="changed"),
                      lambda r: r["results"][0]["hits"][0].update(paper_id="changed"),
                      lambda r: r["results"][0]["hits"][1].update(evidence_id="a"),
                      lambda r: r["results"][0]["hits"][0].update(rank=0)]
        for mutate in mutations:
            bad = copy.deepcopy(right)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_sources(left, bad, "scifact")
        # Evidence consistency also applies across queries, not just paired hits.
        bad = copy.deepcopy(right)
        for report in (left, bad):
            report["results"].append({**copy.deepcopy(report["results"][0]), "query_id": "q2"})
        bad["results"][1]["hits"][0]["text"] = "conflict across queries"
        with self.assertRaises(ValueError):
            validate_sources(left, bad, "scifact")

    def test_file_roundtrip_evaluator_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in ("qasper", "scifact"):
                for method in ("bm25", "dense"):
                    report = source(method)
                    report["dataset"] = dataset
                    report["results"][0]["scope"] = "p1" if dataset == "qasper" else "corpus"
                    write_new(root / method / f"{dataset}_retrieval.json", report)
                args = argparse.Namespace(dataset=dataset, bm25_dir=root / "bm25",
                    dense_dir=root / "dense", report_dir=root / "out", rrf_k=60, top_k=10)
                # Only the two prediction files exist; there is no gold/manifest/data.
                run(args)
                path = root / "out" / f"{dataset}_retrieval.json"
                output = json.loads(path.read_text())
                docs = {eid: {k: v for k, v in hit(eid, 1).items()
                              if k not in ("rank", "score")} for eid in ("a", "b", "c")}
                validate_report(output, dataset, "dev", {"q1": dict(text="question", paper_id="p1")},
                                docs, report["source_hashes"])
                self.assertEqual(output["config"]["retriever"], "hybrid_rrf")
                self.assertIn("excludes source retrieval", output["timing_protocol"])
                for method in ("bm25", "dense"):
                    metadata = output["fusion_sources"][method]
                    self.assertEqual(metadata["sha256"], file_hash(root / method / f"{dataset}_retrieval.json"))
                    self.assertEqual(metadata["returned_depth_by_query"], {"q1": 2})
                original = path.read_bytes()
                with self.assertRaises(ValueError):
                    run(args)
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
