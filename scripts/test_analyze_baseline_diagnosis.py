"""Offline fixtures: duplicate/annotation semantics, leakage guards and paired CI."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.analyze_baseline_diagnosis import (
    Inputs, classify, digest, paired_bootstrap, validate_summary,
)


class DiagnosisAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.docs = {eid: dict(evidence_id=eid, paper_id="p", text=text)
                     for eid, text in (("a", "same text"), ("dup", "same  text"),
                                       ("b", "second"), ("c", "alternative"))}
        self.reference = dict(annotations=[self.annotation(["a", "b"]), self.annotation(["c"])])

    @staticmethod
    def annotation(ids, **kwargs):
        return dict(evidence_ids=ids, answer=dict(unanswerable=False),
                    figure_table_evidence=[], unmatched_text_evidence=[], **kwargs)

    def test_pool_vs_selection_and_alternative_duplicate_semantics(self):
        self.assertEqual(classify("qasper", self.reference, ["dup", "b"], ["dup", "b"], self.docs), ("success", True))
        self.assertEqual(classify("qasper", self.reference, ["a", "b"], ["a"], self.docs), ("selection_loss", True))
        self.assertEqual(classify("qasper", self.reference, ["a"], ["a"], self.docs), ("pool_missing", False))
        self.assertEqual(classify("qasper", self.reference, ["c"], ["c"], self.docs), ("success", True))
        # Even equivalent duplicate text consumes rank; b at rank 11 is not top10 success.
        docs = {**self.docs, **{f"d{i}": dict(paper_id="p", text="same text") for i in range(10)}}
        ranked = [f"d{i}" for i in range(10)] + ["b"]
        self.assertEqual(classify("qasper", self.reference, ranked, ranked, docs), ("selection_loss", True))

    def test_skips_and_scifact_any_complete_rationale(self):
        ref = copy.deepcopy(self.reference)
        for a in ref["annotations"]:
            a["answer"]["unanswerable"] = True
        self.assertEqual(classify("qasper", ref, [], [], self.docs), ("skipped", None))
        ref = dict(evidence_sets=[dict(evidence_ids=["a", "b"], paper_id="p"),
                                  dict(evidence_ids=["c"], paper_id="p")])
        self.assertEqual(classify("scifact", ref, ["c"], ["c"], self.docs), ("success", True))
        self.assertEqual(classify("scifact", dict(evidence_sets=[]), [], [], self.docs), ("skipped", None))

    def test_cluster_bootstrap_macro_queries_not_paper_means(self):
        rows = []
        for group, delta in (("p1", 1), ("p1", 1), ("p1", 1), ("p2", -1)):
            rows.append(dict(group_id=group, metrics=dict(reranker=dict(m=float(delta > 0)), bm25=dict(m=float(delta < 0)))))
        rows.append(dict(group_id="only_skipped", metrics=dict(reranker=None, bm25=None)))
        a = paired_bootstrap(rows, "m", "bm25", 300, 42)
        self.assertEqual(a, paired_bootstrap(rows, "m", "bm25", 300, 42))
        self.assertEqual(a["delta"], .5)
        self.assertEqual(a["resampling_groups"], 3)
        self.assertGreater(a["empty_draws"], 0)
        self.assertEqual((a["wins"], a["ties"], a["losses"]), (3, 0, 1))
        self.assertLessEqual(a["ci_low"], .5)
        self.assertGreaterEqual(a["ci_high"], .5)

    def test_denominator_mismatch_stops(self):
        assignments = {"q": dict(partition="diagnosis", group_id="p")}
        summary = dict(selected_queries=1, groups=1, scored_queries=1, skipped_count=0,
                       skipped_reasons={}, statuses=dict(complete=1), metrics=dict(m=1.))
        validate_summary(summary, assignments, "diagnosis")
        with self.assertRaisesRegex(ValueError, "denominator"):
            validate_summary({**summary, "scored_queries": 0}, assignments, "diagnosis")

    def test_archive_hash_and_gold_reporting_parse_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, side = root / "a.zip", root / "a.sha256.json"
            raw = b'not even JSON: reporting gold stays opaque'
            manifest = dict(files={"data/processed/qasper/dev/gold.jsonl": digest(raw)})
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("backup_manifest.json", json.dumps(manifest))
                z.writestr("data/processed/qasper/dev/gold.jsonl", raw)
            side.write_text(json.dumps(dict(file=archive.name, sha256=digest(archive.read_bytes()))))
            # Only manifest and sidecar are decoded: hashing gold must not json.loads it.
            original = json.loads
            def guarded(value, *args, **kwargs):
                self.assertNotIn(b"reporting gold", value if isinstance(value, bytes) else value.encode())
                return original(value, *args, **kwargs)
            with patch("scripts.analyze_baseline_diagnosis.json.loads", side_effect=guarded):
                inputs = Inputs(root, archive, side)
                try:
                    with self.assertRaisesRegex(ValueError, "Full gold"):
                        inputs.arch_json("data/processed/qasper/dev/gold.jsonl")
                    with self.assertRaisesRegex(ValueError, "Reporting"):
                        inputs.local_json(root / "qasper_metrics_final.json")
                finally:
                    inputs.zip.close()
            side.write_text(json.dumps(dict(file=archive.name, sha256="0" * 64)))
            with self.assertRaisesRegex(ValueError, "Archive SHA256"):
                Inputs(root, archive, side)


if __name__ == "__main__":
    unittest.main()
