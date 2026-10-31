import unittest
from unittest.mock import patch

from scripts.prepare_qasper_review import adjacent, duplicate_pairs, observe, select_groups


def doc(eid, paragraph, text="text", section=0, paper="p"):
    return dict(evidence_id=eid, paper_id=paper, section_index=section,
                paragraph_index=paragraph, text=text)


class QasperReviewTests(unittest.TestCase):
    def test_sampling_definition_overlap_and_reproducibility(self):
        rows = [dict(query_id=str(i), bucket="selection_loss" if i < 4 else "success",
                     metrics=dict(reranker={"complete_evidence_hit@10": int(i >= 4)},
                                  rrf_pool50={"complete_evidence_hit@10": 1})) for i in range(8)]
        with patch("scripts.prepare_qasper_review.GROUP_SIZES", dict(selection_loss=3, regression_complete=3, success=2)):
            selected, population = select_groups(rows)
            self.assertEqual((selected, population), select_groups(list(reversed(rows))))
            self.assertEqual(population, dict(selection_loss=4, regression_complete=4, success=4))
            self.assertEqual(len(selected["selection_loss"]), 3)
            self.assertTrue(set(selected["selection_loss"]) & set(selected["regression_complete"]))
            self.assertFalse(set(selected["success"]) & set(selected["regression_complete"]))

    def test_adjacency_requires_paper_section_and_paragraph_metadata(self):
        a = doc("a", 1)
        self.assertTrue(adjacent(a, doc("b", 2)))
        for b in (doc("b", None), doc("b", 2, section=None), doc("b", 2, section=1),
                  doc("b", 2, paper="other"), doc("b", 3)):
            self.assertFalse(adjacent(a, b))

    def test_optional_alternative_is_not_failure_and_outside_pool_not_added(self):
        a, b, c = doc("a", 0, "answer"), doc("b", 1, "context"), doc("c", 2, "optional")
        annotations = [dict(annotation_id="one", evidence_ids=["a"]), dict(annotation_id="two", evidence_ids=["c"])]
        result = observe(annotations, {"a": a, "b": b}, [a, b], {"c": c})
        self.assertEqual(result["alternatives"][0]["missing_ids"], [])
        self.assertEqual(result["missing_gold_adjacency"], [])
        self.assertFalse(result["incomplete_annotation_one_hop_available"])

    def test_multi_evidence_one_hop_requires_every_missing_unit(self):
        a, b, c = doc("a", 0, "a"), doc("b", 1, "b"), doc("c", 5, "c")
        refs = [dict(annotation_id="one", evidence_ids=["a", "c"])]
        result = observe(refs, {d["evidence_id"]: d for d in (a, b, c)}, [b])
        self.assertEqual(len(result["missing_gold_adjacency"]), 1)
        self.assertFalse(result["incomplete_annotation_one_hop_available"])
        result = observe([dict(annotation_id="one", evidence_ids=["a", "b"])], {"a": a, "b": b}, [b])
        self.assertTrue(result["incomplete_annotation_one_hop_available"])

    def test_duplicate_equivalence_and_near_overlap_are_separate(self):
        exact, overlap = duplicate_pairs([doc("a", 0, "same text"), doc("b", 1, "same  text")])
        self.assertEqual(exact, [["a", "b"]])
        self.assertEqual(overlap, [])  # lexical overlap is restricted to >=20 unique word tokens
        text = " ".join(f"word{i}" for i in range(30))
        exact, overlap = duplicate_pairs([doc("a", 0, text), doc("b", 1, text + " extra")])
        self.assertEqual(exact, [])
        self.assertGreaterEqual(overlap[0]["token_jaccard"], .8)


if __name__ == "__main__":
    unittest.main()
