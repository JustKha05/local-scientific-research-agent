"""Prepare reproducible diagnosis-only QASPER review packets from immutable artifacts.

No models, retrieval, fusion, bootstrap, full-gold parsing or reporting access.
Automatic observations are structural; human verification is always left pending.
"""
import argparse
import csv
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.analyze_baseline_diagnosis import Inputs, digest, hash_file, evidence_key
from scripts.benchmark_common import index, require, validate_manifest, write_new
from scripts.evaluate_bm25_smoke import normalize
from scripts.evaluate_retrieval_benchmark import score_query
from scripts.fuse_retrieval_rrf import validate_sources

COMPLETE = "complete_evidence_hit@10"
GROUP_SIZES = {"selection_loss": 20, "regression_complete": 10, "success": 10}


def select_groups(rows, seed=42):
    """Sample independently from sorted frames, resetting RNG to seed per group.

    Regression is explicitly complete-hit loss (reranker=0, RRF pool50=1), not
    a post-hoc union of metrics. Group overlap is retained and disclosed.
    """
    require(len({r["query_id"] for r in rows}) == len(rows), "Duplicate query rows")
    populations = {
        "selection_loss": [r for r in rows if r["bucket"] == "selection_loss"],
        "regression_complete": [r for r in rows if r["metrics"]["reranker"] is not None
                                and r["metrics"]["reranker"][COMPLETE] == 0
                                and r["metrics"]["rrf_pool50"][COMPLETE] == 1],
        "success": [r for r in rows if r["bucket"] == "success"],
    }
    selected = {}
    for group, count in GROUP_SIZES.items():
        population = sorted(populations[group], key=lambda r: r["query_id"])
        require(len(population) >= count, f"Insufficient {group} population: {len(population)} < {count}")
        selected[group] = sorted(r["query_id"] for r in random.Random(seed).sample(population, count))
    return selected, {k: len(v) for k, v in populations.items()}


def adjacent(a, b):
    """Same paper/section, numeric paragraph positions one apart; no ID guessing."""
    return (a["paper_id"] == b["paper_id"] and type(a.get("section_index")) is int
            and a["section_index"] == b.get("section_index")
            and type(a.get("paragraph_index")) is int and type(b.get("paragraph_index")) is int
            and abs(a["paragraph_index"] - b["paragraph_index"]) == 1)


def eligible_annotations(gold):
    return [a for a in gold["annotations"] if not a["answer"].get("unanswerable")
            and not a["figure_table_evidence"] and not a["unmatched_text_evidence"] and a["evidence_ids"]]


def duplicate_pairs(hits):
    exact, high_overlap = [], []
    for a, b in itertools.combinations(hits, 2):
        if evidence_key("qasper", a["evidence_id"], {a["evidence_id"]: a}) == evidence_key(
                "qasper", b["evidence_id"], {b["evidence_id"]: b}):
            exact.append([a["evidence_id"], b["evidence_id"]])
        left, right = (set(re.findall(r"\w+", normalize(h["text"]).casefold())) for h in (a, b))
        if min(len(left), len(right)) >= 20:
            similarity = len(left & right) / len(left | right)
            if similarity >= .8:
                high_overlap.append(dict(left=a["evidence_id"], right=b["evidence_id"], token_jaccard=similarity))
    return exact, high_overlap


def observe(annotations, pool, top10, gold_documents=None):
    """Gold-dependent containment/adjacency audit; NOT a deployable selector."""
    docs = {**(gold_documents or {}), **pool, **{h["evidence_id"]: h for h in top10}}
    top_keys = {evidence_key("qasper", h["evidence_id"], docs) for h in top10}
    alternatives, links = [], []
    for annotation in annotations:
        ids = annotation["evidence_ids"]
        require(all(e in docs for e in ids), "Gold document must be supplied for observation")
        missing = [e for e in ids if evidence_key("qasper", e, docs) not in top_keys]
        local_links = []
        for eid in missing:
            # Match normalized equivalent IDs in pool, not only the annotation ID.
            equivalents = [e for e in pool if evidence_key("qasper", e, docs) == evidence_key("qasper", eid, docs)]
            for candidate_id in equivalents:
                local_links += [dict(missing_gold=eid, candidate_id=candidate_id, anchor=h["evidence_id"])
                                for h in top10 if adjacent(pool[candidate_id], h)]
        alternatives.append(dict(annotation_id=annotation["annotation_id"], evidence_ids=ids,
                                 missing_ids=missing, one_hop_available=bool(missing) and all(
                                     any(link["missing_gold"] == e for link in local_links) for e in missing)))
        links.extend(local_links)
    exact, high_overlap = duplicate_pairs(top10)
    return dict(alternatives=alternatives, missing_gold_adjacency=links,
                incomplete_annotation_one_hop_available=any(a["one_hop_available"] for a in alternatives),
                exact_duplicate_pairs=exact, high_overlap_pairs=high_overlap)


def render_case(case):
    def anchor(eid):
        return "e-" + eid.replace(":", "-")

    def ref(eid):
        return f"[{eid}](#{anchor(eid)})"

    gold_ids = {e for a in case["eligible_annotations"] for e in a["evidence_ids"]}
    gold_keys = {evidence_key("qasper", e, case["documents"]) for e in gold_ids}
    lines = [f"# {case['query_id']}", "", f"**Question:** {case['question']}", "",
             f"Paper: `{case['paper_id']}` — {case['title']}", "",
             f"Groups: {', '.join(case['groups'])}. Human verification: **PENDING**. "
             "Everything labelled automatic below is a deterministic structural observation, not a semantic cause.", "",
             "## Eligible gold evidence alternatives", "",
             "QASPER completion needs any ONE eligible alternative. Optional gold from another alternative is not a failure. "
             "Normalized duplicate text in the same paper is equivalent; each hit still consumes a rank.", ""]
    for a in case["eligible_annotations"]:
        lines += [f"### Annotation `{a['annotation_id']}`", "",
                  "Evidence: " + ", ".join(ref(e) for e in a["evidence_ids"]), ""]
        for e in a["evidence_ids"]:
            lines += [f"**`{e}`**", "", case["documents"][e]["text"], ""]
    lines += ["## Automatic observations (not human labels)", "", "```json",
              json.dumps(case["observations"], ensure_ascii=False, indent=2), "```", "",
              "An adjacent missed paragraph indicates a potential context signal, not that all neighboring text is relevant. "
              "One-hop availability uses diagnosis gold, allows additions without replacements and ignores a token budget; "
              "it is not a fixed-budget performance estimate or experiment.", ""]
    for method in ("reranker", "rrf_pool50"):
        lines += [f"## {method} top10", "", "Metrics: `" + json.dumps(case["metrics"][method], sort_keys=True) + "`", "",
                  "| Rank | Evidence (full text below) | Saved score | Eligible-gold equivalent | Text preview |",
                  "|---:|---|---:|---|---|"]
        for hit in case["top10"][method]:
            text = hit["text"].replace("\n", " ").replace("|", "\\|")[:180]
            gold = evidence_key("qasper", hit["evidence_id"], case["documents"]) in gold_keys
            lines.append(f"| {hit['rank']} | {ref(hit['evidence_id'])} | {hit['score']:.8g} | {gold} | {text}… |")
        lines.append("")
    lines += ["## Full candidate pool and related evidence", "",
              f"All {len(case['pool'])} original union candidates are shown, not only heuristic matches. "
              "Gold/neighbor flags are diagnosis-only audit hints. Section titles and abstracts are not treated as adjacent paragraphs. "
              "BM25/dense scores are original source scores, not cross-encoder logits. "
              "Unstored cross-encoder scores outside top10 are **unknown**, never filled with zero.", ""]
    ranks = {m: {h["evidence_id"]: h for h in hits} for m, hits in case["top10"].items()}
    for eid, doc in case["pool"].items():
        flags = []
        if evidence_key("qasper", eid, case["documents"]) in gold_keys:
            flags.append("eligible_gold_equivalent")
        if any(adjacent(doc, case["documents"][e]) for e in gold_ids):
            flags.append("neighbor_of_gold")
        if eid in ranks["reranker"]:
            flags.append("reranker_top10")
        if eid in ranks["rrf_pool50"]:
            flags.append("rrf_top10")
        lines += [f"<a id=\"{anchor(eid)}\"></a>", "", f"### `{eid}`", "",
                  f"Section: {doc['section_name']} | unit: {doc['unit_type']} | "
                  f"section_index={doc['section_index']}, paragraph_index={doc['paragraph_index']}", "",
                  f"Flags: {', '.join(flags) or 'none'}", "",
                  "Source rank/score: `" + json.dumps(case["source_positions"][eid], sort_keys=True) + "`", "",
                  "Saved reranker score: " + (str(ranks["reranker"][eid]["score"]) if eid in ranks["reranker"] else "UNKNOWN (outside stored top10)"),
                  "", doc["text"], ""]
    for eid in sorted(gold_ids - set(case["pool"])):
        lines += [f"<a id=\"{anchor(eid)}\"></a>", "", f"### Gold outside pool: `{eid}`", "",
                  case["documents"][eid]["text"], ""]
    lines += ["## Human verification (not completed by packet generation)", "",
              "- reviewer / date:", "- confirmed labels (or uncertain):",
              "- evidence IDs and exact supporting text:", "- counterevidence / alternative explanation:",
              "- support for duplicate/context hypothesis:", ""]
    return "\n".join(lines)


def prepare(root, analysis_dir, archive_path, sidecar, output, seed=42):
    root, analysis_dir, output = Path(root), Path(analysis_dir), Path(output)
    require(not output.exists(), f"Refusing to overwrite review directory: {output}")
    baseline_hashes = json.loads((analysis_dir / "output_hashes.json").read_text(encoding="utf8"))
    for name, expected in baseline_hashes.items():
        require(hash_file(analysis_dir / name) == expected, f"Baseline output changed: {name}")
    provenance = json.loads((analysis_dir / "provenance.json").read_text(encoding="utf8"))
    with (analysis_dir / "diagnosis_queries.csv").open(encoding="utf8", newline="") as f:
        rows = [dict(query_id=r["query_id"], bucket=r["bucket"], question=r["query"],
                     metrics=json.loads(r["diagnosis_metrics"])) for r in csv.DictReader(f) if r["dataset"] == "qasper"]
    selected, populations = select_groups(rows, seed)
    memberships = defaultdict(list)
    for group, ids in selected.items():
        for qid in ids:
            memberships[qid].append(group)
    inputs = Inputs(root, archive_path, sidecar)
    try:
        require(inputs.manifest == provenance["archive_manifest"], "Archive differs from audited baseline")
        config = inputs.arch_json("reranker_run_config.json")
        manifest_raw = inputs.arch_raw("configs/dev_qasper_seed42.json")
        manifest = json.loads(manifest_raw)
        require(hash_file(root / "configs/dev_qasper_seed42.json") == digest(manifest_raw), "Local manifest mismatch")
        queries_raw = inputs.arch_raw("data/processed/qasper/dev/queries.jsonl")
        queries = index([json.loads(line) for line in queries_raw.splitlines() if line.strip()], "query_id")
        assignments = validate_manifest(manifest, list(queries.values()), "qasper", digest(queries_raw))
        allowed = {q for q, a in assignments.items() if a["partition"] == "diagnosis"}
        require({r["query_id"] for r in rows} == allowed, "Baseline CSV diagnosis coverage mismatch")
        require(set(memberships) <= allowed, "Non-diagnosis review query")
        documents = index([json.loads(line) for line in inputs.arch_raw("data/processed/qasper/dev/documents.jsonl").splitlines()], "evidence_id")
        diag = index(inputs.arch_json(config["report_dir"] + "/qasper_diagnosis.json"), "query_id")
        require(set(diag) == allowed, "Archived diagnosis coverage mismatch")
        rr_dir = root / "evaluation/reports/dev_rrf_pool50_top10_v1"
        for suffix in ("retrieval", "diagnosis", "metrics_development"):
            path = rr_dir / f"qasper_{suffix}.json"
            expected = provenance["inputs"].get(str(path))
            require(expected and hash_file(path) == expected, f"RRF input hash mismatch: {path}")
        rr_diag = index(inputs.local_json(rr_dir / "qasper_diagnosis.json"), "query_id")
        rr_report = inputs.local_json(rr_dir / "qasper_retrieval.json")
        rr_metrics = inputs.local_json(rr_dir / "qasper_metrics_development.json")
        require(rr_metrics["retrieval_sha256"] == hash_file(rr_dir / "qasper_retrieval.json"), "RRF retrieval binding mismatch")
        require(rr_metrics["reporting_unlocked"] is False and set(rr_metrics["partitions"]) == {"diagnosis", "calibration"}, "Reporting metrics not allowed")
        require(set(rr_diag) == allowed, "RRF diagnosis coverage mismatch")
        sources = [inputs.arch_json(config[k] + "/qasper_retrieval.json") for k in ("bm25_dir", "dense_dir")]
        source_rows = validate_sources(*sources, "qasper")
        for method, key in (("bm25", "bm25_dir"), ("dense", "dense_dir")):
            require(rr_report["fusion_sources"][method]["sha256"] == digest(inputs.arch_raw(config[key] + "/qasper_retrieval.json")), "RRF pool binding mismatch")
        cases = []
        by_id = {r["query_id"]: r for r in rows}
        for qid in sorted(memberships):
            d, r = diag[qid], rr_diag[qid]
            require(d["query"] == r["query"] == by_id[qid]["question"] == queries[qid]["text"], "Question mismatch")
            require(d["gold"] == r["gold"], "Diagnosis gold mismatch")
            annotations = eligible_annotations(d["gold"])
            require(annotations, "Unscored query in review sample")
            pool, positions = {}, defaultdict(dict)
            for method, source in zip(("bm25", "dense"), source_rows):
                row = source[qid]
                require(len(row["hits"]) >= min(50, row["candidate_count"]), "Incomplete top50 source")
                for hit in row["hits"][:50]:
                    pool[hit["evidence_id"]] = documents[hit["evidence_id"]]
                    positions[hit["evidence_id"]][method] = dict(rank=hit["rank"], score=hit["score"])
            for method, diagnosis in (("reranker", d), ("rrf_pool50", r)):
                hits = diagnosis["hits"]
                require(len(hits) == min(10, len(pool)) and {h["evidence_id"] for h in hits} <= set(pool), "Top10/pool mismatch")
                require(all(all(h.get(k) == v for k, v in documents[h["evidence_id"]].items()) for h in hits), "Hit payload mismatch")
                metric, reason = score_query("qasper", d["gold"], [h["evidence_id"] for h in hits], documents)
                require(reason is None and metric == diagnosis["metrics"] == by_id[qid]["metrics"][method], "Diagnosis score mismatch")
            # observe() needs gold docs outside pool too; observations can match only original pool members.
            relevant_docs = {e: documents[e] for a in annotations for e in a["evidence_ids"]}
            # An optional gold alternative outside the pool must never become a candidate.
            observations = observe(annotations, pool, d["hits"], relevant_docs)
            for item in observations["missing_gold_adjacency"]:
                require(item["candidate_id"] in pool, "Context hint outside original candidate pool")
            case = dict(query_id=qid, paper_id=d["paper_id"], title=documents[next(iter(pool))]["title"],
                        question=d["query"], groups=memberships[qid], bucket=by_id[qid]["bucket"],
                        eligible_annotations=annotations, metrics={"reranker": d["metrics"], "rrf_pool50": r["metrics"]},
                        top10={"reranker": d["hits"], "rrf_pool50": r["hits"]},
                        pool=dict(sorted(pool.items())), documents={**pool, **relevant_docs},
                        source_positions=dict(positions), observations=observations)
            cases.append(case)
        output.mkdir(parents=True, exist_ok=False)
        (output / "cases").mkdir()
        for case in cases:
            path = output / "cases" / (case["query_id"].split(":", 1)[1] + ".md")
            with path.open("x", encoding="utf8") as stream:
                stream.write(render_case(case))
        observations_csv = []
        for c in cases:
            o = c["observations"]
            observations_csv.append(dict(query_id=c["query_id"], paper_id=c["paper_id"], groups=";".join(c["groups"]),
                bucket=c["bucket"], human_verified=False, question=c["question"], pool_size=len(c["pool"]),
                reranker_recall=c["metrics"]["reranker"]["recall@10"], rrf_recall=c["metrics"]["rrf_pool50"]["recall@10"],
                exact_duplicate_pairs=len(o["exact_duplicate_pairs"]), high_overlap_pairs=len(o["high_overlap_pairs"]),
                missing_gold_adjacency=json.dumps(o["missing_gold_adjacency"], sort_keys=True),
                incomplete_annotation_one_hop_available=o["incomplete_annotation_one_hop_available"],
                alternatives=json.dumps(o["alternatives"], sort_keys=True)))
        with (output / "observations.csv").open("x", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(observations_csv[0]))
            writer.writeheader()
            writer.writerows(observations_csv)
        group_stats = {}
        for group in (*GROUP_SIZES, "unique"):
            subset = [c for c in cases if group == "unique" or group in c["groups"]]
            group_stats[group] = dict(queries=len(subset),
                exact_duplicate_cases=sum(bool(c["observations"]["exact_duplicate_pairs"]) for c in subset),
                high_overlap_cases=sum(bool(c["observations"]["high_overlap_pairs"]) for c in subset),
                missing_gold_adjacent_cases=sum(bool(c["observations"]["missing_gold_adjacency"]) for c in subset),
                one_hop_available_cases=sum(c["observations"]["incomplete_annotation_one_hop_available"] for c in subset))
        write_new(output / "selection.json", dict(seed=seed, regression_definition="reranker complete@10=0 and RRF pool50 complete@10=1",
            sampling="independent random.sample from query-ID sorted populations; RNG reset to seed for each group",
            populations=populations, selected=selected, unique_queries=len(cases),
            selected_papers=len({c["paper_id"] for c in cases}), group_statistics=group_stats,
            baseline_provenance_sha256=hash_file(analysis_dir / "provenance.json"),
            baseline_outputs_sha256=hash_file(analysis_dir / "output_hashes.json"),
            archive_sha256=hash_file(archive_path), script_sha256=hash_file(__file__), human_verified_queries=0,
            hypotheses="Structural adjacency/overlap facts only; not semantic causes or selector performance",
            missing_scores="Cross-encoder logits outside stored top10; tokenizer-specific costs/context budget not logged"))
        write_new(output / "output_hashes.json", {p.relative_to(output).as_posix(): hash_file(p)
                   for p in sorted(output.rglob("*")) if p.is_file()})
        print(json.dumps(dict(unique_queries=len(cases), groups=group_stats), indent=2))
    finally:
        inputs.zip.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis/baseline_diagnosis_v1"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/qasper_review_v1"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.root, args.analysis_dir, args.archive, args.sha256_file, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
