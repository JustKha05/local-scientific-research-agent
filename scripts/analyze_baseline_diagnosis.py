"""Offline, diagnosis-only baseline audit. Never opens/deserializes full gold JSONL.

The archive and full-gold files are hashed as opaque bytes for integrity only.
Only exported diagnosis gold is scored. Calibration comes from saved aggregates.
No retrieval, model, network, CUDA, manifest creation or reporting switch exists.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from scripts.benchmark_common import index, require, validate_manifest, write_new
from scripts.evaluate_bm25_smoke import normalize
from scripts.evaluate_retrieval_benchmark import score_query, validate_report
from scripts.fuse_retrieval_rrf import validate_sources

MODEL_SHA = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
METHOD_DIRS = {
    "bm25": "dev_bm25_v1", "dense": "dev_dense_gpu_v1",
    "rrf_top10": "dev_rrf_top10_v1", "rrf_pool50": "dev_rrf_pool50_top10_v1",
}
BUCKETS = ("pool_missing", "selection_loss", "success", "skipped")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Inputs:
    def __init__(self, root, archive, sidecar):
        self.root = Path(root)
        self.files = {}
        side = self.local_json(sidecar)
        require(side["file"] == Path(archive).name, "Archive sidecar filename mismatch")
        actual = hash_file(archive)
        require(actual == side["sha256"], f"Archive SHA256 mismatch: {archive}")
        self.files[str(archive)] = actual
        self.zip = zipfile.ZipFile(archive)
        names = self.zip.namelist()
        require(len(names) == len(set(names)), "Duplicate archive members")
        for name in names:
            path = PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
                    f"Unsafe archive member: {name}")
        self.manifest = self.arch_json("backup_manifest.json", listed=False)
        require(set(names) == set(self.manifest["files"]) | {"backup_manifest.json"},
                "Unlisted/missing archive files")
        for name, expected in self.manifest["files"].items():
            # Integrity hashing only: do NOT decode or parse any full gold file.
            with self.zip.open(name) as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            require(h.hexdigest() == expected, f"Archive member SHA256 mismatch: {name}")
        self.files.update({f"{archive}!/{k}": v for k, v in self.manifest["files"].items()})

    @staticmethod
    def allowed(name):
        require(not str(name).endswith("gold.jsonl"), "Full gold parsing is forbidden")
        require("metrics_final" not in str(name), "Reporting metrics are forbidden")

    def local_json(self, path):
        self.allowed(path)
        path = Path(path)
        require(path.is_file(), f"Missing required file: {path}")
        raw = path.read_bytes()
        self.files[str(path)] = digest(raw)
        return json.loads(raw)

    def arch_raw(self, name):
        self.allowed(name)
        require(name in self.manifest["files"], f"Missing required archive member: {name}")
        return self.zip.read(name)

    def arch_json(self, name, listed=True):
        self.allowed(name)
        return json.loads(self.arch_raw(name) if listed else self.zip.read(name))


def evidence_key(dataset, eid, docs):
    return (docs[eid]["paper_id"], normalize(docs[eid]["text"])) if dataset == "qasper" else eid


def targets(dataset, reference, docs):
    """Mirror evaluator eligibility; normalized duplicate paragraphs are equivalent."""
    if dataset == "qasper":
        sets = [a["evidence_ids"] for a in reference["annotations"]
                if not a["answer"].get("unanswerable") and not a["figure_table_evidence"]
                and not a["unmatched_text_evidence"] and a["evidence_ids"]]
    else:
        sets = [s["evidence_ids"] for s in reference["evidence_sets"]]
    return [{evidence_key(dataset, eid, docs) for eid in ids} for ids in sets]


def classify(dataset, reference, pool_ids, ranked_ids, docs):
    gold_sets = targets(dataset, reference, docs)
    if not gold_sets:
        return "skipped", None
    pool = {evidence_key(dataset, eid, docs) for eid in pool_ids}
    available = any(t <= pool for t in gold_sets)
    metric, _ = score_query(dataset, reference, ranked_ids[:10], docs)
    complete = metric["complete_evidence_hit@10" if dataset == "qasper" else "complete_rationale_hit@10"]
    require(not complete or available, "Successful reranker used evidence outside pool")
    return "success" if complete else "selection_loss" if available else "pool_missing", available


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def paired_bootstrap(rows, metric, baseline, repetitions=10000, seed=42):
    """Resample all diagnosis groups, including zero-scored groups; macro by query.

    A sampled paper contributes all its scored queries, including multiplicities.
    This is NOT a macro average of paper means. SciFact group is claim/query ID.
    """
    groups = defaultdict(lambda: [0.0, 0])
    wins = ties = losses = 0
    for row in rows:
        total = groups[row["group_id"]]
        if row["metrics"]["reranker"] is None:
            continue
        delta = row["metrics"]["reranker"][metric] - row["metrics"][baseline][metric]
        total[0] += delta
        total[1] += 1
        wins += delta > 1e-12
        losses += delta < -1e-12
        ties += abs(delta) <= 1e-12
    values = [groups[g] for g in sorted(groups)]
    denominator = sum(n for _, n in values)
    require(denominator > 0, "No scored diagnosis queries")
    estimate = sum(v for v, _ in values) / denominator
    rng, draws, empty = random.Random(seed), [], 0
    for _ in range(repetitions):
        sample = rng.choices(values, k=len(values))
        count = sum(n for _, n in sample)
        if count:
            draws.append(sum(v for v, _ in sample) / count)
        else:
            empty += 1
    require(draws, "All bootstrap draws empty")
    return dict(delta=estimate, ci_low=percentile(draws, .025), ci_high=percentile(draws, .975),
                wins=wins, ties=ties, losses=losses, scored_queries=denominator,
                resampling_groups=len(values), empty_draws=empty,
                bootstrap_draws=repetitions, seed=seed)


def assert_close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), f"Metric mismatch: {label}: {a} != {b}")


def validate_summary(summary, assignments, part, reference=None):
    chosen = [a for a in assignments.values() if a["partition"] == part]
    require(summary["selected_queries"] == len(chosen), f"{part}: selected denominator mismatch")
    require(summary["groups"] == len({a["group_id"] for a in chosen}), f"{part}: group denominator mismatch")
    require(summary["scored_queries"] + summary["skipped_count"] == len(chosen),
            f"{part}: scored/skipped denominator mismatch")
    require(sum(summary["skipped_reasons"].values()) == summary["skipped_count"], "Skip count mismatch")
    require(sum(summary["statuses"].values()) == summary["scored_queries"], "Status count mismatch")
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                for v in summary["metrics"].values()), "Invalid saved aggregate")
    if reference is not None:
        for field in ("selected_queries", "groups", "scored_queries", "skipped_count", "skipped_reasons"):
            require(summary[field] == reference[field], f"{part}: paired denominator mismatch: {field}")


def analyze_dataset(inputs, dataset, seed, repetitions):
    root, archive = inputs.root, inputs
    config = archive.arch_json("reranker_run_config.json")
    require(config["revision"] == config["tokenizer_revision"] == MODEL_SHA, "Pinned model SHA mismatch")
    require(config["reporting_unlocked"] is False, "Reporting unlocked config")
    require(digest(archive.arch_raw("reranker_run_config.json")) == inputs.manifest["config_sha256"],
            "Config SHA256 mismatch")
    source_commit = inputs.manifest["source_commit"]
    source = archive.arch_json(f"resume_provenance/source_{source_commit}.json")
    require(source["commit"] == source_commit and source["config_sha256"] == inputs.manifest["config_sha256"],
            "Source provenance mismatch")
    for path, expected in source["files"].items():
        require(digest(archive.arch_raw(path)) == expected, f"Source snapshot mismatch: {path}")
    # The scorer actually imported locally must match the pinned implementation (allow CRLF only).
    for path in ("scripts/benchmark_common.py", "scripts/evaluate_retrieval_benchmark.py",
                 "scripts/evaluate_bm25_smoke.py", "scripts/fuse_retrieval_rrf.py"):
        local = (root / path).read_bytes()
        require(local.replace(b"\r\n", b"\n") == archive.arch_raw(path).replace(b"\r\n", b"\n"),
                f"Local evaluator semantics differ from archived source: {path}")
        inputs.files[str(root / path)] = digest(local)
    queries_path = f"data/processed/{dataset}/dev/queries.jsonl"
    documents_path = (f"data/processed/{dataset}/dev/documents.jsonl" if dataset == "qasper"
                      else f"data/processed/{dataset}/documents.jsonl")
    queries = index([json.loads(line) for line in archive.arch_raw(queries_path).splitlines() if line.strip()], "query_id")
    docs = index([json.loads(line) for line in archive.arch_raw(documents_path).splitlines() if line.strip()], "evidence_id")
    hashes = dict(queries_sha256=digest(archive.arch_raw(queries_path)),
                  documents_sha256=digest(archive.arch_raw(documents_path)))
    manifest_path = f"configs/dev_{dataset}_seed42.json"
    manifest = archive.arch_json(manifest_path)
    manifest_hash = digest(archive.arch_raw(manifest_path))
    require(hash_file(root / manifest_path) == manifest_hash, "Local/archive query manifest mismatch")
    inputs.files[str(root / manifest_path)] = manifest_hash
    assignments = validate_manifest(manifest, list(queries.values()), dataset, hashes["queries_sha256"])
    diagnosis_ids = {q for q, a in assignments.items() if a["partition"] == "diagnosis"}
    gold_hash = inputs.manifest["files"][f"data/processed/{dataset}/dev/gold.jsonl"]
    sources = [archive.arch_json(f"{config[k]}/{dataset}_retrieval.json") for k in ("bm25_dir", "dense_dir")]
    source_hashes = {method: digest(archive.arch_raw(f"{config[key]}/{dataset}_retrieval.json"))
                     for method, key in (("bm25", "bm25_dir"), ("dense", "dense_dir"))}
    validate_sources(*sources, dataset)
    source_rows = []
    for report in sources:
        require(report["manifest_sha256"] == manifest_hash, "Source manifest mismatch")
        source_rows.append(validate_report(report, dataset, "dev", queries, docs, hashes))
    reports, metrics, diagnoses, rows_by_method, report_hashes = {}, {}, {}, {}, {}
    for method in (*METHOD_DIRS, "reranker"):
        if method == "reranker":
            directory = config["report_dir"]
            load = lambda suffix: archive.arch_json(f"{directory}/{dataset}_{suffix}.json")
            report_hash = digest(archive.arch_raw(f"{directory}/{dataset}_retrieval.json"))
        else:
            directory = root / "evaluation/reports" / METHOD_DIRS[method]
            load = lambda suffix: inputs.local_json(directory / f"{dataset}_{suffix}.json")
            report_hash = hash_file(directory / f"{dataset}_retrieval.json")
        report, saved, diagnosis = load("retrieval"), load("metrics_development"), load("diagnosis")
        expected_type = "cross_encoder" if method == "reranker" else "hybrid_rrf" if method.startswith("rrf") else method
        require(report["config"]["retriever"] == saved["retriever"] == expected_type, f"Wrong method: {method}")
        require(report["manifest_sha256"] == saved["manifest_sha256"] == manifest_hash, "Report manifest mismatch")
        require(saved["dataset"] == dataset and saved["split"] == "dev", "Metrics dataset/split mismatch")
        require(saved["reporting_unlocked"] is False and set(saved["partitions"]) == {"diagnosis", "calibration"},
                "Only development aggregates are permitted")
        require(saved["source_hashes"] == hashes and saved["gold_sha256"] == gold_hash, "Metrics data/gold hash mismatch")
        require(saved["retrieval_sha256"] == report_hash, f"Retrieval SHA256 mismatch: {method}")
        require(saved["partition_counts"] == manifest["counts"], "Partition counts mismatch")
        rows = validate_report(report, dataset, "dev", queries, docs, hashes)
        diag = index(diagnosis, "query_id")
        require(set(diag) == diagnosis_ids, f"Diagnosis-only query coverage mismatch: {method}")
        reports[method], metrics[method], diagnoses[method] = report, saved, diag
        rows_by_method[method], report_hashes[method] = rows, report_hash
        for part in ("diagnosis", "calibration"):
            validate_summary(saved["partitions"][part], assignments, part,
                             metrics["bm25"]["partitions"][part] if method != "bm25" else None)
    rr = reports["reranker"]
    for key in ("model_id", "revision", "tokenizer_revision", "batch_size", "max_length", "top_k", "device", "source_depth"):
        require(rr["config"][key] == config[key], f"Reranker config mismatch: {key}")
    require(config["source_depth"] == 50 and config["top_k"] == 10, "Unexpected reranker depths")
    for key, path in (("script_sha256", "scripts/rerank_retrieval.py"),
                      ("validation_script_sha256", "scripts/fuse_retrieval_rrf.py")):
        require(rr[key] == digest(archive.arch_raw(path)), f"Reranker source hash mismatch: {key}")
    for method in ("bm25", "dense"):
        require(rr["rerank_sources"][method]["sha256"] == source_hashes[method], "Rerank pool source hash mismatch")
    for variant, expected in (("rrf_top10", {k: report_hashes[k] for k in ("bm25", "dense")}),
                              ("rrf_pool50", source_hashes)):
        for method in ("bm25", "dense"):
            require(reports[variant]["fusion_sources"][method]["sha256"] == expected[method], "RRF source hash mismatch")
    query_rows, denominators = [], {}
    for qid in sorted(diagnosis_ids):
        base = diagnoses["reranker"][qid]
        gold = base["gold"]  # The only gold object used is from a diagnosis export.
        result = dict(dataset=dataset, query_id=qid, group_id=assignments[qid]["group_id"],
                      query=queries[qid]["text"], metrics={})
        for method in reports:
            diag, row = diagnoses[method][qid], rows_by_method[method][qid]
            require(diag["gold"] == gold, "Diagnosis gold disagreement")
            require(diag["query"] == queries[qid]["text"] and diag["hits"] == row["hits"], "Diagnosis payload mismatch")
            require(diag["paper_id"] == queries[qid].get("paper_id"), "Diagnosis paper mismatch")
            ref_ids = {e for a in gold.get("annotations", []) for e in a["evidence_ids"]}
            ref_ids.update(e for a in gold.get("evidence_sets", []) for e in a["evidence_ids"])
            gold_docs = index(diag["gold_documents"], "evidence_id")
            require(set(gold_docs) == ref_ids and all(gold_docs[e] == docs[e] for e in ref_ids), "Diagnosis gold documents mismatch")
            metric, reason = score_query(dataset, gold, [h["evidence_id"] for h in row["hits"][:10]], docs)
            require(metric == diag["metrics"] and reason == diag["skip_reason"], "Diagnosis score/skip mismatch")
            result["metrics"][method] = metric
        pool = {}
        for source in source_rows:
            row = source[qid]
            require(len(row["hits"]) >= min(50, row["candidate_count"]), "Source lacks top50 hits")
            pool.update({h["evidence_id"]: h for h in row["hits"][:50]})
        rerank = rows_by_method["reranker"][qid]
        ids = [h["evidence_id"] for h in rerank["hits"]]
        require(set(ids) <= set(pool) and rerank["rerank_candidate_count"] == len(pool), "Reranker candidate mismatch")
        require(len(ids) == min(10, len(pool)), "Incomplete reranker top10")
        bucket, available = classify(dataset, gold, list(pool), ids, docs)
        result.update(bucket=bucket, pool_complete=available, pool_size=len(pool), skip_reason=base["skip_reason"],
                      minimum_valid_evidence_units=min((len(t) for t in targets(dataset, gold, docs)), default=None),
                      evidence_ids={m: [h["evidence_id"] for h in rows_by_method[m][qid]["hits"][:10]] for m in reports},
                      gold=gold, gold_documents=base["gold_documents"],
                      pool_ids=sorted(pool), reranker_hits=rerank["hits"][:10])
        query_rows.append(result)
    for method in reports:
        scored = [r["metrics"][method] for r in query_rows if r["metrics"][method] is not None]
        skipped = Counter(r["skip_reason"] for r in query_rows if r["metrics"][method] is None)
        expected = metrics[method]["partitions"]["diagnosis"]
        require(expected["scored_queries"] == len(scored) and expected["skipped_reasons"] == dict(skipped), "Diagnosis denominator mismatch")
        require(expected["statuses"] == dict(Counter(m["status"] for m in scored)), "Diagnosis statuses mismatch")
        names = set(scored[0]) - {"status"} if scored else set()
        require(set(expected["metrics"]) == names, "Diagnosis metric coverage mismatch")
        for name in names:
            assert_close(sum(m[name] for m in scored) / len(scored), expected["metrics"][name], name)
        denominators[method] = {p: {k: v for k, v in metrics[method]["partitions"][p].items() if k != "metrics"}
                                for p in ("diagnosis", "calibration")}
    complete = "complete_evidence_hit@10" if dataset == "qasper" else "complete_rationale_hit@10"
    paired = []
    for baseline in METHOD_DIRS:
        for name in ("recall@10", complete, "mrr@10"):
            paired.append(dict(dataset=dataset, baseline=baseline, metric=name,
                               **paired_bootstrap(query_rows, name, baseline, repetitions, seed)))
    inventory = {m: dict(retrieval_sha256=report_hashes[m], config=reports[m]["config"],
                        script_sha256=reports[m].get("script_sha256"), git=reports[m].get("git"),
                        hardware=reports[m].get("hardware")) for m in reports}
    return dict(dataset=dataset, query_rows=query_rows, metrics=metrics, paired=paired,
                denominator=denominators, inventory=inventory, manifest_sha256=manifest_hash,
                source_commit=source_commit, pool_counts=dict(Counter(r["bucket"] for r in query_rows)))


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_text(path, text):
    with path.open("x", encoding="utf8", newline="") as stream:
        stream.write(text)


def export(results, output, inputs, seed, repetitions, sample_size):
    aggregates, paired, query_csv, pool_csv = [], [], [], []
    lines = ["# Baseline diagnosis audit", "", "Offline artifact analysis; no models/retrieval/GPU were run.", "",
             "Diagnosis CIs are **exploratory**, percentile paired bootstrap 95%, not held-out confirmation. "
             f"Seed={seed}; draws={repetitions}. No multiple-comparison adjustment. "
             "QASPER resamples papers and computes macro over scored queries (not paper means); SciFact resamples claims. "
             "All diagnosis groups, including groups with no scored queries, enter resampling; zero-denominator draws are discarded and counted.", "",
             "Only diagnosis exports supply per-query gold. Full gold JSONL is hashed as opaque bytes, never decoded. "
             "Calibration numbers are copied from existing metrics; no calibration per-query scoring. "
             "Reporting is not scored or exported. Full prediction query coverage is checked without reporting gold.", "",
             "Pool complete means at least one entire eligible annotation/rationale is in the top50 BM25+dense union. "
             "QASPER uses (paper, NFC/whitespace-normalized text) equivalence; duplicate hits still consume top10 ranks. "
             "SciFact complete means any one rationale, while recall uses the union of rationales. "
             "Pool completeness is a diagnostic oracle, not an achievable selector score: some sets can require more than 10 units. "
             "Bucket labels are set-containment observations, not semantic causal diagnoses.", ""]
    review = ["# Diagnosis manual review", "", f"Seed={seed}; at most {sample_size} queries per dataset, "
              "round-robin stratified by observed bucket. Not a population error-rate sample. "
              "No semantic cause is assigned. Full diagnosis list is in diagnosis_queries.csv.", ""]
    for result in results:
        dataset, rows = result["dataset"], result["query_rows"]
        counts = result["pool_counts"]
        den = result["denominator"]["reranker"]["diagnosis"]
        scored = den["scored_queries"]
        lines += [f"## {dataset}", "", f"Diagnosis: selected={den['selected_queries']}, "
                  f"groups={den['groups']}, scored={scored}, skipped={den['skipped_count']}. "
                  f"Skip reasons: `{json.dumps(den['skipped_reasons'], sort_keys=True)}`.", "",
                  "| Observation | Queries | Fraction of scored |", "|---|---:|---:|"]
        for bucket in BUCKETS[:-1]:
            count = counts.get(bucket, 0)
            pool_csv.append(dict(dataset=dataset, bucket=bucket, queries=count, denominator=scored,
                                 fraction=count / scored))
            lines.append(f"| {bucket} | {count} | {count/scored:.4f} |")
        available = counts.get("selection_loss", 0) + counts.get("success", 0)
        lines += ["", f"Pool-complete oracle: {available}/{scored} = {available/scored:.4f}. "
                  f"Top10 failures among pool-complete queries: {counts.get('selection_loss',0)}/{available}. "
                  f"Pool-complete queries needing >10 normalized evidence units: "
                  f"{sum(r['pool_complete'] is True and r['minimum_valid_evidence_units'] > 10 for r in rows)}.", "",
                  "### Saved aggregates", "", "| Partition | Method | Scored / selected | Recall@10 | Complete@10 | MRR@10 |",
                  "|---|---|---:|---:|---:|---:|"]
        complete = "complete_evidence_hit@10" if dataset == "qasper" else "complete_rationale_hit@10"
        for part in ("diagnosis", "calibration"):
            for method, saved in result["metrics"].items():
                summary = saved["partitions"][part]
                for metric, value in summary["metrics"].items():
                    aggregates.append(dict(dataset=dataset, partition=part, method=method, metric=metric,
                                           value=value, selected_queries=summary["selected_queries"],
                                           scored_queries=summary["scored_queries"], skipped_count=summary["skipped_count"],
                                           groups=summary["groups"]))
                m = summary["metrics"]
                lines.append(f"| {part} | {method} | {summary['scored_queries']} / {summary['selected_queries']} | "
                             f"{m['recall@10']:.4f} | {m[complete]:.4f} | {m['mrr@10']:.4f} |")
        lines += ["", "### Paired reranker minus baseline (diagnosis only)", "",
                  "| Baseline | Metric | Delta | Exploratory 95% CI | Win / tie / loss queries |",
                  "|---|---|---:|---|---:|"]
        for item in result["paired"]:
            paired.append(item)
            lines.append(f"| {item['baseline']} | {item['metric']} | {item['delta']:+.4f} | "
                         f"[{item['ci_low']:+.4f}, {item['ci_high']:+.4f}] | "
                         f"{item['wins']} / {item['ties']} / {item['losses']} |")
        for row in rows:
            record = {k: row[k] for k in ("dataset", "query_id", "group_id", "bucket", "pool_complete", "pool_size",
                                           "minimum_valid_evidence_units", "skip_reason")}
            record.update(query=row["query"], pool_ids=json.dumps(row["pool_ids"]),
                          top10_ids=json.dumps(row["evidence_ids"], sort_keys=True),
                          diagnosis_metrics=json.dumps(row["metrics"], sort_keys=True),
                          manual_category="", manual_evidence="", proposed_fix="")
            query_csv.append(record)
        strata = {b: sorted([r for r in rows if r["bucket"] == b], key=lambda r: r["query_id"]) for b in BUCKETS}
        rng = random.Random(seed)
        for bucket in BUCKETS:
            rng.shuffle(strata[bucket])
        sample = []
        while len(sample) < sample_size and any(strata.values()):
            for bucket in BUCKETS:
                if strata[bucket] and len(sample) < sample_size:
                    sample.append(strata[bucket].pop())
        for row in sample:
            review += [f"## {dataset} / {row['query_id']} / {row['bucket']}", "", "```json",
                       json.dumps({k: row[k] for k in ("query_id", "group_id", "query", "bucket", "pool_complete",
                           "minimum_valid_evidence_units", "pool_ids", "evidence_ids", "metrics", "gold", "gold_documents", "reranker_hits")},
                           ensure_ascii=False, indent=2), "```", "", "- category (manual; uncertain until reviewed):",
                       "- observation and supporting evidence:", "- proposed fix / counterexample:", ""]
    lines += ["", "## Limits and next decision", "",
              "VERIFIED_COMPLETE is an integrity/completion label, not proof of superiority. "
              "The primary RRF comparison for the reranker pool is rrf_pool50; rrf_top10 uses a shallower input pool. "
              "No scores are pooled across datasets. Calibration aggregates are not a new independent confirmation after tuning. "
              "No official QA/verification scoring or reader experiment has been run by this audit. "
              "Latency is not compared: runs differ in hardware/protocol, and rerank/fusion timings exclude retrieval.", "",
              "Review selection_loss versus pool_missing examples before choosing a new selector. "
              "Do not infer lexical/semantic/truncation causes from metric changes alone. "
              "Freeze one ablation and its context/token budget before any future calibration use; keep reporting sealed.", "",
              "See provenance.json for exact input hashes, source/model metadata, denominators and bootstrap settings.", ""]
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("aggregate_metrics.csv", aggregates), ("paired_diagnosis.csv", paired),
                         ("diagnosis_queries.csv", query_csv), ("pool_diagnosis.csv", pool_csv)):
        write_text(output / name, csv_text(values))
    write_text(output / "REPORT.md", "\n".join(lines))
    write_text(output / "MANUAL_REVIEW.md", "\n".join(review))
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=inputs.root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    write_new(output / "provenance.json", dict(
        analysis_git_head=git_commit, analysis_script_sha256=hash_file(__file__), seed=seed,
        bootstrap_draws=repetitions, confidence=.95, ci_scope="diagnosis exploratory; no multiplicity adjustment",
        full_gold_policy="opaque byte hashing only; diagnosis-export gold exclusively scored",
        inputs=inputs.files, archive_manifest=inputs.manifest,
        datasets={r["dataset"]: {k: r[k] for k in ("inventory", "manifest_sha256", "source_commit", "denominator", "pool_counts")}
                  for r in results}))
    outputs = {p.name: hash_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_new(output / "output_hashes.json", outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--review-size", type=int, default=60)
    args = parser.parse_args()
    require(args.bootstrap_draws >= 100 and args.review_size > 0, "Invalid bootstrap/review size")
    require(not args.output_dir.exists(), f"Refusing to overwrite output directory: {args.output_dir}")
    inputs = Inputs(args.root, args.archive, args.sha256_file)
    try:
        results = []
        for dataset in ("qasper", "scifact"):
            print(f"Validating and analyzing diagnosis only: {dataset}", flush=True)
            results.append(analyze_dataset(inputs, dataset, args.seed, args.bootstrap_draws))
        export(results, args.output_dir, inputs, args.seed, args.bootstrap_draws, args.review_size)
        print(f"Saved offline audit to {args.output_dir}")
    finally:
        inputs.zip.close()


if __name__ == "__main__":
    main()
