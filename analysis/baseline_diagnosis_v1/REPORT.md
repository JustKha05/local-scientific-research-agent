# Baseline diagnosis audit

Offline artifact analysis; no models/retrieval/GPU were run.

Diagnosis CIs are **exploratory**, percentile paired bootstrap 95%, not held-out confirmation. Seed=42; draws=10000. No multiple-comparison adjustment. QASPER resamples papers and computes macro over scored queries (not paper means); SciFact resamples claims. All diagnosis groups, including groups with no scored queries, enter resampling; zero-denominator draws are discarded and counted.

Only diagnosis exports supply per-query gold. Full gold JSONL is hashed as opaque bytes, never decoded. Calibration numbers are copied from existing metrics; no calibration per-query scoring. Reporting is not scored or exported. Full prediction query coverage is checked without reporting gold.

Pool complete means at least one entire eligible annotation/rationale is in the top50 BM25+dense union. QASPER uses (paper, NFC/whitespace-normalized text) equivalence; duplicate hits still consume top10 ranks. SciFact complete means any one rationale, while recall uses the union of rationales. Pool completeness is a diagnostic oracle, not an achievable selector score: some sets can require more than 10 units. Bucket labels are set-containment observations, not semantic causal diagnoses.

## qasper

Diagnosis: selected=416, groups=113, scored=341, skipped=75. Skip reasons: `{"no_eligible_text_annotation": 75}`.

| Observation | Queries | Fraction of scored |
|---|---:|---:|
| pool_missing | 11 | 0.0323 |
| selection_loss | 97 | 0.2845 |
| success | 233 | 0.6833 |

Pool-complete oracle: 330/341 = 0.9677. Top10 failures among pool-complete queries: 97/330. Pool-complete queries needing >10 normalized evidence units: 0.

### Saved aggregates

| Partition | Method | Scored / selected | Recall@10 | Complete@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| diagnosis | bm25 | 341 / 416 | 0.6262 | 0.5543 | 0.2976 |
| diagnosis | dense | 341 / 416 | 0.5794 | 0.5044 | 0.2968 |
| diagnosis | rrf_top10 | 341 / 416 | 0.6353 | 0.5601 | 0.3861 |
| diagnosis | rrf_pool50 | 341 / 416 | 0.6698 | 0.6041 | 0.3964 |
| diagnosis | reranker | 341 / 416 | 0.7508 | 0.6833 | 0.4525 |
| calibration | bm25 | 165 / 206 | 0.6424 | 0.5636 | 0.3699 |
| calibration | dense | 165 / 206 | 0.5943 | 0.5091 | 0.2963 |
| calibration | rrf_top10 | 165 / 206 | 0.6399 | 0.5636 | 0.3814 |
| calibration | rrf_pool50 | 165 / 206 | 0.6704 | 0.5818 | 0.3919 |
| calibration | reranker | 165 / 206 | 0.7492 | 0.6727 | 0.5024 |

### Paired reranker minus baseline (diagnosis only)

| Baseline | Metric | Delta | Exploratory 95% CI | Win / tie / loss queries |
|---|---|---:|---|---:|
| bm25 | recall@10 | +0.1246 | [+0.0796, +0.1693] | 77 / 240 / 24 |
| bm25 | complete_evidence_hit@10 | +0.1290 | [+0.0823, +0.1768] | 57 / 271 / 13 |
| bm25 | mrr@10 | +0.1550 | [+0.1177, +0.1932] | 174 / 101 / 66 |
| dense | recall@10 | +0.1715 | [+0.1252, +0.2181] | 91 / 228 / 22 |
| dense | complete_evidence_hit@10 | +0.1789 | [+0.1265, +0.2305] | 73 / 256 / 12 |
| dense | mrr@10 | +0.1557 | [+0.1166, +0.1933] | 183 / 95 / 63 |
| rrf_top10 | recall@10 | +0.1155 | [+0.0724, +0.1568] | 74 / 244 / 23 |
| rrf_top10 | complete_evidence_hit@10 | +0.1232 | [+0.0751, +0.1696] | 55 / 273 / 13 |
| rrf_top10 | mrr@10 | +0.0664 | [+0.0201, +0.1117] | 143 / 111 / 87 |
| rrf_pool50 | recall@10 | +0.0810 | [+0.0381, +0.1230] | 61 / 250 / 30 |
| rrf_pool50 | complete_evidence_hit@10 | +0.0792 | [+0.0330, +0.1256] | 44 / 280 / 17 |
| rrf_pool50 | mrr@10 | +0.0561 | [+0.0097, +0.1020] | 140 / 116 / 85 |
## scifact

Diagnosis: selected=120, groups=120, scored=71, skipped=49. Skip reasons: `{"no_annotated_evidence": 49}`.

| Observation | Queries | Fraction of scored |
|---|---:|---:|
| pool_missing | 9 | 0.1268 |
| selection_loss | 3 | 0.0423 |
| success | 59 | 0.8310 |

Pool-complete oracle: 62/71 = 0.8732. Top10 failures among pool-complete queries: 3/62. Pool-complete queries needing >10 normalized evidence units: 0.

### Saved aggregates

| Partition | Method | Scored / selected | Recall@10 | Complete@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| diagnosis | bm25 | 71 / 120 | 0.4576 | 0.6056 | 0.3979 |
| diagnosis | dense | 71 / 120 | 0.6137 | 0.7746 | 0.4850 |
| diagnosis | rrf_top10 | 71 / 120 | 0.6184 | 0.7746 | 0.5227 |
| diagnosis | rrf_pool50 | 71 / 120 | 0.5942 | 0.7465 | 0.4930 |
| diagnosis | reranker | 71 / 120 | 0.6852 | 0.8310 | 0.6108 |
| calibration | bm25 | 39 / 60 | 0.3825 | 0.5128 | 0.3650 |
| calibration | dense | 39 / 60 | 0.5107 | 0.6667 | 0.4286 |
| calibration | rrf_top10 | 39 / 60 | 0.5620 | 0.7436 | 0.3854 |
| calibration | rrf_pool50 | 39 / 60 | 0.4615 | 0.5385 | 0.3513 |
| calibration | reranker | 39 / 60 | 0.5214 | 0.6410 | 0.4832 |

### Paired reranker minus baseline (diagnosis only)

| Baseline | Metric | Delta | Exploratory 95% CI | Win / tie / loss queries |
|---|---|---:|---|---:|
| bm25 | recall@10 | +0.2276 | [+0.1473, +0.3165] | 26 / 44 / 1 |
| bm25 | complete_rationale_hit@10 | +0.2254 | [+0.1324, +0.3247] | 16 / 55 / 0 |
| bm25 | mrr@10 | +0.2129 | [+0.1320, +0.2970] | 34 / 32 / 5 |
| dense | recall@10 | +0.0715 | [+0.0093, +0.1405] | 12 / 51 / 8 |
| dense | complete_rationale_hit@10 | +0.0563 | [-0.0154, +0.1343] | 6 / 63 / 2 |
| dense | mrr@10 | +0.1258 | [+0.0379, +0.2131] | 28 / 32 / 11 |
| rrf_top10 | recall@10 | +0.0668 | [+0.0192, +0.1226] | 10 / 58 / 3 |
| rrf_top10 | complete_rationale_hit@10 | +0.0563 | [+0.0132, +0.1169] | 4 / 67 / 0 |
| rrf_top10 | mrr@10 | +0.0880 | [+0.0166, +0.1628] | 22 / 41 / 8 |
| rrf_pool50 | recall@10 | +0.0910 | [+0.0332, +0.1567] | 13 / 54 / 4 |
| rrf_pool50 | complete_rationale_hit@10 | +0.0845 | [+0.0267, +0.1558] | 6 / 65 / 0 |
| rrf_pool50 | mrr@10 | +0.1178 | [+0.0399, +0.1993] | 25 / 40 / 6 |

## Limits and next decision

VERIFIED_COMPLETE is an integrity/completion label, not proof of superiority. The primary RRF comparison for the reranker pool is rrf_pool50; rrf_top10 uses a shallower input pool. No scores are pooled across datasets. Calibration aggregates are not a new independent confirmation after tuning. No official QA/verification scoring or reader experiment has been run by this audit. Latency is not compared: runs differ in hardware/protocol, and rerank/fusion timings exclude retrieval.

Review selection_loss versus pool_missing examples before choosing a new selector. Do not infer lexical/semantic/truncation causes from metric changes alone. Freeze one ablation and its context/token budget before any future calibration use; keep reporting sealed.

See provenance.json for exact input hashes, source/model metadata, denominators and bootstrap settings.
