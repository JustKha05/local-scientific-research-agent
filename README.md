# Local Scientific Research Agent with Evidence Verification

Scientific paper QA and claim verification with traceable evidence.

## Research direction
Evidence selection and answer / retrieve-more / abstain decisions under a limited context and compute budget. QASPER is the primary paper-scoped QA task; SciFact is the secondary corpus-scoped claim verification task. Research benefits are hypotheses pending experiments.

- [Research plan](docs/RESEARCH_PLAN.md)
- [Progress, evidence and remaining work](docs/PROGRESS.md)
- [Next steps: offline baseline analysis](docs/NEXT_STEPS.md)
- [Diagnosis baseline report](analysis/baseline_diagnosis_v1/REPORT.md)
- [Dataset preparation](docs/datasets.md)

## Verified implementation status
On `feat/dev-retrieval-benchmark`, data preparation, full-dev retrieval/evaluation,
BM25, pinned MiniLM dense, RRF and pinned cross-encoder reranking are implemented.
Local artifacts cover 1,005 QASPER and 300 SciFact dev predictions. Diagnosis and
calibration evaluation exist; reporting remains sealed. The reranker Colab run
and its versioned backup completed; `VERIFIED_COMPLETE` proves artifact integrity
and completion, not superiority over baselines.

The next step is diagnosis-only pool/selection analysis and manual review before
one evidence-selection ablation. See [PROGRESS](docs/PROGRESS.md) for validated
denominators, hashes, results and limits. Generation, a new evidence selector,
an adaptive controller and final held-out evaluation remain unimplemented.

## Offline analysis (no model, GPU or retrieval)

The required local archive and baseline report paths are documented in
[NEXT_STEPS](docs/NEXT_STEPS.md). Outputs are immutable: choose a new directory
for a deliberate re-analysis. No reporting flag exists in the analysis script.

```powershell
.\.venv\Scripts\python.exe -m unittest scripts.test_analyze_baseline_diagnosis -v
```

