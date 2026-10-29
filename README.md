# Local Scientific Research Agent with Evidence Verification

Scientific paper QA and claim verification with traceable evidence.

## Research direction
Evidence selection and answer / retrieve-more / abstain decisions under a limited context and compute budget. QASPER is the primary paper-scoped QA task; SciFact is the secondary corpus-scoped claim verification task. Research benefits are hypotheses pending experiments.

- [Research plan](docs/RESEARCH_PLAN.md)
- [Next steps and verified Windows commands](docs/NEXT_STEPS.md)
- [Dataset preparation](docs/datasets.md)

## Verified implementation status
Data preparation, validation, BM25 and MiniLM dense retrieval smoke runners exist on feat/dense-retrieval. Current smoke evaluation uses 30 TRAIN queries per dataset and is not a final benchmark. The dense smoke runner currently requires CPU.
Full-dev benchmark, hybrid/reranker, generation and adaptive controller are planned.

## Environment check
```powershell
.\.venv\Scripts\python.exe -m scripts.check_environment
```

