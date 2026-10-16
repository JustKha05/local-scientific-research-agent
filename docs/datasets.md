# Dataset sources and usage

## QASPER

- Purpose: scientific paper question answering and evidence retrieval.
- Version: v0.3.
- Source: https://allenai.org/data/qasper
- Dataset card: https://huggingface.co/datasets/allenai/qasper
- License: CC BY 4.0.
- Preserve the official train/dev/test splits.
- Use train for debugging and dev for configuration selection.
- Reserve test for final evaluation.
- Reference: Dasigi et al. (2021),
  A Dataset of Information-Seeking Questions and Answers
  Anchored in Research Papers.

## SciFact

- Purpose: scientific claim verification and evidence retrieval.
- Source: https://github.com/allenai/scifact
- Claims and evidence annotations: CC BY 4.0.
- Corpus abstracts: ODC-By 1.0, from S2ORC.
- Official test labels are not public.
- Use training claims for development and internal tuning.
- Reserve official dev for final local evaluation.
- Report results as official-dev results, not test results.
- Reference: Wadden et al. (2020),
  Fact or Fiction: Verifying Scientific Claims.

## Data handling

- Keep source JSON/JSONL files unchanged.
- Record source URLs and SHA-256 hashes in configs/data_manifest.json.
- Do not add raw datasets to Git.
- Never index gold answers, evidence annotations, or labels
  as retrievable document content.
- QASPER retrieval is restricted to the specified paper.
- SciFact open retrieval uses the corpus, not cited_doc_ids.
- Treat QASPER figure/table evidence separately during preprocessing.