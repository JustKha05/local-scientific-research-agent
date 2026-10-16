import json
from collections import defaultdict
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_qasper(split):
    source = RAW / "qasper" / f"qasper-{split}-v0.3.json"
    papers = json.loads(source.read_text(encoding="utf-8"))

    documents = []
    queries = []
    gold = []
    unmatched_count = 0
    figure_table_count = 0

    for paper_id, paper in papers.items():
        paper_ref = f"qasper:{paper_id}"
        exact_lookup = defaultdict(list)
        normalized_lookup = defaultdict(list)

        for section_index, section in enumerate(paper["full_text"]):
            section_name = section.get("section_name")

            # Section titles can also be annotated as evidence.
            if isinstance(section_name, str) and section_name.strip():
                evidence_id = f"{paper_ref}:s{section_index}:title"

                documents.append({
                    "evidence_id": evidence_id,
                    "dataset": "qasper",
                    "split": split,
                    "paper_id": paper_ref,
                    "source_paper_id": paper_id,
                    "title": paper["title"],
                    "unit_type": "section_title",
                    "section_name": section_name,
                    "section_index": section_index,
                    "paragraph_index": None,
                    "text": section_name,
                })

                exact_lookup[section_name].append(evidence_id)
                normalized_lookup[
                    normalize_for_matching(section_name)
                ].append(evidence_id)

            for paragraph_index, text in enumerate(section["paragraphs"]):
                if not text.strip():
                    continue

                evidence_id = (
                    f"{paper_ref}:s{section_index}:p{paragraph_index}"
                )

                documents.append({
                    "evidence_id": evidence_id,
                    "dataset": "qasper",
                    "split": split,
                    "paper_id": paper_ref,
                    "source_paper_id": paper_id,
                    "title": paper["title"],
                    "unit_type": "paragraph",
                    "section_name": section_name,
                    "section_index": section_index,
                    "paragraph_index": paragraph_index,
                    "text": text,
                })

                exact_lookup[text].append(evidence_id)
                normalized_lookup[
                    normalize_for_matching(text)
                ].append(evidence_id)

        for qa in paper["qas"]:
            query_id = f"qasper:{qa['question_id']}"

            queries.append({
                "query_id": query_id,
                "dataset": "qasper",
                "split": split,
                "mode": "paper_qa",
                "paper_id": paper_ref,
                "text": qa["question"],
            })

            annotations = []

            for annotation in qa["answers"]:
                answer = annotation["answer"]
                mapped_ids = set()
                unmatched = []
                non_text = []

                for evidence_text in answer.get("evidence", []):
                    if evidence_text.startswith("FLOAT SELECTED"):
                        non_text.append(evidence_text)
                        figure_table_count += 1
                        continue

                    # Prefer exact matching.
                    matched_ids = exact_lookup.get(evidence_text)

                    # Fall back only to whitespace/Unicode normalization.
                    if not matched_ids:
                        matched_ids = normalized_lookup.get(
                            normalize_for_matching(evidence_text)
                        )

                    if matched_ids:
                        mapped_ids.update(matched_ids)
                    else:
                        unmatched.append(evidence_text)
                        unmatched_count += 1

                annotations.append({
                    "annotation_id": annotation.get("annotation_id"),
                    "answer": answer,
                    "evidence_ids": sorted(mapped_ids),
                    "unmatched_text_evidence": unmatched,
                    "figure_table_evidence": non_text,
                })

            gold.append({
                "query_id": query_id,
                "annotations": annotations,
            })

    output = PROCESSED / "qasper" / split
    write_jsonl(output / "documents.jsonl", documents)
    write_jsonl(output / "queries.jsonl", queries)
    write_jsonl(output / "gold.jsonl", gold)

    print(
        f"QASPER {split}: documents={len(documents)}, "
        f"queries={len(queries)}, gold={len(gold)}"
    )
    print(
        f"  Unmatched text evidence={unmatched_count}, "
        f"figure/table evidence={figure_table_count}"
    )

def normalize_for_matching(text):
    return " ".join(
        unicodedata.normalize("NFC", text).split()
    )

def prepare_scifact():
    directory = RAW / "scifact"
    corpus_rows = read_jsonl(directory / "corpus.jsonl")
    corpus = {str(row["doc_id"]): row for row in corpus_rows}

    documents = []

    for doc_id, paper in corpus.items():
        paper_ref = f"scifact:{doc_id}"

        for sentence_index, text in enumerate(paper["abstract"]):
            # Preserve every original sentence index.
            documents.append({
                "evidence_id": f"{paper_ref}:s{sentence_index}",
                "dataset": "scifact",
                "paper_id": paper_ref,
                "source_paper_id": doc_id,
                "title": paper["title"],
                "sentence_index": sentence_index,
                "text": text,
            })

    output = PROCESSED / "scifact"
    write_jsonl(output / "documents.jsonl", documents)

    for split in ("train", "dev", "test"):
        claims = read_jsonl(directory / f"claims_{split}.jsonl")
        queries = []
        gold = []

        for claim in claims:
            query_id = f"scifact:{claim['id']}"

            # No gold evidence IDs or cited_doc_ids in query inputs.
            queries.append({
                "query_id": query_id,
                "dataset": "scifact",
                "split": split,
                "mode": "claim_verification",
                "text": claim["claim"],
            })

            if split == "test":
                continue

            evidence_sets = []

            for doc_id, rationales in claim["evidence"].items():
                doc_id = str(doc_id)

                require(
                    doc_id in corpus,
                    f"Unknown SciFact document: {doc_id}",
                )

                for rationale in rationales:
                    indices = rationale["sentences"]

                    require(
                        all(
                            0 <= index < len(corpus[doc_id]["abstract"])
                            for index in indices
                        ),
                        f"Invalid sentence index in claim {claim['id']}",
                    )

                    # Preserve alternative rationale sets separately.
                    evidence_sets.append({
                        "paper_id": f"scifact:{doc_id}",
                        "label": rationale["label"],
                        "evidence_ids": [
                            f"scifact:{doc_id}:s{index}"
                            for index in indices
                        ],
                    })

            gold.append({
                "query_id": query_id,
                "evidence_sets": evidence_sets,
                "has_annotated_evidence": bool(claim["evidence"]),
            })

        split_output = output / split
        write_jsonl(split_output / "queries.jsonl", queries)

        if split != "test":
            write_jsonl(split_output / "gold.jsonl", gold)

        print(
            f"SciFact {split}: queries={len(queries)}, "
            f"gold={len(gold)}"
        )

    print(f"SciFact corpus sentences: {len(documents)}")


def main():
    for split in ("train", "dev", "test"):
        prepare_qasper(split)

    prepare_scifact()
    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()