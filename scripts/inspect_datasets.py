import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def inspect_qasper():
    print("\n=== QASPER ===")

    paper_ids = {}
    question_ids = {}

    for split in ("train", "dev", "test"):
        path = RAW / "qasper" / f"qasper-{split}-v0.3.json"
        papers = json.loads(path.read_text(encoding="utf-8"))

        require(isinstance(papers, dict), f"{path}: expected dictionary")

        paper_ids[split] = set(papers)
        ids = []
        question_count = 0
        answer_count = 0
        paragraph_count = 0
        answer_types = Counter()

        for paper_id, paper in papers.items():
            require(
                isinstance(paper.get("full_text"), list),
                f"{paper_id}: invalid full_text",
            )

            for section in paper["full_text"]:
                paragraphs = section["paragraphs"]
                require(
                    all(isinstance(p, str) for p in paragraphs),
                    f"{paper_id}: invalid paragraph",
                )
                paragraph_count += len(paragraphs)

            for qa in paper["qas"]:
                question_count += 1
                ids.append(qa["question_id"])

                for annotation in qa["answers"]:
                    answer_count += 1
                    answer = annotation["answer"]

                    if answer.get("unanswerable"):
                        answer_types["unanswerable"] += 1
                    elif answer.get("yes_no") is not None:
                        answer_types["yes_no"] += 1
                    elif answer.get("extractive_spans"):
                        answer_types["extractive"] += 1
                    elif answer.get("free_form_answer"):
                        answer_types["free_form"] += 1
                    else:
                        answer_types["other"] += 1

        require(
            len(ids) == len(set(ids)),
            f"Duplicate question IDs in QASPER {split}",
        )
        question_ids[split] = set(ids)

        print(
            f"{split}: papers={len(papers)}, "
            f"questions={question_count}, "
            f"annotations={answer_count}, "
            f"paragraphs={paragraph_count}"
        )
        print(f"  Answer types: {dict(answer_types)}")

        # Show only a training example, not held-out test content.
        if split == "train":
            example = next(
                paper for paper in papers.values() if paper["qas"]
            )
            print(f"  Example title: {example['title']}")
            print(f"  Example question: {example['qas'][0]['question']}")

    for left, right in (
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ):
        require(
            paper_ids[left].isdisjoint(paper_ids[right]),
            f"QASPER paper overlap: {left}/{right}",
        )
        require(
            question_ids[left].isdisjoint(question_ids[right]),
            f"QASPER question overlap: {left}/{right}",
        )

    print("QASPER split ID checks: PASSED")


def inspect_scifact():
    print("\n=== SciFact ===")

    directory = RAW / "scifact"
    documents = read_jsonl(directory / "corpus.jsonl")
    corpus = {str(doc["doc_id"]): doc for doc in documents}

    require(
        len(corpus) == len(documents),
        "Duplicate SciFact document IDs",
    )

    for doc in documents:
        require(
            isinstance(doc["abstract"], list)
            and all(isinstance(s, str) for s in doc["abstract"]),
            f"Invalid abstract: {doc['doc_id']}",
        )

    print(f"Corpus documents: {len(corpus)}")

    split_ids = {}

    for split in ("train", "dev", "test"):
        claims = read_jsonl(directory / f"claims_{split}.jsonl")
        ids = [claim["id"] for claim in claims]
        labels = Counter()
        without_evidence = 0

        require(
            len(ids) == len(set(ids)),
            f"Duplicate SciFact claim IDs in {split}",
        )
        split_ids[split] = set(ids)

        for claim in claims:
            require(
                isinstance(claim["claim"], str),
                f"Invalid claim: {claim['id']}",
            )

            # Test gold labels are not publicly available.
            if split == "test":
                continue

            evidence = claim["evidence"]

            if not evidence:
                without_evidence += 1

            for doc_id, rationales in evidence.items():
                require(
                    str(doc_id) in corpus,
                    f"Unknown evidence document: {doc_id}",
                )
                sentences = corpus[str(doc_id)]["abstract"]

                for rationale in rationales:
                    label = rationale["label"]
                    require(
                        label in {"SUPPORT", "CONTRADICT"},
                        f"Unexpected label: {label}",
                    )
                    labels[label] += 1

                    for index in rationale["sentences"]:
                        require(
                            0 <= index < len(sentences),
                            f"Invalid sentence index: {doc_id}/{index}",
                        )

        print(f"{split}: claims={len(claims)}")

        if split != "test":
            print(f"  Rationale labels: {dict(labels)}")
            print(f"  Claims without annotated evidence: {without_evidence}")
        else:
            print("  Gold evaluation skipped: test is unlabeled")

        if split == "train" and claims:
            print(f"  Example claim: {claims[0]['claim']}")

    for left, right in (
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ):
        require(
            split_ids[left].isdisjoint(split_ids[right]),
            f"SciFact claim overlap: {left}/{right}",
        )

    print("SciFact ID and evidence index checks: PASSED")


def main():
    inspect_qasper()
    inspect_scifact()
    print("\nRaw data inspection: PASSED")


if __name__ == "__main__":
    main()