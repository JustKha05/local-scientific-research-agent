import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = [
    {
        "dataset": "qasper",
        "url": (
            "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
            "qasper-train-dev-v0.3.tgz"
        ),
        "files": [
            "qasper-train-v0.3.json",
            "qasper-dev-v0.3.json",
        ],
    },
    {
        "dataset": "qasper",
        "url": (
            "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
            "qasper-test-and-evaluator-v0.3.tgz"
        ),
        "files": [
            "qasper-test-v0.3.json",
        ],
    },
    {
        "dataset": "scifact",
        "url": (
            "https://scifact.s3-us-west-2.amazonaws.com/"
            "release/latest/data.tar.gz"
        ),
        "files": [
            "corpus.jsonl",
            "claims_train.jsonl",
            "claims_dev.jsonl",
            "claims_test.jsonl",
        ],
    },
]


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def download(url, destination):
    if destination.exists():
        print(f"Using existing archive: {destination.name}")
        return

    temporary = destination.with_name(destination.name + ".part")
    request = Request(
        url,
        headers={"User-Agent": "local-scientific-research-agent/0.1"},
    )

    print(f"Downloading: {url}")

    try:
        with urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)

        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def extract_selected(archive, output_dir, filenames):
    # Copy only expected regular files.
    # Do not execute or extract scripts from the archive.
    with tarfile.open(archive, "r:*") as package:
        members = package.getmembers()

        for filename in filenames:
            matches = [
                member
                for member in members
                if member.isfile()
                and Path(member.name).name == filename
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one '{filename}' in {archive.name}, "
                    f"found {len(matches)}."
                )

            source = package.extractfile(matches[0])

            if source is None:
                raise RuntimeError(f"Cannot read {filename}")

            destination = output_dir / filename
            temporary = destination.with_name(filename + ".part")

            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)

                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

            print(f"  Ready: {destination.relative_to(ROOT)}")


def main():
    records = []

    for source in SOURCES:
        output_dir = RAW / source["dataset"]
        output_dir.mkdir(parents=True, exist_ok=True)

        archive_name = source["url"].rsplit("/", 1)[-1]
        archive = output_dir / archive_name

        download(source["url"], archive)
        extract_selected(archive, output_dir, source["files"])

        records.append({
            "dataset": source["dataset"],
            "source_url": source["url"],
            "archive": str(archive.relative_to(ROOT)),
            "archive_sha256": sha256(archive),
            "files": [
                {
                    "path": str((output_dir / name).relative_to(ROOT)),
                    "sha256": sha256(output_dir / name),
                    "bytes": (output_dir / name).stat().st_size,
                }
                for name in source["files"]
            ],
        })

    # Store provenance outside data/ so Git can track it.
    manifest_path = ROOT / "configs" / "data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "sources": records,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nManifest: {manifest_path}")
    print("Download completed.")


if __name__ == "__main__":
    main()