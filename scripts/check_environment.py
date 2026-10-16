import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def request_json(url, timeout, payload=None):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers)

    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Also run a short local inference.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / "local.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_url = config["ollama_base_url"].rstrip("/")

    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"OS: {platform.platform()}")

    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Virtual environment is not being used. "
            "Run with .\\.venv\\Scripts\\python.exe"
        )

    tags = request_json(f"{base_url}/api/tags", timeout=10)
    installed = [
        item["name"]
        for item in tags.get("models", [])
    ]

    print("Ollama: OK")
    print(f"Installed models: {', '.join(installed) or '(none)'}")

    model = config["model"]

    if model not in installed:
        raise RuntimeError(
            f"Model '{model}' is not installed. "
            f"Run: ollama pull {model}"
        )

    print(f"Configured model: {model} — OK")

    if not args.infer:
        print("Environment check: PASSED")
        return

    context = (
        "In a small controlled experiment, Method A achieved "
        "82% accuracy and Method B achieved 78% accuracy. "
        "Both methods were evaluated on Dataset X."
    )
    question = (
        "Which method achieved higher accuracy, "
        "and on which dataset?"
    )

    payload = {
        "model": model,
        "system": (
            "Answer only from the supplied context. "
            "Use one short sentence and cite [S1]. "
            "If the answer is missing, say insufficient evidence."
        ),
        "prompt": (
            f"Context:\n[S1] {context}\n\n"
            f"Question: {question}"
        ),
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": config["temperature"],
            "num_ctx": config["num_ctx"],
            "num_predict": config["num_predict"],
        },
    }

    print("Running local inference...")
    started = time.perf_counter()

    result = request_json(
        f"{base_url}/api/generate",
        timeout=config["timeout_seconds"],
        payload=payload,
    )

    elapsed = time.perf_counter() - started
    answer = result.get("response", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config,
        "context": context,
        "question": question,
        "answer": answer,
        "elapsed_seconds": round(elapsed, 3),
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
    }

    report_path = (
        PROJECT_ROOT / "evaluation" / "reports" / "smoke_test.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nAnswer: {answer}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Report: {report_path}")
    print("Inference connection: PASSED")
    print("Review the answer manually; this is not a benchmark.")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR {exc.code}: {detail}", file=sys.stderr)
        sys.exit(1)
    except (URLError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)