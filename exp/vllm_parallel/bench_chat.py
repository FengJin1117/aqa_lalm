#!/usr/bin/env python3
"""Simple OpenAI-compatible chat benchmark for vLLM."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "Qwen3-4B-Instruct-2507"
DEFAULT_PROMPTS = [
    "用三句话介绍一下Transformer的核心思想。",
    "解释一下什么是注意力机制，回答要简洁。",
    "给出两个提升LLM推理吞吐的常见方法。",
    "What is vLLM and why is continuous batching useful?",
    "请判断：音频问答评测中，caption-based QA和direct audio QA的主要区别是什么？",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM /v1/chat/completions with fixed prompts."
    )
    parser.add_argument(
        "--base-urls",
        nargs="+",
        required=True,
        help="One or more OpenAI-compatible base URLs, e.g. http://127.0.0.1:8000/v1.",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--name", default="chat_bench")
    parser.add_argument(
        "--output-dir",
        default="exp/vllm_parallel/results",
        help="Directory for JSONL details and summary JSON.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional UTF-8 text file with one prompt per line.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip /v1/models health checks before benchmarking.",
    )
    return parser.parse_args()


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def load_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    prompts = [
        line.strip()
        for line in Path(prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def get_json(url: str, timeout: float) -> tuple[int, dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def health_check(base_urls: list[str], timeout: float) -> None:
    errors: list[str] = []
    for base_url in base_urls:
        url = f"{base_url}/models"
        try:
            status, _ = get_json(url, timeout)
            if status >= 400:
                errors.append(f"{url}: HTTP {status}")
        except Exception as exc:  # noqa: BLE001 - keep health errors explicit.
            errors.append(f"{url}: {exc}")
    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError(f"vLLM health check failed:\n  - {joined}")


def request_once(
    request_id: int,
    base_url: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    started = time.perf_counter()
    record: dict[str, Any] = {
        "request_id": request_id,
        "base_url": base_url,
        "prompt": prompt,
        "ok": False,
    }
    try:
        status, response = post_json(
            f"{base_url}/chat/completions", payload, args.timeout
        )
        latency = time.perf_counter() - started
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        usage = response.get("usage", {})
        record.update(
            {
                "ok": 200 <= status < 300,
                "status": status,
                "latency_s": latency,
                "output_chars": len(content),
                "usage": usage,
                "error": None,
            }
        )
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - started
        error_body = exc.read().decode("utf-8", errors="replace")
        record.update(
            {
                "status": exc.code,
                "latency_s": latency,
                "error": error_body[:1000],
            }
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures.
        latency = time.perf_counter() - started
        record.update({"latency_s": latency, "error": repr(exc)})
    return record


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(args: argparse.Namespace, records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    success_records = [record for record in records if record.get("ok")]
    latencies = [float(record["latency_s"]) for record in success_records]
    num_requests = len(records)
    success_count = len(success_records)
    failure_count = num_requests - success_count
    return {
        "name": args.name,
        "model": args.model,
        "base_urls": args.base_urls,
        "concurrency": args.concurrency,
        "num_requests": num_requests,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / num_requests if num_requests else 0.0,
        "elapsed_s": elapsed,
        "throughput_req_s": success_count / elapsed if elapsed > 0 else 0.0,
        "latency_avg_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "latency_p99_s": percentile(latencies, 99),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }


def main() -> None:
    args = parse_args()
    args.base_urls = [normalize_base_url(url) for url in args.base_urls]
    prompts = load_prompts(args.prompt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_health_check:
        health_check(args.base_urls, args.timeout)

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for request_id in range(args.num_requests):
            base_url = args.base_urls[request_id % len(args.base_urls)]
            prompt = prompts[request_id % len(prompts)]
            futures.append(
                executor.submit(request_once, request_id, base_url, prompt, args)
            )
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
    elapsed = time.perf_counter() - started
    records.sort(key=lambda item: int(item["request_id"]))

    summary = summarize(args, records, elapsed)
    detail_path = output_dir / f"{args.name}.jsonl"
    summary_path = output_dir / f"{args.name}_summary.json"

    with detail_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"details: {detail_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
