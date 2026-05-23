#!/usr/bin/env python3
"""Generate a Markdown report for the vLLM parallel benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXP_NAME = "Qwen3-4B-Instruct-2507的vllm部署性能测试"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SERVED_MODEL = "Qwen3-4B-Instruct-2507"
RESULT_DIR = Path("exp/vllm_parallel/results")
REPORT_PATH = Path("exp/vllm_parallel/report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create vLLM benchmark report.")
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    parser.add_argument("--output", default=str(REPORT_PATH))
    return parser.parse_args()


def load_summaries(result_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(result_dir.glob("*_summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        item["_path"] = str(path)
        summaries.append(item)
    return summaries


def seconds(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def throughput(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def table_row(item: dict[str, Any], speedup_base: float | None = None) -> str:
    gpu_count = len(item.get("base_urls", []))
    speedup = "-"
    current_throughput = item.get("throughput_req_s")
    if speedup_base and current_throughput is not None:
        speedup = f"{float(current_throughput) / speedup_base:.2f}x"
    return (
        f"| {item.get('name', '-')} "
        f"| {gpu_count} "
        f"| {item.get('concurrency', '-')} "
        f"| {item.get('num_requests', '-')} "
        f"| {percent(item.get('success_rate'))} "
        f"| {seconds(item.get('latency_avg_s'))} "
        f"| {seconds(item.get('latency_p95_s'))} "
        f"| {throughput(current_throughput)} "
        f"| {speedup} |"
    )


def render_table(items: list[dict[str, Any]], speedup_base: float | None = None) -> list[str]:
    lines = [
        "| run | instances | concurrency | requests | success | avg latency(s) | p95(s) | throughput(req/s) | speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if not items:
        lines.append("| pending | - | - | - | - | - | - | - | - |")
    else:
        lines.extend(table_row(item, speedup_base) for item in items)
    return lines


def section_items(summaries: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [item for item in summaries if str(item.get("name", "")).startswith(prefix)]


def scale_items(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = section_items(summaries, "scale_")
    return sorted(items, key=lambda item: len(item.get("base_urls", [])))


def render_report(summaries: list[dict[str, Any]]) -> str:
    serial = section_items(summaries, "serial_")
    concurrent = section_items(summaries, "concurrent_")
    scale = scale_items(summaries)
    base_throughput = None
    for item in scale:
        if len(item.get("base_urls", [])) == 1:
            base_throughput = item.get("throughput_req_s")
            break

    lines = [
        f"# {EXP_NAME}",
        "",
        "## 实验设置",
        "",
        f"- 模型：`{MODEL}`",
        f"- served model name：`{SERVED_MODEL}`",
        "- conda 环境：`vllm`",
        "- GPU：4, 5, 6, 7",
        "- 多卡口径：多实例部署，每张 GPU 一个单卡 vLLM 服务，客户端 round-robin 分发请求。",
        "- 请求接口：OpenAI-compatible `/v1/chat/completions`",
        "",
        "## 启动命令",
        "",
        "```bash",
        "bash vllm.sh serve-one 4 8000",
        "bash vllm.sh serve-many 4",
        "```",
        "",
        "## 测试命令",
        "",
        "```bash",
        "bash vllm.sh bench-serial",
        "bash vllm.sh bench-concurrent",
        "bash vllm.sh bench-scale",
        "bash vllm.sh report",
        "```",
        "",
        "## 串行测试",
        "",
        *render_table(serial),
        "",
        "## 并发测试",
        "",
        *render_table(concurrent),
        "",
        "## 多实例扩容测试",
        "",
        *render_table(scale, base_throughput),
        "",
        "## 结论",
        "",
    ]
    if summaries:
        total_requests = sum(int(item.get("num_requests") or 0) for item in summaries)
        total_success = sum(int(item.get("success_count") or 0) for item in summaries)
        overall_success = total_success / total_requests if total_requests else 0.0
        lines.append(
            f"本次共完成 {total_requests} 个请求，成功 {total_success} 个，整体成功率 {percent(overall_success)}。"
        )
        if serial:
            first_serial = serial[0]
            lines.append(
                f"串行单卡平均回复时间为 {seconds(first_serial.get('latency_avg_s'))}s，p95 为 {seconds(first_serial.get('latency_p95_s'))}s。"
            )
        if concurrent:
            best_concurrent = max(
                concurrent, key=lambda item: float(item.get("throughput_req_s") or 0.0)
            )
            lines.append(
                f"单卡并发测试中，并发 {best_concurrent.get('concurrency')} 的吞吐最高，为 {throughput(best_concurrent.get('throughput_req_s'))} req/s。"
            )
        if scale:
            best = max(scale, key=lambda item: float(item.get("throughput_req_s") or 0.0))
            one_gpu = next(
                (item for item in scale if len(item.get("base_urls", [])) == 1),
                None,
            )
            lines.append(
                f"多实例扩容测试中，最高吞吐来自 `{best.get('name')}`，吞吐为 {throughput(best.get('throughput_req_s'))} req/s。"
            )
            if one_gpu and best.get("throughput_req_s") is not None:
                gain = float(best.get("throughput_req_s")) / float(
                    one_gpu.get("throughput_req_s")
                )
                lines.append(
                    f"在固定最多 4 并发的条件下，4 实例相对 1 实例吞吐提升约 {gain:.2f}x，主要改善是平均延迟和 p95 小幅下降；该负载没有把多实例能力充分压满。"
                )
    else:
        lines.append(
            "尚未发现 summary JSON。启动 vLLM 服务并运行测试命令后，重新执行 `bash vllm.sh report` 会填充表格。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(result_dir) if result_dir.exists() else []
    output.write_text(render_report(summaries), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
