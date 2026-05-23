import os
import json
from pathlib import Path
import time
from datetime import datetime
import logging
import signal
import subprocess
import urllib.error
import urllib.request
logging.getLogger().setLevel(logging.ERROR)

# AQA Audio模型
from models.aqa.qwen_omni import QwenOmniAQA
from models.aqa.qwen2_audio import Qwen2AudioAQA

# Text问答模型
from models.text.qwen3_vllm import Qwen3VLLMText

from benchmark.mmau_eval import evaluate
import argparse


AUDIO_ROOT = "../datasets/test-mini-audios"

# JSON_PATH = "./data/mmau-test-mini.json"

JSON_PATH = "./data/mmau-test-mini-shuffled-with-choice.json"
QWEN3_VLLM_MODEL = "Qwen3-4B-Instruct-2507"

def load_captions(caption_path):
    captions = {}
    with open(caption_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            captions[item["id"]] = item.get("caption", "")
    return captions


def load_local_dataset(benchmark_mode="aqa", caption_path=None):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    use_caption = benchmark_mode == "caption_qa"
    captions = load_captions(caption_path) if use_caption else {}

    dataset = []
    for sample in data:
        audio_path = str(Path(AUDIO_ROOT) / f"{sample['id']}.wav")

        item = {
            "id": sample["id"],
            "audio_path": audio_path,
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": sample["answer"],
            "task": sample["task"],
            "difficulty": sample["difficulty"],
            "sub-category": sample.get("sub-category", ""),
        }

        if use_caption:
            item["caption"] = captions.get(sample["id"], "")

        dataset.append(item)

    return dataset


def get_json(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def post_json(url, payload, timeout=120.0):
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


def is_vllm_healthy(base_url):
    try:
        status, _ = get_json(f"{base_url.rstrip('/')}/models", timeout=5.0)
        return 200 <= status < 300
    except Exception:
        return False


def probe_vllm_chat(base_url, model_name):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with the letter A only."}],
        "temperature": 0.0,
        "max_tokens": 8,
    }
    try:
        status, response = post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            payload,
            timeout=120.0,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM probe failed: HTTP {exc.code}: {body}") from exc

    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "").strip()
    if not (200 <= status < 300) or not content:
        raise RuntimeError(f"vLLM probe returned empty response: {response}")


def start_qwen3_vllm(args, base_url):
    if is_vllm_healthy(base_url):
        print(f"Reusing existing vLLM service at {base_url}")
        probe_vllm_chat(base_url, QWEN3_VLLM_MODEL)
        return None, None

    log_dir = Path("exp/vllm_parallel/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_eval_qwen3_gpu{args.vllm_gpu}_port{args.vllm_port}_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")

    cmd = ["bash", "vllm.sh", "serve-one", str(args.vllm_gpu), str(args.vllm_port)]
    print(f"Starting vLLM: {' '.join(cmd)}")
    print(f"vLLM log: {log_path}")
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()

    deadline = time.time() + args.vllm_start_timeout
    last_probe_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited before becoming healthy. Check log: {log_path}"
            )
        if is_vllm_healthy(base_url):
            try:
                probe_vllm_chat(base_url, QWEN3_VLLM_MODEL)
                print("vLLM health check and probe succeeded.")
                return process, log_path
            except Exception as exc:
                last_probe_error = exc
        time.sleep(5)

    stop_process(process)
    detail = f" Last probe error: {last_probe_error}" if last_probe_error else ""
    raise RuntimeError(f"Timed out waiting for vLLM.{detail} Check log: {log_path}")


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def build_model(benchmark_mode, audio_model, text_model, vllm_base_url=None):
    if benchmark_mode == "caption_qa":
        if text_model == "deepseek":
            from models.text.deepseek import DeepSeekText
            return DeepSeekText(), "deepseek"
        if text_model == "QwenOmni":
            from models.text.qwen_omni import QwenOmni
            return QwenOmni(), "QwenOmni"
        if text_model == QWEN3_VLLM_MODEL:
            return Qwen3VLLMText(base_url=vllm_base_url, model_name=QWEN3_VLLM_MODEL), QWEN3_VLLM_MODEL

    # aqa模型
    if audio_model == "QwenOmni":
        return QwenOmniAQA(), "QwenOmni"
    if audio_model == "Qwen2Audio":
        return Qwen2AudioAQA(), "Qwen2Audio"

    raise ValueError(f"Unsupported model: audio_model={audio_model}, text_model={text_model}")


def run_evaluation(model, dataset, model_name, prompt_type, max_samples=None):
    outputs_dir = os.path.join("outputs", model_name)
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(outputs_dir, f"{timestamp}_results.json")

    start_time = time.perf_counter()
    result = evaluate(
        model,
        dataset,
        output_path,
        prompt_type=prompt_type,
        max_samples=max_samples,
        answer_model=model_name,
    )
    end_time = time.perf_counter()

    elapsed_minutes = (end_time - start_time) / 60

    if isinstance(result, tuple):
        acc, task_acc = result
    else:
        acc, task_acc = result, None

    print(f"\nMMAU Accuracy: {acc * 100:.2f}")

    if task_acc is not None:
        print("\n每个Task的准确率:")
        for task, t_acc in task_acc.items():
            print(f"  - {task}: {t_acc * 100:.2f}")

    print(f"推理耗时: {elapsed_minutes:.2f} 分钟")

    return acc, task_acc


def run_benchmark(args):
    if args.benchmark_mode == "caption_qa" and args.caption_path is None:
        raise ValueError("--caption_path is required when --benchmark_mode caption_qa")

    vllm_process = None
    try:
        dataset = load_local_dataset(
            benchmark_mode=args.benchmark_mode,
            caption_path=args.caption_path,
        )

        vllm_base_url = args.vllm_base_url or f"http://127.0.0.1:{args.vllm_port}/v1"
        if args.benchmark_mode == "caption_qa" and args.text_model == QWEN3_VLLM_MODEL:
            vllm_process, _ = start_qwen3_vllm(args, vllm_base_url)

        model, model_name = build_model(
            benchmark_mode=args.benchmark_mode,
            audio_model=args.audio_model,
            text_model=args.text_model,
            vllm_base_url=vllm_base_url,
        )

        # benchmark_mode 同时决定 prompt 类型：
        # - aqa: 基于音频回答
        # - caption_qa: 基于 caption 文本回答
        run_evaluation(
            model,
            dataset,
            model_name=model_name,
            prompt_type=args.benchmark_mode,
            max_samples=args.max_samples,
        )
    finally:
        if vllm_process is not None:
            print("Stopping vLLM service started by run_eval.py")
            stop_process(vllm_process)


def cli():
    parser = argparse.ArgumentParser(description="Run MMAU benchmark in audio QA or caption QA mode.")

    parser.add_argument(
        "--benchmark_mode",
        type=str,
        choices=["aqa", "caption_qa"],
        default="aqa",
        help="aqa uses audio input; caption_qa uses caption JSONL input.",
    )
    parser.add_argument(
        "--audio_model",
        type=str,
        choices=["QwenOmni", "Qwen2Audio"],
        default="QwenOmni",
        help="Audio model used only when --benchmark_mode aqa.",
    )

    parser.add_argument(
        "--text_model",
        type=str,
        choices=["deepseek", "QwenOmni", QWEN3_VLLM_MODEL],
        default=QWEN3_VLLM_MODEL,
        help="Text model used only when --benchmark_mode `caption_qa`.",
    )
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument(
        "--caption_path",
        type=str,
        default=None,
        help="Path to caption JSONL. Required when --benchmark_mode caption_qa.",
    )
    parser.add_argument("--vllm_gpu", type=int, default=2)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible vLLM base URL. Defaults to http://127.0.0.1:<vllm_port>/v1.",
    )
    parser.add_argument("--vllm_start_timeout", type=int, default=600)

    args = parser.parse_args()

    if args.benchmark_mode == "aqa":
        print(f"Running MMAU benchmark in AQA mode with audio model: {args.audio_model}")
    else:
        print(f"Running MMAU benchmark in Caption QA mode with text model: {args.text_model}")
        print(f"Using caption file: {args.caption_path}")

    run_benchmark(args)


if __name__ == "__main__":
    cli()
