import os
import json
from pathlib import Path
import time
from datetime import datetime
import logging
logging.getLogger().setLevel(logging.ERROR)

# AQA Audio模型
from models.aqa.qwen_omni import QwenOmniAQA
from models.aqa.qwen2_audio import Qwen2AudioAQA

# Text问答模型
from models.text.qwen_omni import QwenOmni
from models.text.deepseek import DeepSeekText

from benchmark.mmau_eval import evaluate
import argparse


AUDIO_ROOT = "../datasets/test-mini-audios"

# JSON_PATH = "./data/mmau-test-mini.json"

JSON_PATH = "./data/mmau-test-mini-shuffled-with-choice.json"

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


def build_model(benchmark_mode, audio_model, text_model):
    if benchmark_mode == "caption_qa":
        if text_model == "deepseek":
            return DeepSeekText(), "deepseek"
        if text_model == "QwenOmni":
            return QwenOmni(), "QwenOmni"

    # aqa模型
    if audio_model == "QwenOmni":
        return QwenOmniAQA(), "QwenOmni"
    if audio_model == "Qwen2Audio":
        return Qwen2AudioAQA(), "Qwen2Audio"

    raise ValueError(f"Unsupported audio model: {audio_model}")


def run_evaluation(model, dataset, model_name, prompt_type, max_samples=None):
    outputs_dir = os.path.join("outputs", model_name)
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(outputs_dir, f"{timestamp}_results.json")

    start_time = time.perf_counter()
    result = evaluate(model, dataset, output_path, prompt_type=prompt_type, max_samples=max_samples)
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

    dataset = load_local_dataset(
        benchmark_mode=args.benchmark_mode,
        caption_path=args.caption_path,
    )

    model, model_name = build_model(
        benchmark_mode=args.benchmark_mode,
        audio_model=args.audio_model,
        text_model=args.text_model,
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
        choices=["deepseek", "QwenOmni"],
        default="deepseek",
        help="Text model used only when --benchmark_mode `caption_qa`.",
    )
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument(
        "--caption_path",
        type=str,
        default=None,
        help="Path to caption JSONL. Required when --benchmark_mode caption_qa.",
    )

    args = parser.parse_args()

    if args.benchmark_mode == "aqa":
        print(f"Running MMAU benchmark in AQA mode with audio model: {args.audio_model}")
    else:
        print(f"Running MMAU benchmark in Caption QA mode with text model: {args.text_model}")
        print(f"Using caption file: {args.caption_path}")

    run_benchmark(args)


if __name__ == "__main__":
    cli()
