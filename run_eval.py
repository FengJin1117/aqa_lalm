'''
总结这个文件的功能和结构：
- 这个文件是评测脚本，主要功能是加载数据集、运行评测，并输出结果。
- 主要函数：
  - `load_local_dataset()`: 从本地 JSON 文件加载数据集，并构建包含音频路径的样本列表。
  - `run_evaluation(model, dataset, model_name="QwenOmni", max_samples=None)`: 统一的评测函数，接受模型和数据集，运行评测，并输出准确率和每个 task 的准确率，同时记录推理耗时。结果保存到指定路径。
  - `main()`: 代码模式的入口，默认使用 QwenOmniAQA 模型，并展示每个 task 的准确率。
  - `cli()`: CLI 模式的入口，允许用户选择模型和最大评测样本数。
'''

import os
import json
from tqdm import tqdm
from pathlib import Path
import time
from datetime import datetime
import logging
logging.getLogger().setLevel(logging.ERROR)

from models.qwen_omni import QwenOmni
from models.qwen2_audio import Qwen2AudioAQA
from models.deepseek_text import DeepSeekText

from eval.mmau_eval import evaluate
import argparse

# MMAU 数据集路径配置
AUDIO_ROOT = "../datasets/test-mini-audios"
JSON_PATH = "./data/mmau-test-mini.json"

# 这里是可以动态改变的（这里是qwen3-captioner标注的）
# CAPTION_PATH = "./data/mmau-test-mini-captions.jsonl"
CAPTION_PATH = "/data2/fwh/project_bagpiper/outputs/bagpiper_caption.jsonl"

def load_captions():
    """加载 caption jsonl -> dict[id] = caption"""
    captions = {}
    with open(CAPTION_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            captions[item["id"]] = item["caption"]
    return captions


def load_local_dataset(use_caption=False):
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    captions = load_captions() if use_caption else {}

    dataset = []
    for sample in data:
        audio_path = str(Path(AUDIO_ROOT) / f"{sample['id']}.wav")

        item = {
            "id": sample["id"],
            "audio_path": audio_path,  # 保留（兼容旧模型）
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": sample["answer"],
            "task": sample["task"],
            "difficulty": sample["difficulty"],
            "sub-category": sample.get("sub-category", "")
        }

        if use_caption:
            item["caption"] = captions.get(sample["id"], "")

        dataset.append(item)

    return dataset


def run_evaluation(model, dataset, model_name="QwenOmni", prompt_type="aqa", max_samples=None):
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

    print(f"\n✅ MMAU Accuracy: {acc*100:.2f}")

    if task_acc is not None:
        print("\n📊 每个Task的准确率:")
        for task, t_acc in task_acc.items():
            print(f"  - {task}: {t_acc*100:.2f}")

    print(f"⏱ 推理耗时: {elapsed_minutes:.2f} 分钟")

    return acc, task_acc


def cli():
    parser = argparse.ArgumentParser(description="Run evaluation with selected model.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["QwenOmni", "Qwen2Audio", "deepseek"],
        default="QwenOmni"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["aqa", "caption_aqa", "caption_only"],
        default="aqa"
    )

    args = parser.parse_args()

    
    # 🔥 关键：是否启用 caption
    use_caption = args.prompt_type in ["caption_only"]

    # 加载数据集
    dataset = load_local_dataset(use_caption=use_caption)

    if args.model == "QwenOmni":
        model = QwenOmni()
    elif args.model == "Qwen2Audio":
        model = Qwen2AudioAQA()
    elif args.model == "deepseek":
        if args.prompt_type != "caption_only":
            raise ValueError(
                "deepseek only supports caption_only mode"
            )

        model = DeepSeekText()
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    run_evaluation(
        model,
        dataset,
        model_name=args.model,
        prompt_type=args.prompt_type,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    # CLI 模式
    cli()

    # 代码模式
    # main()