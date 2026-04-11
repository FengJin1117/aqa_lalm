import json
from tqdm import tqdm
from pathlib import Path
import time  # 新增

from models.qwen_omni import QwenOmniAQA
from eval.mmau_eval import evaluate

AUDIO_ROOT = "/workspace/datasets/test-mini-audios"
JSON_PATH = "./mmau-test-mini.json"


def load_local_dataset():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    dataset = []
    for sample in data:
        audio_path = str(Path(AUDIO_ROOT) / f"{sample['id']}.wav")

        dataset.append({
            "id": sample["id"],
            "audio_path": audio_path,
            "question": sample["question"],
            "choices": sample["choices"],
            "answer": sample["answer"],
            "task": sample["task"],
            "difficulty": sample["difficulty"],
            "sub-category": sample.get("sub-category", "")
        })

    return dataset


def main():
    model = QwenOmniAQA()
    dataset = load_local_dataset()

    # ⏱ 开始计时
    start_time = time.perf_counter()
    acc = evaluate(model, dataset, max_samples=1000)
    # ⏱ 结束计时
    end_time = time.perf_counter()

    elapsed_minutes = (end_time - start_time) / 60

    print(f"\n✅ MMAU Test-mini Accuracy: {acc:.4f}")
    print(f"⏱ 推理耗时: {elapsed_minutes:.2f} 分钟")


if __name__ == "__main__":
    main()