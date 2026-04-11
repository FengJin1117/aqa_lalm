import time
import torch
import json
from pathlib import Path

from models.qwen_omni import QwenOmniAQA

AUDIO_ROOT = "/workspace/datasets/test-mini-audios"
JSON_PATH = "./mmau-test-mini.json"


def load_one_sample():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    return data[0]


def main():
    sample = load_one_sample()
    model = QwenOmniAQA()

    audio_path = str(Path(AUDIO_ROOT) / f"{sample['id']}.wav")

    question = sample["question"]
    choices = sample["choices"]

    # 拼接ABCD格式
    # option_str = "\n".join(
    #     [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
    # )
    # full_question = f"{question}\n{option_str}\nAnswer with A/B/C/D only."

    # print("==== 输入 ====")
    # print(full_question)

    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    pred = model.infer(audio_path, question, choices, test_mode=True)
    end = time.time()

    mem = torch.cuda.max_memory_allocated() / 1024**3

    print("\n==== 输出 ====")
    print(pred)

    print("\n==== GT ====")
    print(sample["answer"])

    print("\n==== 性能 ====")
    print(f"⏱ 推理时间: {end-start:.2f}s")
    print(f"💾 显存占用: {mem:.2f} GB")


if __name__ == "__main__":
    main()