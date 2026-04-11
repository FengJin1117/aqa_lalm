import re
from tqdm import tqdm
import json
import os
from datetime import datetime
from pathlib import Path

def extract_choice(pred):
    pred = pred.strip().upper()

    # 优先匹配 assistant 最后的回答
    if "ASSISTANT" in pred:
        pred = pred.split("ASSISTANT")[-1]

    matches = re.findall(r"\b([ABCD])\b", pred)
    if matches:
        return matches[-1]

    return None

def evaluate(model, dataset, max_samples=None):
    corr, total = 0, 0
    results = []

    if max_samples:
        dataset = dataset[:max_samples]

    for i, sample in enumerate(tqdm(dataset)):
        if max_samples and i >= max_samples:
            break

        pred = model.infer(
            sample["audio_path"],
            sample["question"],
            sample["choices"]
        )

        pred_choice = extract_choice(pred)

        gt_idx = sample["choices"].index(sample["answer"])
        gt_choice = chr(65 + gt_idx)

        # 将 ABCD 映射回原始选项文本（若不能解析 ABCD 则保留模型原始回答）
        model_answer_text = None
        if pred_choice:
            idx = ord(pred_choice) - 65
            if 0 <= idx < len(sample["choices"]):
                model_answer_text = sample["choices"][idx]
            else:
                model_answer_text = pred.strip()
        else:
            model_answer_text = pred.strip()

        is_correct = (pred_choice == gt_choice)

        # 记录结果条目
        results.append({
            "id": sample.get("id"),
            "correct": int(is_correct),  # 放在第2位（0/1更方便统计）

            "question": sample.get("question"),
            "choices": sample.get("choices"),  # ✅ 新增

            "gt_choice": gt_choice,
            "gt_answer": sample.get("answer"),
            
            "model_raw": pred.strip(),
            "model_choice": pred_choice,
            "model_answer_text": model_answer_text
        })

        if pred_choice == gt_choice:
            corr += 1

        total += 1

    acc = corr / total if total > 0 else 0

    # 保存到 aqa_lalm/outputs，文件名前缀为时间戳
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(outputs_dir, f"{timestamp}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_path}")
    return acc