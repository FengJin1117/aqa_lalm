import re
from tqdm import tqdm
import json
import os
from datetime import datetime
from pathlib import Path

from prompts.aqa_prompt import AQAPrompt
from prompts.caption_aqa_prompt import CaptionAQAPrompt
from prompts.caption_only_prompt import CaptionOnlyPrompt

def extract_choice(pred):
    pred = pred.strip().upper()

    if "ASSISTANT" in pred:
        pred = pred.split("ASSISTANT")[-1]

    # 优先匹配 "ANSWER IS X"
    match = re.search(r"ANSWER\s+IS\s+([ABCD])", pred)
    if match:
        return match.group(1)

    matches = re.findall(r"\b([ABCD])\b", pred)
    if matches:
        return matches[-1]

    return None

def evaluate(model, dataset, output_path, prompt_type="aqa", max_samples=None):
    corr, total = 0, 0
    task_correct = {"sound": 0, "speech": 0, "music": 0}
    task_total = {"sound": 0, "speech": 0, "music": 0}
    results = []

    if max_samples:
        dataset = dataset[:max_samples]

    for i, sample in enumerate(tqdm(dataset, leave=True)):
        if max_samples and i >= max_samples:
            break

        if prompt_type == "aqa":
            prompt = AQAPrompt()
        elif prompt_type == "caption_aqa":
            prompt = CaptionAQAPrompt()
        elif prompt_type == "caption_only":
            prompt = CaptionOnlyPrompt()
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")
        
        if prompt_type == "caption_only":
            # 纯 caption 模式不使用音频路径
            conversation = prompt.build(
                question=sample["question"],
                choices=sample["choices"],
                caption=sample["caption"]   # 需要额外提供 caption 字段
            )
        else:
            conversation = prompt.build(
                question=sample["question"],
                choices=sample["choices"],
                audio_path=sample["audio_path"]
            )


        pred = model.infer(conversation, debug=False) 

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

        results.append({
            "id": sample.get("id"),
            "correct": int(is_correct),  # 放在第2位（0/1更方便统计）

            "question": sample.get("question"),
            "choices": sample.get("choices"),  # ✅ 新增

            "gt_answer": sample.get("answer"),
            "gt_choice": gt_choice,

            "model_raw": pred.strip(),
            "model_choice": pred_choice,
            "model_answer_text": model_answer_text,

            "task": sample.get("task"),
            "difficulty": sample.get("difficulty"),
            "sub-category": sample.get("sub-category", "")
        })

        if pred_choice == gt_choice:
            corr += 1
            task_correct[sample["task"]] += 1

        task_total[sample["task"]] += 1
        total += 1

    acc = corr / total if total > 0 else 0
    task_acc = {
        task: (task_correct[task] / task_total[task] if task_total[task] > 0 else 0)
        for task in task_correct
    }

    # 保存到 aqa_lalm/outputs，文件名前缀为时间戳
    # outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    # os.makedirs(outputs_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # out_path = os.path.join(outputs_dir, f"{timestamp}_results.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_path}")
    return acc, task_acc