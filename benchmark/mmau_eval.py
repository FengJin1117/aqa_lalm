import re
from tqdm import tqdm
import json
import os
from datetime import datetime
from pathlib import Path

from prompts.aqa_prompt import AQAPrompt
from prompts.caption_aqa_prompt import CaptionAQAPrompt
from prompts.caption_only_prompt import CaptionOnlyPrompt

def extract_choice(pred, num_choices=4):
    if pred is None:
        return None

    pred = pred.strip()

    # 只取 assistant 后面的最终回答，避免匹配 prompt 里的 Choices
    m = re.search(r"assistant\s*\n?(.*)$", pred, flags=re.IGNORECASE | re.DOTALL)
    if m:
        pred = m.group(1).strip()

    valid_choices = "".join(chr(65 + i) for i in range(num_choices))
    pattern = f"[{valid_choices}]"

    # ANSWER IS X / ANSWER: X / FINAL ANSWER: X
    match = re.search(
        rf"(?:FINAL\s+ANSWER|ANSWER)\s*(?:IS|:)?\s*({pattern})\b",
        pred,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()

    # 明确选项格式：D. xxx / D: xxx / D) xxx
    match = re.search(rf"^\s*({pattern})\s*[\.\:\)]", pred, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 极简输出：只有 A/B/C/D/E
    match = re.fullmatch(rf"\s*({pattern})\s*", pred, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

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
        # elif prompt_type == "caption_aqa":
        #     prompt = CaptionAQAPrompt()
        elif prompt_type == "caption_qa":
            prompt = CaptionOnlyPrompt()
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")
        
        if prompt_type == "caption_qa":
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

        pred_choice = extract_choice(pred, num_choices=len(sample["choices"]))

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


def write_captions_jsonl(model, audio_paths, output_path, max_samples=None):
    """Generate captions and write one JSONL record immediately per audio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_samples:
        audio_paths = audio_paths[:max_samples]

    total = 0
    with output_path.open("w", encoding="utf-8") as f:
        for audio_path in tqdm(audio_paths, leave=True):
            caption = model.infer(str(audio_path)).strip()
            record = {
                "id": Path(audio_path).stem,
                "caption": caption,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            total += 1

    print(f"Saved {total} captions to {output_path}")
    return total

