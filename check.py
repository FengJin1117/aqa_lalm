from collections import Counter
import json
from typing import Tuple, Dict, Any, List

def summarize_gt_choice(
    path: str = "./data/mmau-test-mini-with-choice.json",
    key: str = "gt_choice",
    encoding: str = "utf-8",
    debug: bool = False,
) -> Tuple[Dict[Any, int], Dict[Any, float]]:
    """
    Load JSON list from `path`, compute counts and proportions of the given `key`.
    If an item does not contain `key`, its value will be None and counted as such.
    If debug=True, print IDs of items where the value is None.
    Returns (counts, proportions) and prints them.
    """
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)

    values = [item.get(key) for item in data]
    counts = Counter(values)
    proportions = {k: v / len(data) for k, v in counts.items()}

    if debug:
        none_ids = [item.get("id") for item in data if item.get(key) is None]
        print(f"IDs with None for key='{key}':", none_ids)

    print(f"Counts for key='{key}':", counts)
    return counts, proportions

if __name__ == "__main__":

    # 调用summarize_gt_choice
    print("GT Choice Distribution:")
    summarize_gt_choice("./data/mmau-test-mini-shuffled-with-choice.json")

    print("Qwen2.5 Omni 结果分布:")
    summarize_gt_choice("outputs/QwenOmni/20260516_012142_results.json", "model_choice", debug=False)

    # print("DeepSeek 结果分布:")
    # summarize_gt_choice("/data2/fwh/aqa_lalm/outputs/deepseek/20260516_013825_results.json", "model_choice", debug=False)