# scripts/filter_wrong.py

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/filter_wrong.py <input_json> <output_json>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wrong_samples = [
        item for item in data
        if item.get("correct", 0) == 0
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            wrong_samples,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved {len(wrong_samples)} wrong samples "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()