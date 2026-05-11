import argparse
import json
from pathlib import Path
from collections import defaultdict
import html


def compute_stats(data):
    total = len(data)
    correct = sum(x.get("correct", 0) for x in data)

    task_stats = defaultdict(lambda: [0, 0])
    subcat_stats = defaultdict(lambda: [0, 0])

    for item in data:
        c = item.get("correct", 0)
        task = item.get("task", "unknown")
        subcat = item.get("sub-category", "unknown")

        task_stats[task][1] += 1
        subcat_stats[subcat][1] += 1

        if c == 1:
            task_stats[task][0] += 1
            subcat_stats[subcat][0] += 1

    def format_stats(stats_dict):
        result = {}
        for k, (corr, total) in stats_dict.items():
            acc = corr / total if total else 0
            result[k] = acc
        return result

    return {
        "total": total,
        "accuracy": correct / total if total else 0,
        "task": format_stats(task_stats),
        "subcat": format_stats(subcat_stats)
    }


def merge_keys(dict1, dict2):
    keys = set(dict1.keys()) | set(dict2.keys())
    result = []
    for k in keys:
        a = dict1.get(k, 0)
        b = dict2.get(k, 0)
        result.append({
            "name": k,
            "a": a,
            "b": b,
            "delta": b - a
        })
    return result


def bar(acc):
    width = int(acc * 100)
    return f"""
    <div style="background:#eee; width:100px; height:10px; display:inline-block;">
        <div style="background:#4caf50; width:{width}px; height:10px;"></div>
    </div>
    """


def delta_str(d):
    color = "green" if d >= 0 else "red"
    sign = "+" if d >= 0 else ""
    return f'<span style="color:{color}">{sign}{d*100:.2f}%</span>'


def generate_html(name_a, name_b, stats_a, stats_b, output_path):
    task_data = sorted(
        merge_keys(stats_a["task"], stats_b["task"]),
        key=lambda x: -x["delta"]
    )

    subcat_data = sorted(
        merge_keys(stats_a["subcat"], stats_b["subcat"]),
        key=lambda x: -x["delta"]
    )

    top_improve = [x for x in subcat_data if x["delta"] > 0][:10]

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Comparison Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
    background: #f5f7fa;
    padding: 30px;
}}

.card {{
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}

.table {{
    width: 100%;
    border-collapse: collapse;
}}

.table th, .table td {{
    padding: 8px;
    border-bottom: 1px solid #eee;
}}

.a {{ color: #1f77b4; }}
.b {{ color: #d62728; }}
</style>
</head>

<body>

<h1>📊 Benchmark Comparison</h1>

<div class="card">
<h2>Overall</h2>
<p>{name_a}: <b>{stats_a["accuracy"]*100:.2f}%</b></p>
<p>{name_b}: <b>{stats_b["accuracy"]*100:.2f}%</b></p>
<p>Delta: {delta_str(stats_b["accuracy"] - stats_a["accuracy"])}</p>
</div>

<div class="card">
<h2>🚀 Top Improvements (Sub-category)</h2>
<ul>
"""

    for x in top_improve:
        html_content += f"<li>{html.escape(x['name'])}: {delta_str(x['delta'])}</li>"

    html_content += "</ul></div>"

    html_content += """
<div class="card">
<h2>Task Comparison (sorted by improvement)</h2>
<table class="table">
<tr><th>Task</th><th>A</th><th>B</th><th>Δ</th></tr>
"""

    for x in task_data:
        html_content += f"""
<tr>
<td>{html.escape(x['name'])}</td>
<td class="a">{x['a']*100:.2f}% {bar(x['a'])}</td>
<td class="b">{x['b']*100:.2f}% {bar(x['b'])}</td>
<td>{delta_str(x['delta'])}</td>
</tr>
"""

    html_content += "</table></div>"

    html_content += """
<div class="card">
<h2>Sub-category Comparison (sorted by improvement)</h2>
<table class="table">
<tr><th>Category</th><th>A</th><th>B</th><th>Δ</th></tr>
"""

    for x in subcat_data:
        html_content += f"""
<tr>
<td>{html.escape(x['name'])}</td>
<td class="a">{x['a']*100:.2f}% {bar(x['a'])}</td>
<td class="b">{x['b']*100:.2f}% {bar(x['b'])}</td>
<td>{delta_str(x['delta'])}</td>
</tr>
"""

    html_content += "</table></div></body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", required=True)
    parser.add_argument("-b", required=True)
    parser.add_argument("-o", default="compare_results")

    # ✅ 新增：实验名
    parser.add_argument("--name_a", default="Exp-A")
    parser.add_argument("--name_b", default="Exp-B")

    args = parser.parse_args()

    with open(args.a, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    with open(args.b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    stats_a = compute_stats(data_a)
    stats_b = compute_stats(data_b)

    out_dir = Path(args.o)
    out_dir.mkdir(exist_ok=True)

    output_html = out_dir / "compare.html"

    generate_html(
        args.name_a,
        args.name_b,
        stats_a,
        stats_b,
        output_html
    )

    print(f"✅ Compare report: {output_html}")


if __name__ == "__main__":
    main()