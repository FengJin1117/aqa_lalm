import argparse
import json
from pathlib import Path
from collections import defaultdict
import html


def compute_metrics(data):
    total = len(data)
    correct = sum(x.get("correct", 0) for x in data)

    task_stats = defaultdict(lambda: [0, 0])
    subcat_stats = defaultdict(lambda: [0, 0])

    errors = []

    for item in data:
        c = item.get("correct", 0)
        task = item.get("task", "unknown")
        subcat = item.get("sub-category", "unknown")

        task_stats[task][1] += 1
        subcat_stats[subcat][1] += 1

        if c == 1:
            task_stats[task][0] += 1
            subcat_stats[subcat][0] += 1
        else:
            errors.append({
                "id": item.get("id"),
                "question": item.get("question"),
                "choices": item.get("choices"),
                "gt_answer": item.get("gt_answer"),
                "model_answer": item.get("model_answer_text"),
                "task": task,
                "subcat": subcat
            })

    return {
        "total": total,
        "accuracy": correct / total if total else 0,
        "task_stats": task_stats,
        "subcat_stats": subcat_stats,
        "errors": errors
    }


def format_stats(stats_dict):
    result = []
    for k, (corr, total) in stats_dict.items():
        acc = corr / total if total else 0
        result.append({
            "name": k,
            "acc": acc,
            "total": total
        })
    return sorted(result, key=lambda x: -x["acc"])


def generate_html(report, output_path):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evaluation Report</title>
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

h1 {{
    font-weight: 600;
}}

.metric {{
    font-size: 18px;
    margin: 10px 0;
}}

.table {{
    width: 100%;
    border-collapse: collapse;
}}

.table th, .table td {{
    padding: 10px;
    border-bottom: 1px solid #eee;
    text-align: left;
}}

.bad {{
    color: #d9534f;
    font-weight: bold;
}}

.good {{
    color: #5cb85c;
}}

.small {{
    font-size: 12px;
    color: #888;
}}
</style>
</head>

<body>

<h1>📊 Model Evaluation Report</h1>

<div class="card">
<h2>Overall</h2>
<div class="metric">Total Samples: {report["total"]}</div>
<div class="metric">Accuracy: <b>{report["accuracy"]*100:.2f}%</b></div>
</div>

<div class="card">
<h2>Task-wise Accuracy</h2>
<table class="table">
<tr><th>Task</th><th>Accuracy</th><th>Samples</th></tr>
"""

    for x in report["task_stats"]:
        html_content += f"<tr><td>{html.escape(x['name'])}</td><td>{x['acc']*100:.2f}%</td><td>{x['total']}</td></tr>"

    html_content += "</table></div>"

    html_content += """
<div class="card">
<h2>Sub-category Accuracy</h2>
<table class="table">
<tr><th>Category</th><th>Accuracy</th><th>Samples</th></tr>
"""

    for x in report["subcat_stats"]:
        html_content += f"<tr><td>{html.escape(x['name'])}</td><td>{x['acc']*100:.2f}%</td><td>{x['total']}</td></tr>"

    html_content += "</table></div>"

    html_content += """
<div class="card">
<h2>❌ Error Cases</h2>
<table class="table">
<tr>
<th>Question</th>
<th>Choices</th>
<th>GT</th>
<th>Model</th>
<th>Task</th>
<th>SubCat</th>
</tr>
"""

    for e in report["errors"][:200]:
        choices = "<br>".join(html.escape(c) for c in (e["choices"] or []))

        html_content += f"""
<tr>
<td>{html.escape(e["question"] or "")}</td>
<td class="small">{choices}</td>
<td class="good">{html.escape(str(e["gt_answer"]))}</td>
<td class="bad">{html.escape(str(e["model_answer"]))}</td>
<td>{html.escape(e["task"])}</td>
<td>{html.escape(e["subcat"])}</td>
</tr>
"""

    html_content += "</table></div></body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = input_path.parent / "results_analysis"

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = compute_metrics(data)

    report["task_stats"] = format_stats(report["task_stats"])
    report["subcat_stats"] = format_stats(report["subcat_stats"])

    html_path = out_dir / "report.html"
    generate_html(report, html_path)

    print(f"✅ Report generated: {html_path}")


if __name__ == "__main__":
    main()