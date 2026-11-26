"""生成报告用的两张结果图。

图 1：封存盲测（20 条）上三种方案的结果构成——正确、安全转介、错答。
图 2：内部核验集（24 条）与盲测集（20 条）的流程成功率对比。

读 results/ 下由 evaluate.py 与 evaluate_blind.py 生成的逐题记录，
换一次评估后可直接重画。
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

TEAL = "#12635f"
SLATE = "#476479"
SAND = "#c9a227"
CLAY = "#b4543f"
LIGHT = "#d9e4e9"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def outcome_profile(rows):
    correct = sum(row["correct"] == "1" for row in rows)
    referred = sum(row["correct"] != "1" and row["final_action"] == "refer" for row in rows)
    wrong = len(rows) - correct - referred
    return correct, referred, wrong


def figure_outcomes():
    arms = [
        ("等权关键词", "blind_v2_predictions_keyword.csv"),
        ("TF-IDF 检索", "blind_v2_predictions_tfidf.csv"),
        ("逻辑回归分类器", "blind_v2_predictions_classifier.csv"),
    ]
    labels = [name for name, _ in arms]
    profiles = [outcome_profile(load_csv(path)) for _, path in arms]
    total = sum(sum(profile) for profile in profiles) // len(profiles)

    figure, axis = plt.subplots(figsize=(7.4, 3.1))
    left = [0] * len(arms)
    for index, (color, title) in enumerate((
            (TEAL, "正确"), (SAND, "安全转介"), (CLAY, "错答或误追问"))):
        values = [profile[index] for profile in profiles]
        axis.barh(labels, values, left=left, color=color, label=title, height=0.52)
        for row, value in enumerate(values):
            if value:
                axis.text(left[row] + value / 2, row, str(value),
                          ha="center", va="center", color="white", fontsize=10)
        left = [left[row] + values[row] for row in range(len(values))]

    axis.set_xlim(0, total)
    axis.set_xlabel(f"封存盲测集共 {total} 条", fontsize=10)
    axis.invert_yaxis()
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis="y", length=0, labelsize=10)
    figure.tight_layout()
    figure.savefig(HERE / "图1_盲测结果构成.png", dpi=200)
    plt.close(figure)


def figure_gap():
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    blind = json.loads((RESULTS / "blind_v2_metrics.json").read_text(encoding="utf-8"))
    groups = ["等权关键词", "TF-IDF 检索", "逻辑回归分类器"]
    internal = [
        metrics["audit_keyword_workflow"]["workflow_correct"] / metrics["audit_keyword_workflow"]["cases"],
        metrics["audit"]["workflow_correct"] / metrics["audit"]["cases"],
        None,
    ]
    external = [
        blind["keyword"]["workflow_correct"] / blind["keyword"]["cases"],
        blind["tfidf"]["workflow_correct"] / blind["tfidf"]["cases"],
        blind["classifier"]["workflow_correct"] / blind["classifier"]["cases"],
    ]

    figure, axis = plt.subplots(figsize=(7.4, 3.1))
    width = 0.34
    positions = range(len(groups))
    internal_values = [0 if value is None else value * 100 for value in internal]
    axis.bar([p - width / 2 for p in positions], internal_values, width,
             color=SLATE, label="内部核验（24 条）")
    axis.bar([p + width / 2 for p in positions], [value * 100 for value in external], width,
             color=TEAL, label="封存盲测（20 条）")
    for index, value in enumerate(internal_values):
        if value:
            axis.text(index - width / 2, value + 1.5, f"{value:.0f}%", ha="center", fontsize=9)
    for index, value in enumerate(external):
        axis.text(index + width / 2, value * 100 + 1.5, f"{value * 100:.0f}%",
                  ha="center", fontsize=9)
    axis.set_xticks(list(positions))
    axis.set_xticklabels(groups, fontsize=10)
    axis.set_ylim(0, 108)
    axis.set_ylabel("流程成功率（%）", fontsize=10)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(axis="x", length=0)
    figure.tight_layout()
    figure.savefig(HERE / "图2_核验与盲测对比.png", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    figure_outcomes()
    figure_gap()
    print("图片已生成：", HERE / "图1_盲测结果构成.png", HERE / "图2_核验与盲测对比.png")
