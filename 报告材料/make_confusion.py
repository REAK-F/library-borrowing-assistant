"""绘制分类器的交叉验证混淆矩阵（只用开发集与测试集，不涉及封存盲测）。"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import BorrowingGuide
from classifier import OUT_LABEL, RuleClassifier
from evaluate import load_cases

HERE = Path(__file__).resolve().parent
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    rows = [row for row in load_cases("dev") + load_cases("test")
            if row["action"] in {"answer", "ask", "refer"}]
    guide = BorrowingGuide()
    labels = [row["rule_id"] or OUT_LABEL for row in rows]
    queries = [row["query"] for row in rows]
    classes = [rule["id"] for rule in guide.data["rules"]] + [OUT_LABEL]
    index = {label: position for position, label in enumerate(classes)}

    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    folds = min(5, min(Counter(labels).values()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    terms = [(phrase, rule["id"]) for rule in guide.data["rules"] for phrase in rule["terms"].split()]
    for train_index, test_index in splitter.split(queries, labels):
        model = RuleClassifier().fit(
            [queries[i] for i in train_index] + [phrase for phrase, _ in terms],
            [labels[i] for i in train_index] + [label for _, label in terms])
        for i in test_index:
            matrix[index[labels[i]], index[model.predict(queries[i]).rule_id]] += 1

    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    shown = np.where(matrix == 0, np.nan, matrix)
    axis.imshow(shown, cmap="YlGnBu", vmin=0, vmax=matrix.max())
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = matrix[row, column]
            if value:
                axis.text(column, row, str(value), ha="center", va="center", fontsize=9,
                          color="white" if value >= matrix.max() * 0.6 else "#17283a")
    axis.set_xticks(range(len(classes)))
    axis.set_yticks(range(len(classes)))
    axis.set_xticklabels(classes, fontsize=9)
    axis.set_yticklabels(classes, fontsize=9)
    axis.set_xlabel("预测类别", fontsize=10)
    axis.set_ylabel("真实类别", fontsize=10)
    axis.set_title(f"逻辑回归路由的 {folds} 折交叉验证混淆矩阵（{len(rows)} 条问法）",
                   fontsize=10, pad=10)
    figure.tight_layout()
    figure.savefig(HERE / "图3_混淆矩阵.png", dpi=200)
    plt.close(figure)
    print("accuracy:", (matrix.trace() / matrix.sum()).round(4), "saved", HERE / "图3_混淆矩阵.png")


if __name__ == "__main__":
    main()
