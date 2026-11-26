"""在封存的 20 条盲测集上比较三种规则路由方案。

三种方案共用同一套流程（输入检查、范围过滤、槽位追问、出处渲染），
只替换"问题 → 规则编号"这一步：

1. 等权关键词：CountVectorizer 二值特征 + 余弦相似度；
2. TF-IDF：TfidfVectorizer + 余弦相似度；
3. 逻辑回归：同样的字符 n-gram 特征，用带标签问法和规则表中的检索短语
   训练，并给出可用于转介判断的类别概率。

训练侧的规则短语相当于弱标注：它们固定留在训练集内，交叉验证只在人工
问法上划分，避免同一短语同时出现在训练折和验证折。

盲测集先封存（data/blind_manifest.json 记录 sha256），运行前校验哈希。
本次运行结果只用于报告，不据此回改规则、口语映射或阈值。
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

from assistant import BorrowingGuide, ROOT
from classifier import OUT_LABEL, RuleClassifier
from evaluate import load_cases, run_cases, summarize, write_predictions


BLIND_PATH = ROOT / "data" / "blind_cases.csv"
MANIFEST_PATH = ROOT / "data" / "blind_manifest.json"
C_GRID = (0.3, 1.0, 3.0)
TAU_GRID = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)
PRECISION_TARGET = 0.9
COVERAGE_FLOOR = 0.5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_blind(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def training_rows():
    rows = load_cases("dev") + load_cases("test")
    return [row for row in rows if row["action"] in {"answer", "ask", "refer"}]


def weak_samples(rules):
    """规则表里的检索短语作为弱标注样本。"""
    samples = []
    for rule in rules:
        for phrase in rule["terms"].split():
            samples.append((phrase, rule["id"]))
    return samples


def labels_of(rows):
    return [row["rule_id"] or OUT_LABEL for row in rows]


def out_of_fold(rows, terms, C):
    """只在人工问法上做分层交叉验证，规则短语始终留在训练侧。"""
    labels = np.array(labels_of(rows))
    queries = [row["query"] for row in rows]
    folds = min(5, min(Counter(labels).values()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    predicted = np.empty(len(rows), dtype=object)
    confidence = np.zeros(len(rows))
    term_queries = [phrase for phrase, _ in terms]
    term_labels = [label for _, label in terms]
    for train_index, test_index in splitter.split(queries, labels):
        model = RuleClassifier(C=C).fit(
            [queries[index] for index in train_index] + term_queries,
            [labels[index] for index in train_index] + term_labels)
        for index in test_index:
            prediction = model.predict(queries[index])
            predicted[index] = prediction.rule_id
            confidence[index] = prediction.probability
    return float((predicted == labels).mean()), confidence, predicted, labels, folds


def selective_threshold(confidence, predicted, labels):
    """按折外风险-覆盖率选弃权线：优先保证已答部分的精度。"""
    candidates = []
    for tau in TAU_GRID:
        kept = confidence >= tau
        coverage = float(kept.mean())
        accuracy = float((predicted[kept] == labels[kept]).mean()) if kept.any() else 0.0
        candidates.append({"tau": tau, "coverage": round(coverage, 3),
                           "accuracy": round(accuracy, 3)})
    acceptable = [row for row in candidates
                  if row["accuracy"] >= PRECISION_TARGET and row["coverage"] >= COVERAGE_FLOOR]
    chosen = max(acceptable, key=lambda row: row["tau"]) if acceptable else max(
        candidates, key=lambda row: (row["accuracy"], row["tau"]))
    return chosen["tau"], candidates


def calibration_table(confidence, predicted, labels):
    rows = []
    for low, high in ((0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        mask = (confidence >= low) & (confidence < high)
        rows.append({
            "range": f"{low:.1f}–{min(high, 1.0):.1f}",
            "cases": int(mask.sum()),
            "correct": int((predicted[mask] == labels[mask]).sum()),
        })
    return rows


def write_metrics(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--blind", type=Path, default=BLIND_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--tag", default="blind")
    args = parser.parse_args()
    args.blind = args.blind.resolve()
    args.manifest = args.manifest.resolve()
    args.out = args.out.resolve()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if digest(args.blind) != manifest["sha256"]:
        raise SystemExit("盲测集哈希与封存记录不一致，拒绝运行。")

    blind = load_blind(args.blind)
    if len(blind) != manifest["cases"]:
        raise SystemExit("盲测集条数与封存记录不一致。")
    rows = training_rows()
    if any(row["id"] in {case["id"] for case in blind} for row in rows):
        raise SystemExit("盲测集与训练集存在重复编号。")

    guide = BorrowingGuide()
    terms = weak_samples(guide.data["rules"])
    folds_data = {}
    scores = {}
    for C in C_GRID:
        accuracy, confidence, predicted, labels, folds = out_of_fold(rows, terms, C)
        scores[C] = accuracy
        folds_data[C] = (confidence, predicted, labels)
    best_C = max(C_GRID, key=lambda C: (scores[C], -C))
    tau, screening = selective_threshold(*folds_data[best_C])
    table = calibration_table(*folds_data[best_C])

    classifier = RuleClassifier(C=best_C).fit(
        [row["query"] for row in rows] + [phrase for phrase, _ in terms],
        labels_of(rows) + [label for _, label in terms])

    metrics_path = ROOT / "results" / "metrics.json"
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))

    keyword_guide = BorrowingGuide()
    keyword_guide.threshold = stored["keyword_threshold"]
    classifier_guide = BorrowingGuide(router=classifier, router_threshold=tau)

    arms = {
        "keyword": run_cases(keyword_guide, blind, "keyword"),
        "tfidf": run_cases(BorrowingGuide(), blind),
        "classifier": run_cases(classifier_guide, blind),
    }
    for name, arm_rows in arms.items():
        write_predictions(args.out / f"{args.tag}_predictions_{name}.csv", arm_rows)

    output = {
        "blind_set": {
            "file": args.blind.relative_to(ROOT).as_posix(),
            "sha256": manifest["sha256"],
            "cases": len(blind),
            "created": manifest["created"],
            "manifest": args.manifest.relative_to(ROOT).as_posix(),
        },
        "training": {
            "questions": len(rows),
            "phrases": len(terms),
            "splits": ["dev", "test"],
            "folds": folds,
            "c_grid": list(C_GRID),
            "cv_accuracy": {str(C): round(scores[C], 4) for C in C_GRID},
            "chosen_C": best_C,
            "tau": tau,
            "screening": screening,
            "calibration": table,
        },
        "thresholds": {"keyword": stored["keyword_threshold"], "tfidf": stored["threshold"]},
        "keyword": summarize(arms["keyword"]),
        "tfidf": summarize(arms["tfidf"]),
        "classifier": summarize(arms["classifier"]),
        "python": sys.version.split()[0],
        "scikit_learn": version("scikit-learn"),
    }
    write_metrics(args.out / f"{args.tag}_metrics.json", output)

    print(f'训练：{len(rows)} 条问法 + {len(terms)} 条规则短语，{folds} 折交叉验证')
    print(f'交叉验证准确率：{output["training"]["cv_accuracy"]}，选定 C={best_C}，弃权线 tau={tau}')
    for name in ("keyword", "tfidf", "classifier"):
        summary = output[name]
        print(f'{name:10s} 流程 {summary["workflow_correct"]}/{summary["cases"]} '
              f'| 正确转介 {summary["refer_correct"]}/{summary["refer_cases"]} '
              f'| 追问 {summary["followup_correct"]}/{summary["followup_cases"]}')
    print(json.dumps(output["training"]["calibration"], ensure_ascii=False))


if __name__ == "__main__":
    main()
