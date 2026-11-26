"""规则路由的判别式分类器：字符 n-gram TF-IDF + 逻辑回归。

与检索方案共用同一套 2/3 字符片段特征，区别在于用带标签的样本学习
决策边界，并给出可比较的类别概率。训练样本由两部分组成：开发集与测试集
里的人工问法，以及规则表中已有的检索短语（弱标注）。"不属于借阅范围"
也作为一类学出来，用来替代手调相似度阈值。

AI 辅助编写，供 evaluate_blind.py 和 assistant.py 的 --router classifier 使用。
"""

import csv
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from assistant import character_ngrams, normalize


OUT_LABEL = "__OUT__"
DEFAULT_C = 0.3


@dataclass(frozen=True)
class Prediction:
    rule_id: str
    probability: float


class RuleClassifier:
    """把一条问法映射到规则编号，并返回最大类别概率。"""

    def __init__(self, C=DEFAULT_C):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer=character_ngrams, sublinear_tf=True),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    def fit(self, queries, rule_ids):
        self.model.fit([normalize(query) for query in queries], list(rule_ids))
        return self

    def predict(self, query):
        probabilities = self.model.predict_proba([normalize(query)])[0]
        index = int(np.argmax(probabilities))
        return Prediction(str(self.model.classes_[index]), float(probabilities[index]))

    def predict_proba_matrix(self, queries):
        return self.model.predict_proba([normalize(query) for query in queries])


def build_router(data_path=None, cases_path=None, C=DEFAULT_C):
    """用问法标注 + 规则短语弱标注训练一个可直接接进 Agent 的路由器。"""
    from assistant import BorrowingGuide, ROOT

    guide = BorrowingGuide(data_path or ROOT / "data" / "rules.json")
    source = cases_path or ROOT / "data" / "cases.csv"
    samples = []
    with source.open(newline="", encoding="utf-8") as rows:
        for row in csv.DictReader(rows):
            if row["action"] in {"answer", "ask", "refer"}:
                samples.append((row["query"], row["rule_id"] or OUT_LABEL))
    for rule in guide.data["rules"]:
        for phrase in rule["terms"].split():
            samples.append((phrase, rule["id"]))
    return RuleClassifier(C=C).fit([query for query, _ in samples],
                                   [label for _, label in samples])
