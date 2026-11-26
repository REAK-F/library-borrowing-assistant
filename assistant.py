"""上海工程技术大学图书馆借阅助手。

AI 主要编写：中文特征提取、TF-IDF 与余弦检索，以及 BorrowingGuide
中的路由、追问和低相似度转介。scikit-learn 提供向量化和相似度 API。
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import unicodedata

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
HAN_OR_WORD = re.compile(r"[\u3400-\u9fff]+|[a-z0-9]+")
OUTSIDE_TOPICS = (
    "预约座位", "座位预约", "预约座", "约座", "选座", "抢座", "研讨室", "自习室",
    "图书馆座位", "馆际互借", "电子书", "电子资源", "数据库", "知网", "校园卡挂失",
    "校园卡丢", "丢了校园卡", "校园卡补办", "重置密码", "找回密码", "忘记密码",
    "特藏", "开馆时间", "闭馆", "校外访问",
)
ADDITIONAL_OUTSIDE_TOPICS = ("座位", "打印", "复印", "扫描", "充电器", "笔记本电脑")
COLLOQUIAL_TERMS = (
    (r"自助机|自助终端|机器上", "自助借书机"),
    (r"多少本|多少册|几本|几册|借阅额度", "借阅册数"),
    (r"交回|交还|送回|放回|拿回馆|还回", "归还图书"),
    (r"假.{0,8}到期|到期.{0,8}假|开学以后|返校.{0,4}再交", "假期到期"),
    (r"往后延|宽限几天|还想读|延长借阅", "延长借期"),
    (r"延.{0,4}过.{0,4}[回次]|超过.{0,3}期限|过了.{0,3}期限|已经.{0,8}预约", "续借失败"),
    (r"排.{0,2}队|纸质书.{0,6}预约|书.{0,4}全部外借", "图书预约"),
    (r"弄没|不见了|弄丢|丢了|买一本一样", "遗失赔书"),
)


def normalize(text):
    return unicodedata.normalize("NFKC", text).lower()


def character_ngrams(text):
    """Use the same 2- and 3-character terms in both retrieval methods."""
    for phrase in HAN_OR_WORD.findall(normalize(text)):
        for size in (2, 3):
            for start in range(len(phrase) - size + 1):
                yield phrase[start:start + size]


def expand_query(text):
    text = normalize(text)
    terms = [keyword for pattern, keyword in COLLOQUIAL_TERMS if re.search(pattern, text)]
    if re.search(r"丢[了失书]|遗失|不见了", text) and re.search(
        r"哪里|哪儿|哪个|哪间|去哪|找谁|何处", text
    ):
        terms.append("遗失赔偿办理地点")
    return " ".join([text, *terms])


@dataclass(frozen=True)
class Match:
    rule_id: str
    score: float


@dataclass(frozen=True)
class Response:
    action: str
    rule_id: str = ""
    choice: int | None = None
    score: float = 0.0
    reason: str = ""


class BorrowingGuide:
    """由 AI 主要编写的检索与 Agent 决策核心。"""

    def __init__(self, data_path=ROOT / "data" / "rules.json", threshold=None, enhanced=True,
                 router=None, router_threshold=0.4):
        with data_path.open(encoding="utf-8") as document:
            self.data = json.load(document)
        self.rules = {rule["id"]: rule for rule in self.data["rules"]}
        self.rule_ids = list(self.rules)
        self.enhanced = enhanced
        self.router = router
        self.router_threshold = router_threshold
        threshold_key = "threshold" if enhanced else "initial_threshold"
        self.threshold = self.data[threshold_key] if threshold is None else threshold

        documents = [rule["title"] + " " + rule["terms"] for rule in self.rules.values()]
        self.indexes = {}
        for name, vectorizer in (
            ("keyword", CountVectorizer(analyzer=character_ngrams, binary=True)),
            ("tfidf", TfidfVectorizer(analyzer=character_ngrams, sublinear_tf=True)),
        ):
            self.indexes[name] = (vectorizer, vectorizer.fit_transform(documents))

    def rank(self, query, method="tfidf"):
        vectorizer, matrix = self.indexes[method]
        query_vector = vectorizer.transform([expand_query(query) if self.enhanced else query])
        scores = cosine_similarity(query_vector, matrix)[0]
        matches = [Match(rule_id, float(score)) for rule_id, score in zip(self.rule_ids, scores)]
        return sorted(matches, key=lambda match: (-match.score, match.rule_id))

    def _choice(self, slot, text, allow_number=False):
        text = normalize(text).strip()
        if allow_number and text in ("1", "2"):
            return int(text) - 1
        choices = self.data["slots"][slot]["choices"]
        matching = []
        for index, choice in enumerate(choices):
            aliases = choice["aliases"] + choice.get("enhanced_aliases", []) if self.enhanced else choice["aliases"]
            if any(alias in text for alias in aliases):
                matching.append(index)
        return matching[0] if len(matching) == 1 else None

    def start(self, query, method="tfidf"):
        query = normalize(query).strip()
        if not query or len(query) > 200:
            return Response("invalid", reason="请输入 1 到 200 个字符的问题。")
        topics = OUTSIDE_TOPICS + ADDITIONAL_OUTSIDE_TOPICS if self.enhanced else OUTSIDE_TOPICS
        if any(topic in query for topic in topics):
            return Response("refer", reason="这个问题超出了本工具的借阅指南范围。")

        if self.router is None:
            best = self.rank(query, method)[0]
            rule_id, score = best.rule_id, best.score
            if score < self.threshold:
                return Response("refer", score=score, reason="资料中没有找到足够匹配的借阅规则。")
        else:
            prediction = self.router.predict(query)
            rule_id, score = prediction.rule_id, prediction.probability
            if rule_id not in self.rules:
                return Response("refer", score=score,
                                reason="分类器认为这条问题不属于本工具的借阅范围。")
            if score < self.router_threshold:
                return Response("refer", score=score,
                                reason="分类器对这条问题的把握不足，先不套用借阅规则。")

        rule = self.rules[rule_id]
        slot = rule.get("slot")
        if slot is None:
            return Response("answer", rule_id, score=score)
        choice = self._choice(slot, query)
        if choice is None:
            return Response("ask", rule_id, score=score)
        return Response("answer", rule_id, choice, score)

    def follow_up(self, pending, reply):
        if pending.action != "ask":
            raise ValueError("Only a clarification request can receive a follow-up.")
        slot = self.rules[pending.rule_id]["slot"]
        choice = self._choice(slot, reply, allow_number=True)
        if choice is None:
            return pending
        return Response("answer", pending.rule_id, choice, pending.score)

    def render(self, response):
        if response.action == "invalid":
            return response.reason
        if response.action == "refer":
            return (
                response.reason + "\n建议通过图书馆官网或读者服务部核实。\n"
                + self.data["sources"]["contact"]["url"]
            )

        rule = self.rules[response.rule_id]
        if response.action == "ask":
            question = self.data["slots"][rule["slot"]]["question"]
            return f"需要确认 · {rule['title']}\n{question}"

        title = rule["title"]
        steps = list(rule["steps"])
        if response.choice is not None:
            title += " · " + self.data["slots"][rule["slot"]]["choices"][response.choice]["label"]
            steps.extend(rule["variants"][response.choice])
        lines = [title, *(f"{index}. {step}" for index, step in enumerate(steps, 1))]
        lines.append("官方出处")
        for key, locator in rule["sources"]:
            source = self.data["sources"][key]
            lines.append(f"- {source['title']} · {locator}")
            lines.append(f"  {source['url']}")
        return "\n".join(lines)


def run_session(guide):
    print("上海工程技术大学图书馆借阅助手")
    print("可问借书、还书、续借、纸质图书预约、图书遗失。输入 q 退出。")
    pending = None
    attempts = 0
    while True:
        try:
            query = input("\n你：" if pending is None else "\n补充：")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        command = query.strip().lower()
        if command in ("q", "quit", "退出"):
            return
        if command in ("重来", "取消"):
            pending = None
            attempts = 0
            print("已取消。可以输入新的问题。")
            continue
        if pending is None:
            response = guide.start(query)
        else:
            response = guide.follow_up(pending, query)
            attempts += 1
            # 两次补充都对不上选项时不再反复追问，改为给出官方核实入口。
            if response.action == "ask" and attempts >= 2:
                response = Response("refer", pending.rule_id, score=pending.score,
                                    reason="两次补充都没有对上选项，先按未确认处理。")
        pending = response if response.action == "ask" else None
        if pending is None:
            attempts = 0
        print(guide.render(response))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "rules.json",
                        help="借阅规则 JSON 的路径")
    parser.add_argument("--query", help="用一条问题运行助手")
    parser.add_argument("--reply", help="给需要追问的问题补充信息")
    parser.add_argument("--router", choices=("retrieval", "classifier"), default="retrieval",
                        help="规则路由方式：默认相似度检索，可选逻辑回归分类器")
    args = parser.parse_args()
    if args.reply is not None and args.query is None:
        parser.error("--reply 需要与 --query 一起使用。")
    try:
        router = None
        if args.router == "classifier":
            from classifier import build_router
            router = build_router(data_path=args.data)
        guide = BorrowingGuide(data_path=args.data, router=router, router_threshold=0.0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"资料库无法加载。请确认 data/rules.json 完整，或前往图书馆官网核实。{error}", file=sys.stderr)
        return 1

    if args.query is None:
        run_session(guide)
        return 0

    print("问题：" + args.query)
    response = guide.start(args.query)
    if response.action == "ask" and args.reply is not None:
        print(guide.render(response))
        print("\n补充：" + args.reply)
        response = guide.follow_up(response, args.reply)
    print(guide.render(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
