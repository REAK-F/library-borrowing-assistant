import unittest
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from sklearn.feature_extraction.text import TfidfVectorizer

from assistant import BorrowingGuide, character_ngrams


class BorrowingGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guide = BorrowingGuide()

    def test_character_terms_respect_punctuation(self):
        self.assertEqual(list(character_ngrams("续借，ＡＢ")), ["续借", "ab"])

    def test_tfidf_tool_prefers_rare_terms_and_handles_unknown_text(self):
        vectorizer = TfidfVectorizer(analyzer=character_ngrams, sublinear_tf=True)
        vectorizer.fit(["图书 续借", "图书 预约", "图书 预约"])
        weights = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        self.assertGreater(weights["续借"], weights["预约"])
        self.assertGreater(weights["预约"], weights["图书"])
        self.assertEqual(self.guide.rank("🪐")[0].score, 0)

    def test_renewal_answer_has_official_source(self):
        response = self.guide.start("怎么续借")
        self.assertEqual((response.action, response.rule_id), ("answer", "N1"))
        rendered = self.guide.render(response)
        self.assertIn("https://lib.sues.edu.cn/", rendered)
        self.assertIn("未被其他读者预约", rendered)

    def test_clarification_only_uses_matching_campus(self):
        pending = self.guide.start("图书丢了去哪个柜台")
        self.assertEqual((pending.action, pending.rule_id), ("ask", "L2"))
        self.assertEqual(self.guide.follow_up(pending, "两边不知道"), pending)
        answer = self.guide.follow_up(pending, "２")
        text = self.guide.render(answer)
        self.assertEqual((answer.action, answer.choice), ("answer", 1))
        self.assertIn("329 室", text)
        self.assertNotIn("152 室", text)

    def test_seat_reservation_does_not_become_a_book_reservation(self):
        self.assertEqual(self.guide.start("怎么预约靠窗的座位").action, "refer")

    def test_blank_or_overlong_query_receives_input_guidance(self):
        self.assertEqual(self.guide.start("　 ").action, "invalid")
        self.assertEqual(self.guide.start("书" * 201).action, "invalid")

    def test_missing_knowledge_file_prompts_official_verification(self):
        script = Path(__file__).resolve().parents[1] / "assistant.py"
        with TemporaryDirectory() as target:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(script), "--data", str(Path(target) / "missing.json"),
                 "--query", "怎么续借"],
                capture_output=True, text=True, encoding="utf-8", timeout=15,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("前往图书馆官网核实", result.stderr)

    def test_classifier_router_answers_and_rejects_out_of_scope(self):
        from classifier import build_router

        guide = BorrowingGuide(router=build_router(), router_threshold=0.0)
        answer = guide.start("怎么续借")
        self.assertEqual((answer.action, answer.rule_id), ("answer", "N1"))
        self.assertEqual(guide.start("图书馆里有热水吗").action, "refer")

    def test_blind_set_still_matches_its_sealed_manifest(self):
        import hashlib
        import json

        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "data" / "blind_manifest_v2.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256((root / "data" / "blind_cases_v2.csv").read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["sha256"])
