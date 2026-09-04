from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from scoring import score_dataset, score_item  # noqa: E402


def item(item_id: str, item_type: int, keywords: list[str], **extra: object) -> dict[str, object]:
    scoring = {
        1: "keyword_any",
        2: "keyword_ratio",
        3: "json_field",
        4: "refusal",
    }[item_type]
    return {
        "id": item_id,
        "type": item_type,
        "context_doc": "doc.txt",
        "question": "question",
        "answer_keywords": keywords,
        "scoring": scoring,
        **extra,
    }


class ScoringTest(unittest.TestCase):
    def test_keyword_any(self) -> None:
        result = score_item(item("L001", 1, ["12시간", "열두 시간"]), "주 12 시간입니다.")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["matched_keywords"], ["12시간"])

    def test_keyword_ratio_partial_credit(self) -> None:
        result = score_item(item("L002", 2, ["서면 통지", "30일 전"]), "서면 통지했습니다.")
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["matched_count"], 1)

    def test_json_field_partial_credit_and_fence(self) -> None:
        evaluation = item(
            "L003",
            3,
            [],
            answer_fields={"사건번호": "2026가1", "당사자.원고": "김민수"},
        )
        response = '```json\n{"사건번호":"2026가1","당사자":{"원고":"이민수"}}\n```'
        result = score_item(evaluation, response)
        self.assertEqual(result["score"], 0.5)
        self.assertFalse(result["json_parse_failed"])

    def test_json_parse_failure_is_separate(self) -> None:
        evaluation = item("L004", 3, [], answer_fields={"사건번호": "2026가1"})
        result = score_item(evaluation, "JSON이 아닙니다")
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(result["json_parse_failed"])
        self.assertEqual(result["reason"], "json_parse_failed")

    def test_refusal(self) -> None:
        result = score_item(
            item("L005", 4, ["확인할 수 없습니다", "알 수 없습니다"]),
            "제공된 자료에서는 확인할 수 없습니다.",
        )
        self.assertEqual(result["score"], 1.0)

    def test_type_and_overall_aggregation(self) -> None:
        dataset = [
            item("L001", 1, ["정답"]),
            item("L002", 2, ["A", "B"]),
            item("L003", 3, [], answer_fields={"field": "value"}),
            item("L004", 4, ["알 수 없습니다"]),
        ]
        predictions = {
            "L001": "정답",
            "L002": "A",
            "L003": "not json",
            "EXTRA": "ignored",
        }
        report = score_dataset(dataset, predictions)
        self.assertEqual(report["summary"]["total_score"], 1.5)
        self.assertEqual(report["summary"]["average_score"], 0.375)
        self.assertEqual(report["summary"]["json_parse_failures"], 1)
        self.assertEqual(report["summary"]["missing_predictions"], 1)
        self.assertEqual(report["summary"]["extra_predictions"], 1)
        self.assertEqual(report["by_type"]["2"]["average_score"], 0.5)
        self.assertEqual(report["by_type"]["4"]["average_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
