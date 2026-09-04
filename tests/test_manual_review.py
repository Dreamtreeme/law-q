from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import manual_review  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


class ManualReviewTest(unittest.TestCase):
    def make_run(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "eval" / "questions.jsonl"
        write_jsonl(
            dataset,
            [
                {
                    "id": "L002",
                    "type": 2,
                    "context_doc": "doc.txt",
                    "question": "조건을 적용하라",
                    "answer_keywords": ["요건"],
                    "scoring": "keyword_ratio",
                }
            ],
        )
        (dataset.parent / "doc.txt").write_text("가상 문서", encoding="utf-8")
        results = root / "run" / "results.csv"
        results.parent.mkdir()
        with results.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "quantization", "status", "overall_score"],
            )
            writer.writeheader()
            for model, score in (("m1", "0.8"), ("m2", "0.9"), ("m3", "0.7")):
                writer.writerow(
                    {
                        "model": model,
                        "quantization": "Q4",
                        "status": "success",
                        "overall_score": score,
                    }
                )
                combo = results.parent / "evaluations" / model / "Q4"
                write_jsonl(combo / "responses.jsonl", [{"id": "L002", "status": "success", "response": model}])
                write_json(
                    combo / "scores.json",
                    {"items": [{"id": "L002", "score": score, "matched_keywords": ["요건"]}]},
                )
        return results, dataset

    def test_export_is_blind_and_keeps_auto_score_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, dataset = self.make_run(root)
            exported = manual_review.export_review(results, dataset, root / "review", top_k=3)
            rows = manual_review.read_csv(exported["review"])
            key = json.loads(exported["key"].read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["candidate_alias"] for row in rows}, {"C01", "C02", "C03"})
        self.assertTrue(all(row["human_score_0_to_2"] == "" for row in rows))
        self.assertNotIn("model", rows[0])
        self.assertEqual(set(key["aliases"]), {"C01", "C02", "C03"})

    def test_summary_requires_complete_review_and_unblinds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, dataset = self.make_run(root)
            exported = manual_review.export_review(results, dataset, root / "review", top_k=3)
            rows = manual_review.read_csv(exported["review"])
            for row in rows:
                row["human_score_0_to_2"] = "2"
                row["logical_error_yes_no"] = "no"
            manual_review.write_csv(exported["review"], manual_review.REVIEW_FIELDS, rows)
            summarized = manual_review.summarize_review(
                exported["review"], exported["key"], root / "summary"
            )
            summary_rows = manual_review.read_csv(summarized["csv"])
        self.assertEqual(len(summary_rows), 3)
        self.assertEqual({row["model"] for row in summary_rows}, {"m1", "m2", "m3"})
        self.assertTrue(all(row["human_score_percent"] == "100.0" for row in summary_rows))

    def test_summary_rejects_blank_human_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, dataset = self.make_run(root)
            exported = manual_review.export_review(results, dataset, root / "review", top_k=3)
            with self.assertRaises(manual_review.ManualReviewError):
                manual_review.summarize_review(exported["review"], exported["key"], root / "summary")

    def test_export_rejects_zero_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results, dataset = self.make_run(root)
            with self.assertRaisesRegex(manual_review.ManualReviewError, "1 이상"):
                manual_review.export_review(results, dataset, root / "review", top_k=0)


if __name__ == "__main__":
    unittest.main()
