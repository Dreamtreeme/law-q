from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import validate_dataset  # noqa: E402


class ValidateDatasetTest(unittest.TestCase):
    def make_dataset(self, root: Path, missing_document: bool = False) -> Path:
        records = []
        scoring = {1: "keyword_any", 2: "keyword_ratio", 3: "json_field", 4: "refusal"}
        for item_type in range(1, 5):
            document = root / f"doc-{item_type}.txt"
            if not missing_document or item_type != 4:
                document.write_text("문서 내용", encoding="utf-8")
            record = {
                "id": f"L00{item_type}",
                "type": item_type,
                "context_doc": document.name,
                "question": "질문",
                "answer_keywords": [] if item_type == 3 else ["정답"],
                "scoring": scoring[item_type],
            }
            if item_type == 3:
                record["answer_fields"] = {"필드": "값"}
            records.append(record)
        path = root / "questions.jsonl"
        path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_valid_dataset_produces_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_dataset(Path(temporary))
            report = validate_dataset.validate_artifacts(path, {1: 1, 2: 1, 3: 1, 4: 1})
            self.assertEqual(report["status"], "success")
            self.assertEqual(len(report["documents"]), 4)
            self.assertEqual(len(report["dataset"]["sha256"]), 64)

    def test_missing_document_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_dataset(Path(temporary), missing_document=True)
            report = validate_dataset.validate_artifacts(path, {1: 1, 2: 1, 3: 1, 4: 1})
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("참조 문서" in issue for issue in report["issues"]))

    def test_distribution_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_dataset(Path(temporary))
            report = validate_dataset.validate_artifacts(path, {1: 24, 2: 12, 3: 12, 4: 12})
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("유형 1" in issue for issue in report["issues"]))

    def test_lock_verification_detects_document_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_dataset(root)
            lock = root / "dataset.lock.json"
            report = validate_dataset.validate_artifacts(path, {1: 1, 2: 1, 3: 1, 4: 1})
            validate_dataset.write_json_atomic(lock, report)
            verified = validate_dataset.assert_dataset_lock(path, lock)
            self.assertEqual(verified["status"], "success")
            (root / "doc-2.txt").write_text("변경된 문서", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_dataset.assert_dataset_lock(path, lock)

    def test_lock_verification_detects_configured_distribution_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.make_dataset(root)
            lock = root / "dataset.lock.json"
            report = validate_dataset.validate_artifacts(path, {1: 1, 2: 1, 3: 1, 4: 1})
            validate_dataset.write_json_atomic(lock, report)
            with self.assertRaisesRegex(ValueError, "설정의 유형별 문항 수"):
                validate_dataset.assert_dataset_lock(
                    path, lock, {1: 24, 2: 12, 3: 12, 4: 12}
                )


if __name__ == "__main__":
    unittest.main()
