from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_models  # noqa: E402


class PrepareModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        for handler in prepare_models.LOGGER.handlers:
            handler.close()
        prepare_models.LOGGER.handlers.clear()
        prepare_models.LOGGER.addHandler(logging.NullHandler())

    def tearDown(self) -> None:
        for handler in prepare_models.LOGGER.handlers:
            handler.close()
        prepare_models.LOGGER.handlers.clear()

    def test_artifact_info_records_actual_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "model.gguf"
            artifact.write_bytes(b"a" * 1234)
            info = prepare_models.artifact_info(artifact)
        self.assertEqual(info["size_bytes"], 1234)

    def test_existing_f16_is_skipped_without_running_converter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "model-F16.gguf"
            output.write_bytes(b"complete")
            model = {"name": "test-model"}
            with patch.object(prepare_models, "run_command") as runner:
                result = prepare_models.convert_model(
                    model, root / "hf", output, root / "convert_hf_to_gguf.py"
                )
            runner.assert_not_called()
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["artifact"]["size_bytes"], 8)

    def test_matching_download_marker_enables_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}", encoding="utf-8")
            marker = {
                "hf_repo": "org/model",
                "requested_revision": "abc123",
                "resolved_revision": "abc123",
            }
            (root / prepare_models.DOWNLOAD_MARKER).write_text(
                json.dumps(marker), encoding="utf-8"
            )
            model = {
                "name": "test-model",
                "hf_repo": "org/model",
                "revision": "abc123",
            }
            result = prepare_models.download_model(model, root, {})
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["resolved_revision"], "abc123")

    def test_failed_stage_contains_model_stage_and_error_in_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "prepare.log"
            prepare_models.setup_logging(log_path, verbose=False)

            def fail() -> dict[str, object]:
                raise prepare_models.StageError("boom")

            result, failed = prepare_models.execute_stage("model-a", "convert_f16", fail)
            log_text = log_path.read_text(encoding="utf-8")
            for handler in prepare_models.LOGGER.handlers:
                handler.close()
            prepare_models.LOGGER.handlers.clear()
        self.assertTrue(failed)
        self.assertEqual(result["status"], "failed")
        self.assertIn("StageError: boom", result["traceback"])
        self.assertIn("[model-a][convert_f16]", log_text)
        self.assertIn("boom", log_text)

    def test_failure_is_persisted_to_jsonl_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "failures.jsonl"
            report: dict[str, object] = {"failures": []}
            stage_record = {
                "error_type": "StageError",
                "error": "download failed",
                "traceback": "trace details",
            }
            prepare_models.persist_failure(
                path, report, "model-a", "download", stage_record
            )
            persisted = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(persisted["model"], "model-a")
        self.assertEqual(persisted["stage"], "download")
        self.assertEqual(persisted["traceback"], "trace details")
        self.assertEqual(len(report["failures"]), 1)


if __name__ == "__main__":
    unittest.main()
