from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_pipeline  # noqa: E402


class RunPipelineTest(unittest.TestCase):
    def test_only_success_rows_are_complete_for_resume(self) -> None:
        rows = [
            {"model": "a", "quantization": "Q4", "status": "success"},
            {"model": "b", "quantization": "Q5", "status": "failed"},
            {"model": "c", "quantization": "Q8", "status": "partial_failure"},
        ]
        self.assertEqual(run_pipeline.completed_keys(rows), {("a", "Q4")})

    def test_csv_is_immediately_readable_after_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            fields = ["model", "quantization", "status"]
            rows = [{"model": "a", "quantization": "Q4", "status": "success"}]
            run_pipeline.write_csv_atomic(path, rows, fields)
            loaded = run_pipeline.read_csv(path)
        self.assertEqual(loaded, rows)

    def test_dynamic_columns_cover_every_context_length(self) -> None:
        fields = run_pipeline.result_fields([512, 2048, 4096])
        for length in (512, 2048, 4096):
            self.assertIn(f"pp_{length}_mean_tps", fields)
            self.assertIn(f"tg_{length}_stddev_tps", fields)

    def test_ordered_rows_follow_configuration_order(self) -> None:
        combinations = [
            {"model": "a", "quantization": "Q4"},
            {"model": "b", "quantization": "Q5"},
        ]
        rows = {
            ("b", "Q5"): {"model": "b", "quantization": "Q5"},
            ("a", "Q4"): {"model": "a", "quantization": "Q4"},
        }
        ordered = run_pipeline.ordered_rows(rows, combinations)
        self.assertEqual([row["model"] for row in ordered], ["a", "b"])

    def test_only_filters_pilot_combinations_in_config_order(self) -> None:
        combinations = [
            {"model": "a", "quantization": "Q4"},
            {"model": "a", "quantization": "Q8"},
            {"model": "b", "quantization": "Q4"},
        ]
        selected = run_pipeline.select_combinations(
            combinations, ["a:Q8", "a:Q4"]
        )
        self.assertEqual(
            [(item["model"], item["quantization"]) for item in selected],
            [("a", "Q4"), ("a", "Q8")],
        )

    def test_only_rejects_unknown_combination(self) -> None:
        with self.assertRaisesRegex(ValueError, "설정에 없는"):
            run_pipeline.select_combinations(
                [{"model": "a", "quantization": "Q4"}], ["a:Q8"]
            )

    def test_resume_rejects_changed_config_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "experiment.yaml"
            snapshot = root / "run" / "experiment.yaml"
            snapshot.parent.mkdir()
            config_path.write_text("version: 2\n", encoding="utf-8")
            snapshot.write_text("version: 1\n", encoding="utf-8")
            config = {"_config_path": str(config_path)}
            with self.assertRaisesRegex(ValueError, "설정 스냅샷이 다릅니다"):
                run_pipeline.assert_resume_config_matches(config, snapshot.parent)


if __name__ == "__main__":
    unittest.main()
