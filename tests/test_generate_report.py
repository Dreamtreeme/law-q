from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_report  # noqa: E402


def row(model: str, quantization: str, score: float, vram: int) -> dict[str, str]:
    value = str(score)
    return {
        "model": model,
        "quantization": quantization,
        "status": "success",
        "overall_score": value,
        "type1_score": value,
        "type2_score": value,
        "type3_score": value,
        "type4_score": value,
        "max_vram_used_mib": str(vram),
        "ttft_mean_seconds": "0.1",
        "pp_2048_mean_tps": "1000",
        "tg_2048_mean_tps": "50",
    }


class GenerateReportTest(unittest.TestCase):
    def test_f16_is_preferred_as_baseline(self) -> None:
        rows = [
            row("a", "F16", 0.9, 9000),
            row("a", "Q8_0", 0.85, 7000),
            row("a", "Q4_K_M", 0.8, 5000),
        ]
        baselines = generate_report.choose_baselines(rows)
        self.assertEqual(baselines["a"]["quantization"], "F16")

    def test_q8_is_fallback_baseline(self) -> None:
        rows = [row("a", "Q8_0", 0.9, 7000), row("a", "Q4_K_M", 0.8, 5000)]
        baselines = generate_report.choose_baselines(rows)
        self.assertEqual(baselines["a"]["quantization"], "Q8_0")

    def test_unscored_f16_falls_back_to_scored_q8(self) -> None:
        failed_f16 = row("a", "F16", 0.9, 9000)
        for field in [*generate_report.SCORE_FIELDS, "overall_score"]:
            failed_f16[field] = ""
        failed_f16["status"] = "failed"
        rows = [failed_f16, row("a", "Q8_0", 0.85, 7000)]
        baselines = generate_report.choose_baselines(rows)
        self.assertEqual(baselines["a"]["quantization"], "Q8_0")

    def test_quantization_loss_is_percentage_point_drop(self) -> None:
        rows = [row("a", "Q8_0", 0.9, 7000), row("a", "Q4_K_M", 0.8, 5000)]
        aggregate, details, excluded = generate_report.quantization_losses(rows)
        q4 = next(item for item in aggregate if item["quantization"] == "Q4_K_M")
        self.assertAlmostEqual(q4["type1_score_drop_pp"], 10.0)
        self.assertEqual(excluded, [])
        self.assertEqual(len(details), 2)

    def test_vram_groups_respect_tolerance_from_group_minimum(self) -> None:
        rows = [
            row("a", "Q4", 0.8, 5000),
            row("b", "Q4", 0.9, 5400),
            row("c", "Q5", 0.85, 5700),
        ]
        groups, missing = generate_report.group_similar_vram(rows, 512)
        memberships = [(item["model"], item["group"]) for item in groups]
        self.assertEqual(dict(memberships)["a"], dict(memberships)["b"])
        self.assertNotEqual(dict(memberships)["a"], dict(memberships)["c"])
        self.assertEqual(missing, [])

    def test_generate_report_writes_tables_json_and_svg_graphs(self) -> None:
        rows = [row("a", "Q8_0", 0.9, 7000), row("a", "Q4_K_M", 0.8, 5000)]
        for item in rows:
            item["gguf_size_bytes"] = str(5 * 1024**3)
            for length in (512, 4096):
                item[f"pp_{length}_mean_tps"] = "1000"
                item[f"tg_{length}_mean_tps"] = "50"
        config = {
            "benchmark": {"prompt_lengths": [512, 2048, 4096]},
            "report": {
                "speed_context_length": 2048,
                "similar_vram_tolerance_mib": 512,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "results.csv"
            output_dir = root / "report"
            generate_report.write_csv(input_path, rows)
            result = generate_report.generate_report(input_path, output_dir, config)

            self.assertTrue(result["report"].is_file())
            for filename in (
                "analysis.json",
                "combination-summary.csv",
                "quantization-loss.csv",
                "quantization-loss-detail.csv",
                "memory-budget-groups.csv",
                "accuracy-vs-speed.svg",
                "type-scores-by-quantization.svg",
            ):
                self.assertGreater((output_dir / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
