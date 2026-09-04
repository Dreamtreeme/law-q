from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_benchmarks  # noqa: E402


def config() -> dict:
    return {
        "runtime": {"gpu_layers": 37, "threads": 8},
        "benchmark": {
            "prompt_lengths": [512, 2048, 4096],
            "generation_tokens": 128,
            "repetitions": 3,
            "batch_size": 2048,
            "ubatch_size": 512,
        },
    }


def raw_row(n_prompt: int, n_gen: int, n_depth: int) -> dict:
    return {
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "n_depth": n_depth,
        "samples_ts": [10.0, 20.0, 30.0],
        "samples_ns": [100000000, 50000000, 33333333],
        "avg_ts": 20.0,
        "stddev_ts": 10.0,
        "backends": "CUDA",
        "gpu_info": "RTX 3080",
        "n_gpu_layers": 37,
    }


class RunBenchmarksTest(unittest.TestCase):
    def test_repetitions_must_be_at_least_three(self) -> None:
        value = config()
        value["benchmark"]["repetitions"] = 2
        with self.assertRaises(run_benchmarks.BenchmarkError):
            run_benchmarks.validate_benchmark_config(value)

    def test_prompt_command_uses_all_lengths_and_builtin_warmup(self) -> None:
        command = run_benchmarks.build_command(
            Path("llama-bench.exe"), Path("model.gguf"), "prompt_processing", config()
        )
        self.assertEqual(command[command.index("-p") + 1], "512,2048,4096")
        self.assertEqual(command[command.index("-n") + 1], "0")
        self.assertEqual(command[command.index("-r") + 1], "3")
        self.assertNotIn("--no-warmup", command)

    def test_generation_command_uses_context_depth(self) -> None:
        command = run_benchmarks.build_command(
            Path("llama-bench.exe"), Path("model.gguf"), "token_generation", config()
        )
        self.assertEqual(command[command.index("-p") + 1], "0")
        self.assertEqual(command[command.index("-n") + 1], "128")
        self.assertEqual(command[command.index("-d") + 1], "512,2048,4096")
        self.assertEqual(command[command.index("-ngl") + 1], "37")

    def test_normalizes_mean_stddev_and_samples(self) -> None:
        rows = [raw_row(length, 0, 0) for length in (512, 2048, 4096)]
        normalized = run_benchmarks.normalize_rows(
            rows,
            "prompt_processing",
            "model-a",
            "Q4_K_M",
            [512, 2048, 4096],
            3,
        )
        self.assertEqual(len(normalized), 3)
        self.assertEqual(normalized[0]["tokens_per_second"]["mean"], 20.0)
        self.assertEqual(normalized[0]["tokens_per_second"]["stddev"], 10.0)
        self.assertEqual(len(normalized[0]["latency_ms"]["samples"]), 3)

    def test_generation_rows_are_keyed_by_prompt_depth(self) -> None:
        rows = [raw_row(0, 128, length) for length in (512, 2048, 4096)]
        normalized = run_benchmarks.normalize_rows(
            rows,
            "token_generation",
            "model-a",
            "Q4_K_M",
            [512, 2048, 4096],
            3,
        )
        self.assertEqual([item["prompt_length"] for item in normalized], [512, 2048, 4096])
        self.assertTrue(all(item["generated_tokens"] == 128 for item in normalized))


if __name__ == "__main__":
    unittest.main()
