from __future__ import annotations

import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_evaluation  # noqa: E402


class FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        yield from self.lines


class RunEvaluationTest(unittest.TestCase):
    def test_stream_records_ttft_total_and_full_response(self) -> None:
        response = FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"안녕"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"하세요"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        ticks = iter([10.0, 10.25, 10.8])
        with patch.object(run_evaluation.requests, "post", return_value=response):
            result = run_evaluation.stream_chat_completion(
                "http://localhost/v1/chat/completions",
                {"stream": True},
                timeout=10,
                clock=lambda: next(ticks),
            )
        self.assertEqual(result["response"], "안녕하세요")
        self.assertEqual(result["ttft_seconds"], 0.25)
        self.assertEqual(result["total_response_seconds"], 0.8)

    def test_vram_csv_parser(self) -> None:
        records = run_evaluation.parse_vram_csv(
            "0, NVIDIA GeForce RTX 3080, 9123, 10240", "2026-01-01T00:00:00Z"
        )
        self.assertEqual(records[0]["memory_used_mib"], 9123)
        self.assertEqual(records[0]["memory_total_mib"], 10240)

    def test_context_path_cannot_escape_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "questions.jsonl"
            dataset.touch()
            with self.assertRaises(run_evaluation.EvaluationRunError):
                run_evaluation.safe_context_path(dataset, "../secret.txt")

    def test_json_field_prompt_requests_json_only(self) -> None:
        messages = run_evaluation.build_messages(
            {
                "question": "추출",
                "scoring": "json_field",
                "answer_fields": {"사건번호": "x", "원고": "y"},
            },
            "문서 내용",
            "시스템",
        )
        self.assertIn("JSON 객체만 출력", messages[1]["content"])
        self.assertIn("사건번호, 원고", messages[1]["content"])

    def test_server_command_uses_configured_ngl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "llama-server.exe"
            model = root / "model.gguf"
            executable.touch()
            model.touch()
            server = run_evaluation.ServerProcess(
                executable,
                model,
                root / "server.log",
                "127.0.0.1",
                18080,
                4096,
                37,
                5,
            )
            fake_process = MagicMock()
            with patch.object(run_evaluation, "assert_port_available"), patch.object(
                run_evaluation.subprocess, "Popen", return_value=fake_process
            ) as popen:
                server.start()
            command = popen.call_args.args[0]
            server._close_log()
        self.assertEqual(command[command.index("-ngl") + 1], "37")

    def test_server_stop_uses_interrupt_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = run_evaluation.ServerProcess(
                root / "llama-server.exe",
                root / "model.gguf",
                root / "server.log",
                "127.0.0.1",
                18080,
                4096,
                99,
                5,
            )
            process = MagicMock()
            process.poll.return_value = None
            process.returncode = 0
            server.process = process
            server._log_handle = (root / "server.log").open("wb")
            result = server.stop()
        expected_signal = (
            run_evaluation.signal.CTRL_BREAK_EVENT
            if run_evaluation.os.name == "nt"
            else run_evaluation.signal.SIGINT
        )
        process.send_signal.assert_called_once_with(expected_signal)
        process.terminate.assert_not_called()
        self.assertEqual(result["method"], "interrupt")

    def test_server_stop_falls_back_to_terminate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = run_evaluation.ServerProcess(
                root / "llama-server.exe",
                root / "model.gguf",
                root / "server.log",
                "127.0.0.1",
                18080,
                4096,
                99,
                0.01,
            )
            process = MagicMock()
            process.poll.return_value = None
            process.wait.side_effect = [subprocess.TimeoutExpired("server", 0.01), None]
            process.returncode = 0
            server.process = process
            server._log_handle = (root / "server.log").open("wb")
            result = server.stop()
        process.terminate.assert_called_once()
        self.assertEqual(result["method"], "terminate")


if __name__ == "__main__":
    unittest.main()
