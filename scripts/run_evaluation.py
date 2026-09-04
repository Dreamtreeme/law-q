from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from common import (
    DEFAULT_CONFIG,
    create_run_directory,
    experiment_matrix,
    load_config,
    resolve_paths,
)
from scoring import load_dataset, score_dataset


LOGGER = logging.getLogger("run_evaluation")


class EvaluationRunError(RuntimeError):
    """서버 실행 또는 평가 요청이 실패했을 때 발생합니다."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def setup_logging(path: Path, verbose: bool) -> None:
    LOGGER.setLevel(logging.DEBUG)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def find_server(config: dict[str, Any]) -> Path:
    paths = resolve_paths(config)
    root = paths["llama_cpp"]
    configured = paths.get("llama_server")
    candidates = [
        configured,
        root / "build" / "bin" / "Release" / "llama-server.exe",
        root / "build" / "bin" / "llama-server.exe",
        root / "build" / "Release" / "bin" / "llama-server.exe",
        root / "llama-server.exe",
        root / "build" / "bin" / "llama-server",
        root / "llama-server",
    ]
    unique_candidates = list(dict.fromkeys(item for item in candidates if item is not None))
    for candidate in unique_candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(item) for item in unique_candidates)
    raise EvaluationRunError(f"llama-server 실행 파일을 찾지 못했습니다:\n  - {searched}")


def assert_port_available(host: str, port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as error:
        raise EvaluationRunError(f"서버 포트 {host}:{port}를 사용할 수 없습니다: {error}") from error


class ServerProcess:
    """llama-server와 로그 핸들의 전체 생명주기를 관리합니다."""

    def __init__(
        self,
        executable: Path,
        model_path: Path,
        log_path: Path,
        host: str,
        port: int,
        context_size: int,
        gpu_layers: int | str,
        shutdown_timeout: float,
    ) -> None:
        self.executable = executable
        self.model_path = model_path
        self.log_path = log_path
        self.host = host
        self.port = port
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.shutdown_timeout = shutdown_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None

    def start(self) -> None:
        assert_port_available(self.host, self.port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("wb")
        command = [
            str(self.executable),
            "--model",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "-ngl",
            str(self.gpu_layers),
            "--parallel",
            "1",
            "--no-webui",
        ]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        LOGGER.info("서버 시작: %s", subprocess.list2cmdline(command))
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception:
            self._close_log()
            raise

    def wait_until_ready(self, timeout: float, poll_interval: float) -> float:
        if self.process is None:
            raise EvaluationRunError("서버 프로세스가 시작되지 않았습니다.")
        started = time.perf_counter()
        deadline = started + timeout
        health_url = f"http://{self.host}:{self.port}/health"
        last_error = "응답 없음"
        while time.perf_counter() < deadline:
            returncode = self.process.poll()
            if returncode is not None:
                raise EvaluationRunError(
                    f"health check 전에 llama-server가 종료됐습니다 (exit={returncode}). "
                    f"로그: {self.log_path}"
                )
            try:
                response = requests.get(health_url, timeout=min(2.0, poll_interval + 1.0))
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("status") == "ok":
                        elapsed = time.perf_counter() - started
                        LOGGER.info("서버 준비 완료: %.3f초", elapsed)
                        return elapsed
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            except (requests.RequestException, ValueError, AttributeError) as error:
                last_error = str(error)
            time.sleep(poll_interval)
        raise EvaluationRunError(
            f"llama-server 준비 시간 초과 ({timeout}초), 마지막 상태: {last_error}. "
            f"로그: {self.log_path}"
        )

    def stop(self) -> dict[str, Any]:
        process = self.process
        if process is None:
            self._close_log()
            return {"method": "not_started", "returncode": None}
        if process.poll() is not None:
            returncode = process.returncode
            self._close_log()
            return {"method": "already_exited", "returncode": returncode}

        method = "interrupt"
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=self.shutdown_timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            method = "terminate"
            try:
                process.terminate()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                method = "kill_tree"
                self._kill_tree(process)
        finally:
            self._close_log()
        LOGGER.info("서버 종료: method=%s returncode=%s", method, process.returncode)
        return {"method": method, "returncode": process.returncode}

    @staticmethod
    def _kill_tree(process: subprocess.Popen[bytes]) -> None:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                LOGGER.error("taskkill 실패: pid=%s error=%s", process.pid, error)
            if process.poll() is None:
                process.kill()
        else:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            LOGGER.error("서버 프로세스 종료 확인 실패: pid=%s", process.pid)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "ServerProcess":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def parse_vram_csv(text: str, sampled_at: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"예상하지 못한 nvidia-smi 출력: {line}")
        records.append(
            {
                "sampled_at_utc": sampled_at,
                "gpu_index": int(parts[0]),
                "gpu_name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
            }
        )
    return records


class VramSampler:
    def __init__(self, output: Path, interval: float) -> None:
        self.output = output
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_count = 0
        self._failed_samples = 0
        self._max_total_mib = 0
        self._max_by_gpu: dict[int, int] = {}
        self._last_error: str | None = None

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="vram-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sampled_at = utc_now()
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                )
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or f"exit={completed.returncode}")
                samples = parse_vram_csv(completed.stdout, sampled_at)
                total = sum(sample["memory_used_mib"] for sample in samples)
                for sample in samples:
                    append_jsonl(self.output, sample)
                with self._lock:
                    self._sample_count += 1
                    self._max_total_mib = max(self._max_total_mib, total)
                    for sample in samples:
                        index = sample["gpu_index"]
                        self._max_by_gpu[index] = max(
                            self._max_by_gpu.get(index, 0), sample["memory_used_mib"]
                        )
            except Exception as error:
                with self._lock:
                    self._failed_samples += 1
                    self._last_error = f"{type(error).__name__}: {error}"
                LOGGER.warning("VRAM 샘플링 실패: %s", error)
            self._stop_event.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(15.0, self.interval + 10.0))
            if self._thread.is_alive():
                LOGGER.error("VRAM 샘플러 스레드가 제한 시간 안에 종료되지 않았습니다.")
        with self._lock:
            return {
                "sample_count": self._sample_count,
                "failed_samples": self._failed_samples,
                "max_total_memory_used_mib": self._max_total_mib,
                "max_memory_used_mib_by_gpu": {
                    str(index): value for index, value in sorted(self._max_by_gpu.items())
                },
                "last_error": self._last_error,
                "samples_path": str(self.output.resolve()),
            }


def safe_context_path(dataset_path: Path, context_doc: str) -> Path:
    eval_root = dataset_path.resolve().parent
    context_path = (eval_root / context_doc).resolve()
    try:
        context_path.relative_to(eval_root)
    except ValueError as error:
        raise EvaluationRunError(f"context_doc가 평가셋 디렉터리를 벗어납니다: {context_doc}") from error
    if not context_path.is_file():
        raise EvaluationRunError(f"문맥 문서를 찾지 못했습니다: {context_path}")
    return context_path


def build_messages(
    item: dict[str, Any], context: str, system_prompt: str
) -> list[dict[str, str]]:
    format_instruction = ""
    if item["scoring"] == "json_field":
        fields = ", ".join(item["answer_fields"])
        format_instruction = f"\n반드시 다음 필드를 가진 JSON 객체만 출력하세요: {fields}"
    user_prompt = (
        "[문서]\n"
        f"{context}\n\n"
        "[질문]\n"
        f"{item['question']}"
        f"{format_instruction}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def stream_chat_completion(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """SSE 첫 콘텐츠와 종료 시각을 측정하고 응답 원문을 합칩니다."""
    started = clock()
    first_token_at: float | None = None
    chunks: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    event_count = 0

    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(10.0, timeout),
        ) as response:
            response.raise_for_status()
            # llama-server does not currently declare an SSE charset.  Letting
            # requests decode first can therefore select ISO-8859-1; Python's
            # Unicode splitlines then mistakes UTF-8 continuation bytes such as
            # 0x85 for line separators and cuts Korean JSON strings in half.
            # Split the HTTP stream as bytes and decode each complete SSE line.
            for raw_line in response.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise EvaluationRunError(f"스트림 JSON 파싱 실패: {data[:300]}") from error
                event_count += 1
                if event.get("usage") is not None:
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = clock()
                    chunks.append(content)
    except requests.RequestException as error:
        raise EvaluationRunError(f"llama-server 요청 실패: {error}") from error

    finished = clock()
    return {
        "response": "".join(chunks),
        "ttft_seconds": round(first_token_at - started, 6) if first_token_at else None,
        "total_response_seconds": round(finished - started, 6),
        "stream_event_count": event_count,
        "finish_reason": finish_reason,
        "usage": usage,
    }


def evaluate_combination(
    config: dict[str, Any],
    dataset: list[dict[str, Any]],
    dataset_path: Path,
    combination: dict[str, str],
    run_dir: Path,
    server_executable: Path,
) -> dict[str, Any]:
    paths = resolve_paths(config)
    runtime = config["runtime"]
    server_config = config["server"]
    model_name = combination["model"]
    quantization = combination["quantization"]
    model_path = paths["gguf"] / model_name / f"{model_name}-{quantization}.gguf"
    combo_dir = run_dir / "evaluations" / model_name / quantization
    combo_dir.mkdir(parents=True, exist_ok=True)
    responses_path = combo_dir / "responses.jsonl"
    failures_path = combo_dir / "failures.jsonl"
    server_log = combo_dir / "server.log"
    vram_samples = combo_dir / "vram-samples.jsonl"
    # 중단된 조합을 --resume으로 재시도할 때 이전 append-only 원본과 섞이지 않게 한다.
    for previous in (responses_path, failures_path, vram_samples):
        if previous.exists():
            previous.unlink()
    record: dict[str, Any] = {
        "model": model_name,
        "quantization": quantization,
        "model_path": str(model_path.resolve()),
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "questions_total": len(dataset),
        "questions_completed": 0,
        "request_failures": 0,
        "server_startup_seconds": None,
        "server_shutdown": None,
        "vram": None,
    }
    write_json(combo_dir / "summary.json", record)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise EvaluationRunError(f"GGUF 파일을 찾지 못했습니다: {model_path}")

    host = str(server_config.get("host", "127.0.0.1"))
    port = int(server_config.get("port", 18080))
    sampler = VramSampler(
        vram_samples, float(server_config.get("vram_sample_interval_seconds", 0.5))
    )
    server = ServerProcess(
        executable=server_executable,
        model_path=model_path,
        log_path=server_log,
        host=host,
        port=port,
        context_size=int(runtime["context_size"]),
        gpu_layers=runtime["gpu_layers"],
        shutdown_timeout=float(server_config.get("shutdown_timeout_seconds", 15)),
    )
    predictions: dict[str, str] = {}
    sampler.start()
    try:
        server.start()
        record["server_startup_seconds"] = server.wait_until_ready(
            timeout=float(server_config.get("startup_timeout_seconds", 300)),
            poll_interval=float(server_config.get("health_poll_interval_seconds", 0.5)),
        )
        endpoint = f"http://{host}:{port}/v1/chat/completions"
        for item in dataset:
            LOGGER.info("[%s][%s][%s] 평가 시작", model_name, quantization, item["id"])
            started_at = utc_now()
            try:
                context_path = safe_context_path(dataset_path, item["context_doc"])
                context = context_path.read_text(encoding="utf-8-sig")
                messages = build_messages(
                    item, context, str(config["evaluation"]["system_prompt"])
                )
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": float(runtime["temperature"]),
                    "seed": int(runtime["seed"]),
                    "max_tokens": int(runtime["max_tokens"]),
                    "stream": True,
                }
                measured = stream_chat_completion(
                    endpoint,
                    payload,
                    timeout=float(server_config.get("request_timeout_seconds", 300)),
                )
                response_record = {
                    "id": item["id"],
                    "type": item["type"],
                    "scoring": item["scoring"],
                    "context_doc": item["context_doc"],
                    "question": item["question"],
                    "status": "success",
                    "started_at_utc": started_at,
                    "finished_at_utc": utc_now(),
                    "temperature": float(runtime["temperature"]),
                    "seed": int(runtime["seed"]),
                    **measured,
                }
                predictions[item["id"]] = measured["response"]
                LOGGER.info(
                    "[%s][%s][%s] 완료: ttft=%s total=%s",
                    model_name,
                    quantization,
                    item["id"],
                    measured["ttft_seconds"],
                    measured["total_response_seconds"],
                )
            except Exception as error:
                record["request_failures"] += 1
                predictions[item["id"]] = ""
                response_record = {
                    "id": item["id"],
                    "type": item["type"],
                    "scoring": item["scoring"],
                    "context_doc": item["context_doc"],
                    "question": item["question"],
                    "response": "",
                    "status": "failed",
                    "started_at_utc": started_at,
                    "finished_at_utc": utc_now(),
                    "ttft_seconds": None,
                    "total_response_seconds": None,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(failures_path, response_record)
                LOGGER.exception(
                    "[%s][%s][%s] 평가 실패: %s",
                    model_name,
                    quantization,
                    item["id"],
                    error,
                )
            append_jsonl(responses_path, response_record)
            record["questions_completed"] += 1
            write_json(combo_dir / "summary.json", record)

        scores = score_dataset(dataset, predictions)
        write_json(combo_dir / "scores.json", scores)
        record["score_summary"] = scores["summary"]
        record["score_by_type"] = scores["by_type"]
        record["status"] = "partial_failure" if record["request_failures"] else "success"
    finally:
        record["server_shutdown"] = server.stop()
        record["vram"] = sampler.stop()
        write_json(combo_dir / "vram-summary.json", record["vram"])
        record["finished_at_utc"] = utc_now()
        if record["status"] == "running":
            record["status"] = "failed"
        write_json(combo_dir / "summary.json", record)
    return record


def resolve_dataset(config: dict[str, Any], override: str | None) -> Path:
    base = Path(config["_project_root"])
    value = Path(override) if override else Path(config["evaluation"]["dataset"])
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def run_all(
    config: dict[str, Any],
    dataset_path: Path,
    run_name: str | None,
    verbose: bool,
) -> int:
    run_dir = create_run_directory(config, run_name)
    setup_logging(run_dir / "evaluation.log", verbose)
    summary_path = run_dir / "evaluation-run.json"
    failures_path = run_dir / "failures.jsonl"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "dataset": str(dataset_path),
        "dataset_items": None,
        "sampling": {
            "temperature": float(config["runtime"]["temperature"]),
            "seed": int(config["runtime"]["seed"]),
        },
        "gpu_layers": config["runtime"]["gpu_layers"],
        "combinations": [],
        "failures": [],
    }
    write_json(summary_path, report)
    try:
        if float(config["runtime"]["temperature"]) != 0.0:
            raise EvaluationRunError(
                "재현 가능한 평가를 위해 runtime.temperature는 0이어야 합니다."
            )
        dataset = load_dataset(dataset_path)
        report["dataset_items"] = len(dataset)
        write_json(summary_path, report)
        server_executable = find_server(config)
    except Exception as error:
        failure = {
            "occurred_at_utc": utc_now(),
            "model": "__pipeline__",
            "quantization": None,
            "stage": "preflight",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        LOGGER.exception("[preflight] 평가 준비 실패: %s", error)
        report["failures"].append(failure)
        append_jsonl(failures_path, failure)
        report["status"] = "failed"
        report["finished_at_utc"] = utc_now()
        report["successful_combinations"] = 0
        report["failed_combinations"] = 0
        write_json(summary_path, report)
        return 1

    interrupted = False
    for combination in experiment_matrix(config):
        LOGGER.info(
            "[%s][%s] 조합 평가 시작", combination["model"], combination["quantization"]
        )
        try:
            combo_record = evaluate_combination(
                config, dataset, dataset_path, combination, run_dir, server_executable
            )
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.warning("사용자 중단을 감지했습니다.")
            break
        except Exception as error:
            LOGGER.exception(
                "[%s][%s] 조합 평가 실패: %s",
                combination["model"],
                combination["quantization"],
                error,
            )
            combo_record = {
                "model": combination["model"],
                "quantization": combination["quantization"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            report["failures"].append(combo_record)
            append_jsonl(
                failures_path,
                {
                    "occurred_at_utc": utc_now(),
                    "stage": "combination",
                    **combo_record,
                },
            )
        report["combinations"].append(combo_record)
        write_json(summary_path, report)

    failures = sum(item.get("status") != "success" for item in report["combinations"])
    report["finished_at_utc"] = utc_now()
    if interrupted:
        report["status"] = "interrupted"
    elif failures:
        report["status"] = "partial_failure"
    else:
        report["status"] = "success"
    report["successful_combinations"] = len(report["combinations"]) - failures
    report["failed_combinations"] = failures
    write_json(summary_path, report)
    return 130 if interrupted else (1 if failures else 0)


def dry_run(config: dict[str, Any], dataset_path: Path) -> None:
    print(f"dataset={dataset_path}")
    print(f"gpu_layers={config['runtime']['gpu_layers']}")
    print(f"temperature={config['runtime']['temperature']}")
    print(f"seed={config['runtime']['seed']}")
    for combination in experiment_matrix(config):
        print(f"{combination['model']} / {combination['quantization']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="llama-server로 모든 모델·양자화 조합의 평가셋을 실행합니다."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--dataset", help="설정의 evaluation.dataset을 임시로 대체")
    parser.add_argument("--run-name", help="results 아래에 생성할 실행 이름")
    parser.add_argument("--verbose", action="store_true", help="상세 로그를 콘솔에도 표시")
    parser.add_argument("--dry-run", action="store_true", help="서버를 띄우지 않고 조합만 출력")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = resolve_dataset(config, args.dataset)
    if args.dry_run:
        dry_run(config, dataset_path)
        return
    raise SystemExit(run_all(config, dataset_path, args.run_name, args.verbose))


if __name__ == "__main__":
    main()
