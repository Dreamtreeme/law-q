from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    create_run_directory,
    experiment_matrix,
    load_config,
    resolve_paths,
)


LOGGER = logging.getLogger("run_benchmarks")


class BenchmarkError(RuntimeError):
    """llama-bench 실행 또는 출력 검증이 실패했을 때 발생합니다."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
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


def find_benchmark_executable(config: dict[str, Any]) -> Path:
    paths = resolve_paths(config)
    root = paths["llama_cpp"]
    configured = paths.get("llama_bench")
    candidates = [
        configured,
        root / "build" / "bin" / "Release" / "llama-bench.exe",
        root / "build" / "bin" / "llama-bench.exe",
        root / "build" / "Release" / "bin" / "llama-bench.exe",
        root / "llama-bench.exe",
        root / "build" / "bin" / "llama-bench",
        root / "llama-bench",
    ]
    unique_candidates = list(dict.fromkeys(item for item in candidates if item is not None))
    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(item) for item in unique_candidates)
    raise BenchmarkError(f"llama-bench 실행 파일을 찾지 못했습니다:\n  - {searched}")


def validate_benchmark_config(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, dict):
        raise BenchmarkError("benchmark 설정이 필요합니다.")
    lengths = benchmark.get("prompt_lengths")
    if (
        not isinstance(lengths, list)
        or not lengths
        or any(not isinstance(value, int) or value <= 0 for value in lengths)
    ):
        raise BenchmarkError("benchmark.prompt_lengths는 양의 정수 배열이어야 합니다.")
    if len(lengths) != len(set(lengths)):
        raise BenchmarkError("benchmark.prompt_lengths에 중복 값이 있습니다.")
    repetitions = benchmark.get("repetitions")
    if not isinstance(repetitions, int) or repetitions < 3:
        raise BenchmarkError("benchmark.repetitions는 최소 3이어야 합니다.")
    generation_tokens = benchmark.get("generation_tokens")
    if not isinstance(generation_tokens, int) or generation_tokens <= 0:
        raise BenchmarkError("benchmark.generation_tokens는 양의 정수여야 합니다.")
    gpu_layers = config["runtime"].get("gpu_layers")
    if not isinstance(gpu_layers, int):
        raise BenchmarkError("llama-bench용 runtime.gpu_layers는 정수여야 합니다.")
    return benchmark


def build_command(
    executable: Path,
    model_path: Path,
    phase: str,
    config: dict[str, Any],
) -> list[str]:
    benchmark = config["benchmark"]
    prompt_lengths = ",".join(str(value) for value in benchmark["prompt_lengths"])
    command = [
        str(executable),
        "-m",
        str(model_path),
        "-r",
        str(benchmark["repetitions"]),
        "-o",
        "json",
        "-ngl",
        str(config["runtime"]["gpu_layers"]),
        "-t",
        str(config["runtime"]["threads"]),
        "-b",
        str(benchmark.get("batch_size", 2048)),
        "-ub",
        str(benchmark.get("ubatch_size", 512)),
    ]
    if phase == "prompt_processing":
        command.extend(["-p", prompt_lengths, "-n", "0"])
    elif phase == "token_generation":
        command.extend(
            [
                "-p",
                "0",
                "-n",
                str(benchmark["generation_tokens"]),
                "-d",
                prompt_lengths,
            ]
        )
    else:
        raise ValueError(f"알 수 없는 벤치마크 단계: {phase}")
    # --no-warmup을 전달하지 않는다. llama-bench 기본 워밍업 후 -r 본 측정을 사용한다.
    return command


def parse_json_output(stdout: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"llama-bench JSON 출력 파싱 실패: {error}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise BenchmarkError("llama-bench JSON 출력은 객체 배열이어야 합니다.")
    return value


def _mean_and_stddev(values: list[float]) -> tuple[float, float]:
    if not values:
        raise BenchmarkError("반복 측정값이 비어 있습니다.")
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stddev


def normalize_rows(
    rows: list[dict[str, Any]],
    phase: str,
    model: str,
    quantization: str,
    prompt_lengths: list[int],
    repetitions: int,
) -> list[dict[str, Any]]:
    normalized = []
    seen_lengths: set[int] = set()
    for row in rows:
        if phase == "prompt_processing":
            if int(row.get("n_prompt", 0)) <= 0 or int(row.get("n_gen", 0)) != 0:
                continue
            prompt_length = int(row["n_prompt"])
        else:
            if int(row.get("n_prompt", 0)) != 0 or int(row.get("n_gen", 0)) <= 0:
                continue
            prompt_length = int(row.get("n_depth", 0))
        if prompt_length not in prompt_lengths:
            continue

        samples_ts = [float(value) for value in row.get("samples_ts", [])]
        samples_ns = [int(value) for value in row.get("samples_ns", [])]
        if len(samples_ts) < repetitions or len(samples_ns) < repetitions:
            raise BenchmarkError(
                f"{model}/{quantization}/{phase}/{prompt_length}: "
                f"반복값이 {repetitions}개보다 적습니다."
            )
        measured_ts = samples_ts[-repetitions:]
        measured_ns = samples_ns[-repetitions:]
        throughput_mean, throughput_stddev = _mean_and_stddev(measured_ts)
        latency_ms = [value / 1_000_000 for value in measured_ns]
        latency_mean, latency_stddev = _mean_and_stddev(latency_ms)
        normalized.append(
            {
                "model": model,
                "quantization": quantization,
                "benchmark_type": phase,
                "prompt_length": prompt_length,
                "generated_tokens": int(row.get("n_gen", 0)),
                "repetitions": repetitions,
                "warmup": "llama-bench_builtin",
                "tokens_per_second": {
                    "mean": round(throughput_mean, 6),
                    "stddev": round(throughput_stddev, 6),
                    "samples": measured_ts,
                    "llama_bench_mean": row.get("avg_ts"),
                    "llama_bench_stddev": row.get("stddev_ts"),
                },
                "latency_ms": {
                    "mean": round(latency_mean, 6),
                    "stddev": round(latency_stddev, 6),
                    "samples": latency_ms,
                },
                "backend": row.get("backends"),
                "gpu_info": row.get("gpu_info"),
                "n_gpu_layers": row.get("n_gpu_layers"),
                "model_size_bytes": row.get("model_size"),
                "build_commit": row.get("build_commit"),
                "test_time": row.get("test_time"),
            }
        )
        seen_lengths.add(prompt_length)

    missing = sorted(set(prompt_lengths) - seen_lengths)
    if missing:
        raise BenchmarkError(
            f"{model}/{quantization}/{phase}: 결과에 프롬프트 길이가 없습니다: {missing}"
        )
    return sorted(normalized, key=lambda item: item["prompt_length"])


def run_phase(
    executable: Path,
    model_path: Path,
    model: str,
    quantization: str,
    phase: str,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    command = build_command(executable, model_path, phase, config)
    LOGGER.info("[%s][%s][%s] 실행: %s", model, quantization, phase, subprocess.list2cmdline(command))
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(config["benchmark"].get("timeout_seconds", 3600)),
        )
    except subprocess.TimeoutExpired as error:
        raise BenchmarkError(
            f"{model}/{quantization}/{phase}: {error.timeout}초 시간 제한 초과"
        ) from error
    except OSError as error:
        raise BenchmarkError(f"{model}/{quantization}/{phase}: 실행 실패: {error}") from error

    duration = round(time.perf_counter() - started, 3)
    write_text(output_dir / f"{phase}.stdout.json", completed.stdout)
    write_text(output_dir / f"{phase}.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise BenchmarkError(
            f"{model}/{quantization}/{phase}: llama-bench exit={completed.returncode}; "
            f"상세 로그: {output_dir / f'{phase}.stderr.log'}"
        )
    rows = parse_json_output(completed.stdout)
    normalized = normalize_rows(
        rows,
        phase,
        model,
        quantization,
        config["benchmark"]["prompt_lengths"],
        config["benchmark"]["repetitions"],
    )
    return {
        "phase": phase,
        "status": "success",
        "duration_seconds": duration,
        "command": command,
        "warmup": "llama-bench_builtin",
        "measurements": normalized,
    }


def benchmark_combination(
    executable: Path,
    combination: dict[str, str],
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    paths = resolve_paths(config)
    model = combination["model"]
    quantization = combination["quantization"]
    model_path = paths["gguf"] / model / f"{model}-{quantization}.gguf"
    output_dir = run_dir / "benchmarks" / model / quantization
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_failures = output_dir / "failures.jsonl"
    if previous_failures.exists():
        previous_failures.unlink()
    record: dict[str, Any] = {
        "model": model,
        "quantization": quantization,
        "model_path": str(model_path.resolve()),
        "model_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "phases": {},
    }
    write_json(output_dir / "summary.json", record)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise BenchmarkError(f"GGUF 파일을 찾지 못했습니다: {model_path}")

    failures = 0
    all_measurements: list[dict[str, Any]] = []
    for phase in ("prompt_processing", "token_generation"):
        try:
            phase_record = run_phase(
                executable,
                model_path,
                model,
                quantization,
                phase,
                config,
                output_dir,
            )
            all_measurements.extend(phase_record["measurements"])
        except Exception as error:
            failures += 1
            LOGGER.exception("[%s][%s][%s] 실패: %s", model, quantization, phase, error)
            phase_record = {
                "phase": phase,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(output_dir / "failures.jsonl", phase_record)
        record["phases"][phase] = phase_record
        write_json(output_dir / "summary.json", record)

    record["measurements"] = all_measurements
    record["status"] = "partial_failure" if failures else "success"
    record["finished_at_utc"] = utc_now()
    write_json(output_dir / "summary.json", record)
    return record


def run_all(config: dict[str, Any], run_name: str | None, verbose: bool) -> int:
    run_dir = create_run_directory(config, run_name)
    setup_logging(run_dir / "benchmark.log", verbose)
    summary_path = run_dir / "benchmark-results.json"
    failures_path = run_dir / "failures.jsonl"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "configuration": config.get("benchmark"),
        "gpu_layers": config["runtime"].get("gpu_layers"),
        "combinations": [],
        "measurements": [],
        "failures": [],
    }
    write_json(summary_path, report)
    try:
        validate_benchmark_config(config)
        executable = find_benchmark_executable(config)
    except Exception as error:
        failure = {
            "occurred_at_utc": utc_now(),
            "model": "__pipeline__",
            "quantization": None,
            "phase": "preflight",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        LOGGER.exception("[preflight] 벤치마크 준비 실패: %s", error)
        report["failures"].append(failure)
        append_jsonl(failures_path, failure)
        report["status"] = "failed"
        report["finished_at_utc"] = utc_now()
        write_json(summary_path, report)
        return 1

    any_failure = False
    for combination in experiment_matrix(config):
        try:
            record = benchmark_combination(executable, combination, config, run_dir)
            if record["status"] != "success":
                any_failure = True
            report["measurements"].extend(record.get("measurements", []))
        except KeyboardInterrupt:
            report["status"] = "interrupted"
            report["finished_at_utc"] = utc_now()
            write_json(summary_path, report)
            return 130
        except Exception as error:
            any_failure = True
            LOGGER.exception(
                "[%s][%s] 벤치마크 실패: %s",
                combination["model"],
                combination["quantization"],
                error,
            )
            record = {
                "model": combination["model"],
                "quantization": combination["quantization"],
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            report["failures"].append(record)
            append_jsonl(
                failures_path,
                {"occurred_at_utc": utc_now(), "phase": "combination", **record},
            )
        report["combinations"].append(record)
        write_json(summary_path, report)

    report["status"] = "partial_failure" if any_failure else "success"
    report["finished_at_utc"] = utc_now()
    write_json(summary_path, report)
    return 1 if any_failure else 0


def dry_run(config: dict[str, Any]) -> None:
    benchmark = validate_benchmark_config(config)
    print(f"prompt_lengths={benchmark['prompt_lengths']}")
    print(f"generation_tokens={benchmark['generation_tokens']}")
    print(f"repetitions={benchmark['repetitions']} (각 테스트 전 llama-bench 기본 워밍업)")
    print(f"gpu_layers={config['runtime']['gpu_layers']}")
    for combination in experiment_matrix(config):
        print(f"{combination['model']} / {combination['quantization']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="llama-bench로 모든 GGUF 조합의 PP/TG 속도를 측정합니다."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--run-name", help="results 아래에 생성할 실행 이름")
    parser.add_argument("--verbose", action="store_true", help="상세 로그를 콘솔에도 표시")
    parser.add_argument("--dry-run", action="store_true", help="실행 없이 설정과 조합만 출력")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        dry_run(config)
        return
    raise SystemExit(run_all(config, args.run_name, args.verbose))


if __name__ == "__main__":
    main()
