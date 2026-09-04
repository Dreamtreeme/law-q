from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import statistics
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import prepare_models
import run_benchmarks
import run_evaluation
import validate_dataset
from common import (
    DEFAULT_CONFIG,
    create_run_directory,
    experiment_matrix,
    load_config,
    resolve_paths,
)
from scoring import load_dataset


LOGGER = logging.getLogger("run_pipeline")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
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

    for child in (
        prepare_models.LOGGER,
        run_evaluation.LOGGER,
        run_benchmarks.LOGGER,
    ):
        child.setLevel(logging.DEBUG)
        child.handlers.clear()
        child.handlers.extend(LOGGER.handlers)
        child.propagate = False


def result_fields(prompt_lengths: list[int]) -> list[str]:
    fields = [
        "combination_index",
        "model",
        "quantization",
        "status",
        "step2_gguf_status",
        "step3_evaluation_status",
        "step4_benchmark_status",
        "failed_stage",
        "error",
        "gguf_path",
        "gguf_size_bytes",
        "overall_score",
        "overall_percent",
        "type1_score",
        "type2_score",
        "type3_score",
        "type4_score",
        "json_parse_failures",
        "request_failures",
        "ttft_mean_seconds",
        "ttft_stddev_seconds",
        "response_time_mean_seconds",
        "response_time_stddev_seconds",
        "max_vram_used_mib",
    ]
    for length in prompt_lengths:
        fields.extend(
            [
                f"pp_{length}_mean_tps",
                f"pp_{length}_stddev_tps",
                f"tg_{length}_mean_tps",
                f"tg_{length}_stddev_tps",
            ]
        )
    fields.extend(["started_at_utc", "finished_at_utc"])
    return fields


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """조합 하나가 끝날 때마다 전체 CSV를 원자적으로 교체합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def combination_key(value: dict[str, Any]) -> tuple[str, str]:
    return str(value["model"]), str(value["quantization"])


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        combination_key(row)
        for row in rows
        if row.get("status") == "success"
    }


def ordered_rows(
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
    combinations: list[dict[str, str]],
) -> list[dict[str, Any]]:
    ordered = [
        rows_by_key[combination_key(combination)]
        for combination in combinations
        if combination_key(combination) in rows_by_key
    ]
    known = {combination_key(combination) for combination in combinations}
    ordered.extend(row for key, row in rows_by_key.items() if key not in known)
    return ordered


def resolve_run_directory(
    config: dict[str, Any], run_name: str | None, resume: bool
) -> tuple[Path, bool]:
    results_dir = resolve_paths(config)["results"]
    if run_name and not re.fullmatch(r"[A-Za-z0-9._-]+", run_name):
        raise ValueError("run-name에는 영문자, 숫자, 점, 밑줄, 하이픈만 사용할 수 있습니다.")
    if not resume:
        name = run_name or datetime.now().astimezone().strftime("pipeline-%Y%m%d-%H%M%S")
        return create_run_directory(config, name), False

    if run_name:
        run_dir = results_dir / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"재개할 실행 디렉터리가 없습니다: {run_dir}")
        return run_dir, True

    candidates = sorted(
        (
            path
            for path in results_dir.glob("pipeline-*")
            if path.is_dir()
            and (
                (path / "results.csv").is_file()
                or (path / "pipeline-state.json").is_file()
            )
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("재개할 pipeline-* 실행 결과를 찾지 못했습니다.")
    return candidates[0], True


def assert_resume_config_matches(config: dict[str, Any], run_dir: Path) -> None:
    snapshot = run_dir / "experiment.yaml"
    if not snapshot.is_file():
        raise ValueError(f"재개 실행의 설정 스냅샷이 없습니다: {snapshot}")
    current_hash = hashlib.sha256(Path(config["_config_path"]).read_bytes()).hexdigest()
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if current_hash != snapshot_hash:
        raise ValueError(
            "현재 설정과 재개 대상의 설정 스냅샷이 다릅니다. "
            f"current={current_hash}, snapshot={snapshot_hash}"
        )


def resolve_tools(config: dict[str, Any]) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    try:
        tools["converter"], tools["quantizer"] = prepare_models.validate_llama_cpp(config)
    except Exception as error:
        tools["prepare_error"] = error
    try:
        tools["server"] = run_evaluation.find_server(config)
    except Exception as error:
        tools["evaluation_error"] = error
    try:
        run_benchmarks.validate_benchmark_config(config)
        tools["bench"] = run_benchmarks.find_benchmark_executable(config)
    except Exception as error:
        tools["benchmark_error"] = error
    try:
        dataset_path = run_evaluation.resolve_dataset(config, None)
        tools["dataset_path"] = dataset_path
        tools["dataset"] = load_dataset(dataset_path)
        evaluation_config = config.get("evaluation", {})
        if evaluation_config.get("require_dataset_lock", False):
            lock_value = evaluation_config.get(
                "dataset_lock", str(dataset_path.parent / "dataset.lock.json")
            )
            lock_path = Path(lock_value)
            if not lock_path.is_absolute():
                lock_path = Path(config["_project_root"]) / lock_path
            tools["dataset_lock"] = validate_dataset.assert_dataset_lock(
                dataset_path,
                lock_path,
                {
                    int(key): int(value)
                    for key, value in evaluation_config.get(
                        "expected_type_counts", {}
                    ).items()
                },
            )
    except Exception as error:
        tools["dataset_error"] = error
    return tools


def find_model_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    for model in config["models"]:
        if model["name"] == model_name:
            return model
    raise KeyError(f"모델 설정을 찾지 못했습니다: {model_name}")


def prepare_combination(
    config: dict[str, Any], combination: dict[str, str], tools: dict[str, Any]
) -> dict[str, Any]:
    paths = resolve_paths(config)
    model = find_model_config(config, combination["model"])
    model_name = combination["model"]
    quantization = combination["quantization"]
    output_dir = paths["gguf"] / model_name
    quantized_path = output_dir / f"{model_name}-{quantization}.gguf"
    if prepare_models.existing_artifact(quantized_path):
        return {
            "status": "success",
            "action": "skipped_existing",
            "artifact": prepare_models.artifact_info(quantized_path),
        }
    if "prepare_error" in tools:
        raise prepare_models.StageError(str(tools["prepare_error"]))

    model_dir = paths["models"] / model_name
    if not (model_dir / "config.json").is_file():
        raise prepare_models.StageError(
            f"원본 모델이 없습니다. 단계 1을 먼저 실행하세요: {model_dir}"
        )
    marker = prepare_models._matching_download_marker(
        model_dir / prepare_models.DOWNLOAD_MARKER, model
    )
    if marker is None:
        raise prepare_models.StageError(
            f"원본 모델의 저장소/revision 검증 마커가 없거나 설정과 다릅니다: "
            f"{model_dir / prepare_models.DOWNLOAD_MARKER}"
        )
    f16_path = output_dir / f"{model_name}-F16.gguf"
    conversion = prepare_models.convert_model(
        model, model_dir, f16_path, tools["converter"]
    )
    quantized = prepare_models.quantize_model(
        model,
        f16_path,
        quantized_path,
        quantization,
        combination["quantize_type"],
        tools["quantizer"],
        int(config["runtime"]["threads"]),
    )
    return {
        "status": "success",
        "action": "prepared",
        "conversion": conversion,
        "quantization": quantized,
        "artifact": prepare_models.artifact_info(quantized_path),
    }


def _metric_stats(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    return (
        round(statistics.fmean(values), 6),
        round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    )


def add_evaluation_metrics(
    row: dict[str, Any], record: dict[str, Any], run_dir: Path
) -> None:
    summary = record.get("score_summary", {})
    row["overall_score"] = summary.get("average_score", "")
    row["overall_percent"] = summary.get("percent", "")
    row["json_parse_failures"] = summary.get("json_parse_failures", "")
    row["request_failures"] = record.get("request_failures", "")
    by_type = record.get("score_by_type", {})
    for item_type in range(1, 5):
        row[f"type{item_type}_score"] = by_type.get(str(item_type), {}).get(
            "average_score", ""
        )
    vram = record.get("vram") or {}
    row["max_vram_used_mib"] = vram.get("max_total_memory_used_mib", "")

    responses_path = (
        run_dir
        / "evaluations"
        / row["model"]
        / row["quantization"]
        / "responses.jsonl"
    )
    ttft: list[float] = []
    total: list[float] = []
    if responses_path.is_file():
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            response = json.loads(line)
            if response.get("ttft_seconds") is not None:
                ttft.append(float(response["ttft_seconds"]))
            if response.get("total_response_seconds") is not None:
                total.append(float(response["total_response_seconds"]))
    row["ttft_mean_seconds"], row["ttft_stddev_seconds"] = _metric_stats(ttft)
    row["response_time_mean_seconds"], row["response_time_stddev_seconds"] = (
        _metric_stats(total)
    )


def add_benchmark_metrics(row: dict[str, Any], record: dict[str, Any]) -> None:
    for measurement in record.get("measurements", []):
        length = measurement["prompt_length"]
        prefix = "pp" if measurement["benchmark_type"] == "prompt_processing" else "tg"
        row[f"{prefix}_{length}_mean_tps"] = measurement["tokens_per_second"]["mean"]
        row[f"{prefix}_{length}_stddev_tps"] = measurement["tokens_per_second"]["stddev"]


def failure_event(
    combination: dict[str, str], stage: str, error: BaseException
) -> dict[str, Any]:
    return {
        "occurred_at_utc": utc_now(),
        "model": combination["model"],
        "quantization": combination["quantization"],
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def process_combination(
    index: int,
    config: dict[str, Any],
    combination: dict[str, str],
    tools: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = resolve_paths(config)
    model = combination["model"]
    quantization = combination["quantization"]
    gguf_path = paths["gguf"] / model / f"{model}-{quantization}.gguf"
    row: dict[str, Any] = {
        "combination_index": index,
        "model": model,
        "quantization": quantization,
        "status": "running",
        "step2_gguf_status": "pending",
        "step3_evaluation_status": "pending",
        "step4_benchmark_status": "pending",
        "failed_stage": "",
        "error": "",
        "gguf_path": str(gguf_path.resolve()),
        "gguf_size_bytes": "",
        "started_at_utc": utc_now(),
        "finished_at_utc": "",
    }
    failures: list[dict[str, Any]] = []

    try:
        prepared = prepare_combination(config, combination, tools)
        row["step2_gguf_status"] = prepared["status"]
        row["gguf_size_bytes"] = prepared["artifact"]["size_bytes"]
    except Exception as error:
        row["step2_gguf_status"] = "failed"
        row["step3_evaluation_status"] = "blocked"
        row["step4_benchmark_status"] = "blocked"
        row["failed_stage"] = "step2_gguf"
        row["error"] = str(error)
        failures.append(failure_event(combination, "step2_gguf", error))
        row["status"] = "failed"
        row["finished_at_utc"] = utc_now()
        return row, failures

    evaluation_record: dict[str, Any] | None = None
    try:
        if "evaluation_error" in tools:
            raise run_evaluation.EvaluationRunError(str(tools["evaluation_error"]))
        if "dataset_error" in tools:
            raise run_evaluation.EvaluationRunError(str(tools["dataset_error"]))
        evaluation_record = run_evaluation.evaluate_combination(
            config,
            tools["dataset"],
            tools["dataset_path"],
            combination,
            run_dir,
            tools["server"],
        )
        row["step3_evaluation_status"] = evaluation_record["status"]
        add_evaluation_metrics(row, evaluation_record, run_dir)
    except Exception as error:
        row["step3_evaluation_status"] = "failed"
        failures.append(failure_event(combination, "step3_evaluation", error))

    benchmark_record: dict[str, Any] | None = None
    try:
        if "benchmark_error" in tools:
            raise run_benchmarks.BenchmarkError(str(tools["benchmark_error"]))
        benchmark_record = run_benchmarks.benchmark_combination(
            tools["bench"], combination, config, run_dir
        )
        row["step4_benchmark_status"] = benchmark_record["status"]
        add_benchmark_metrics(row, benchmark_record)
    except Exception as error:
        row["step4_benchmark_status"] = "failed"
        failures.append(failure_event(combination, "step4_benchmark", error))

    failed_stages = [failure["stage"] for failure in failures]
    if row["step3_evaluation_status"] == "partial_failure":
        failed_stages.append("step3_evaluation:partial")
    if row["step4_benchmark_status"] == "partial_failure":
        failed_stages.append("step4_benchmark:partial")
    row["failed_stage"] = ";".join(failed_stages)
    row["error"] = " | ".join(failure["error"] for failure in failures)
    row["status"] = "success" if not failed_stages else "partial_failure"
    row["finished_at_utc"] = utc_now()
    return row, failures


def select_combinations(
    combinations: list[dict[str, str]], only: list[str] | None
) -> list[dict[str, str]]:
    if not only:
        return combinations
    selectors: set[tuple[str, str]] = set()
    for value in only:
        model, separator, quantization = value.partition(":")
        if not separator or not model or not quantization:
            raise ValueError(f"--only는 MODEL:QUANTIZATION 형식이어야 합니다: {value}")
        selectors.add((model, quantization))
    known = {combination_key(item) for item in combinations}
    unknown = selectors - known
    if unknown:
        rendered = ", ".join(f"{model}:{quant}" for model, quant in sorted(unknown))
        raise ValueError(f"설정에 없는 --only 조합: {rendered}")
    return [item for item in combinations if combination_key(item) in selectors]


def run_pipeline(
    config: dict[str, Any],
    run_name: str | None,
    resume: bool,
    verbose: bool,
    only: list[str] | None = None,
) -> int:
    combinations = select_combinations(experiment_matrix(config), only)
    run_dir, resumed = resolve_run_directory(config, run_name, resume)
    if resumed:
        assert_resume_config_matches(config, run_dir)
    setup_logging(run_dir / "pipeline.log", verbose)
    csv_path = run_dir / "results.csv"
    state_path = run_dir / "pipeline-state.json"
    failures_path = run_dir / "failures.jsonl"
    fields = result_fields(config["benchmark"]["prompt_lengths"])
    existing_rows = read_csv(csv_path) if resumed else []
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {
        combination_key(row): row for row in existing_rows
    }
    already_complete = completed_keys(existing_rows)
    tools = resolve_tools(config)
    config_hash = hashlib.sha256(Path(config["_config_path"]).read_bytes()).hexdigest()
    state: dict[str, Any] = {
        "schema_version": 1,
        "run_directory": str(run_dir.resolve()),
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "resumed": resumed,
        "config_sha256": config_hash,
        "total_combinations": len(combinations),
        "processed_this_run": 0,
        "skipped_by_resume": 0,
        "current_combination": None,
        "csv_path": str(csv_path.resolve()),
    }
    write_json(state_path, state)

    if "dataset_error" in tools:
        error = tools["dataset_error"]
        failure = {
            "occurred_at_utc": utc_now(),
            "stage": "dataset_preflight",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        append_jsonl(failures_path, failure)
        state["finished_at_utc"] = utc_now()
        state["status"] = "failed_preflight"
        state["error"] = str(error)
        write_json(state_path, state)
        print(f"평가셋 사전 검증 실패: {error}", flush=True)
        return 1

    if "dataset_lock" in tools:
        write_json(run_dir / "dataset-lock-verification.json", tools["dataset_lock"])

    interrupted = False
    for index, combination in enumerate(combinations, start=1):
        key = combination_key(combination)
        label = f"{combination['model']} / {combination['quantization']}"
        if resume and key in already_complete:
            print(f"[{index}/{len(combinations)}] {label} - 완료됨, 건너뜀", flush=True)
            state["skipped_by_resume"] += 1
            continue

        print(f"[{index}/{len(combinations)}] {label} - 시작", flush=True)
        state["current_combination"] = {
            "index": index,
            "model": combination["model"],
            "quantization": combination["quantization"],
        }
        write_json(state_path, state)
        try:
            row, failures = process_combination(
                index, config, combination, tools, run_dir
            )
        except KeyboardInterrupt:
            interrupted = True
            row = {
                "combination_index": index,
                "model": combination["model"],
                "quantization": combination["quantization"],
                "status": "interrupted",
                "failed_stage": "user_interrupt",
                "error": "사용자가 실행을 중단했습니다.",
                "finished_at_utc": utc_now(),
            }
            failures = []
        except Exception as error:
            row = {
                "combination_index": index,
                "model": combination["model"],
                "quantization": combination["quantization"],
                "status": "failed",
                "failed_stage": "pipeline_internal",
                "error": str(error),
                "finished_at_utc": utc_now(),
            }
            failures = [failure_event(combination, "pipeline_internal", error)]

        for failure in failures:
            append_jsonl(failures_path, failure)
            LOGGER.error(
                "[%s][%s][%s] %s",
                failure["model"],
                failure["quantization"],
                failure["stage"],
                failure["error"],
            )
        rows_by_key[key] = row
        write_csv_atomic(
            csv_path, ordered_rows(rows_by_key, combinations), fields
        )
        state["processed_this_run"] += 1
        state["current_combination"] = None
        write_json(state_path, state)
        print(
            f"[{index}/{len(combinations)}] {label} - {row['status']} (CSV 저장 완료)",
            flush=True,
        )
        if interrupted:
            break

    final_rows = ordered_rows(rows_by_key, combinations)
    successful = sum(row.get("status") == "success" for row in final_rows)
    state["finished_at_utc"] = utc_now()
    state["successful_combinations"] = successful
    state["recorded_combinations"] = len(final_rows)
    if interrupted:
        state["status"] = "interrupted"
    elif successful == len(combinations):
        state["status"] = "success"
    else:
        state["status"] = "partial_failure"
    write_json(state_path, state)
    print(
        f"완료: 성공 {successful}/{len(combinations)}, 결과 {csv_path}",
        flush=True,
    )
    return 130 if interrupted else (0 if state["status"] == "success" else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GGUF 준비, 평가, 벤치마크, CSV 집계를 순서대로 실행합니다."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--run-name", help="신규 또는 재개할 results 하위 디렉터리 이름")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="지정 실행 또는 가장 최근 pipeline-* 실행의 성공 조합을 건너뜀",
    )
    parser.add_argument("--verbose", action="store_true", help="상세 로그를 콘솔에도 표시")
    parser.add_argument(
        "--only",
        action="append",
        metavar="MODEL:QUANTIZATION",
        help="지정 조합만 실행(여러 번 지정 가능, 파일럿용)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    raise SystemExit(
        run_pipeline(config, args.run_name, args.resume, args.verbose, args.only)
    )


if __name__ == "__main__":
    main()
