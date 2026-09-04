from __future__ import annotations

# huggingface_hub는 환경 변수를 import 시점에 읽으므로 반드시 먼저 설정합니다.
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from common import DEFAULT_CONFIG, create_run_directory, load_config, resolve_paths


LOGGER = logging.getLogger("prepare_models")
DOWNLOAD_MARKER = ".law-q-download.json"


class StageError(RuntimeError):
    """모델 준비 단계가 실패했을 때 발생합니다."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_info(path: Path) -> dict[str, Any]:
    """산출물의 디스크 실제 크기를 바이트와 GiB로 반환합니다."""
    size = path.stat().st_size
    return {
        "path": str(path.resolve()),
        "size_bytes": size,
        "size_gib": round(size / (1024**3), 4),
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, value: dict[str, Any]) -> None:
    """중단 시에도 기존 결과가 보존되도록 JSON을 원자적으로 갱신합니다."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """실패 이벤트를 즉시 디스크에 추가하여 후속 장애에도 보존합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def persist_failure(
    failures_path: Path,
    report: dict[str, Any],
    model: str,
    stage: str,
    stage_record: dict[str, Any],
) -> dict[str, Any]:
    """실패를 종합 결과와 append-only 실패 로그 양쪽에 기록합니다."""
    failure = {
        "occurred_at_utc": utc_now(),
        "model": model,
        "stage": stage,
        "error_type": stage_record.get("error_type"),
        "error": stage_record.get("error"),
        "traceback": stage_record.get("traceback"),
        "prepare_log": "prepare.log",
    }
    report.setdefault("failures", []).append(failure)
    append_jsonl(failures_path, failure)
    return failure


def setup_logging(log_path: Path, verbose: bool) -> None:
    LOGGER.setLevel(logging.DEBUG)
    for handler in LOGGER.handlers:
        handler.close()
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def run_command(command: list[str], model: str, stage: str) -> None:
    """외부 명령 출력을 모델/단계 문맥과 함께 실시간 로그에 기록합니다."""
    LOGGER.info("[%s][%s] 실행: %s", model, stage, subprocess.list2cmdline(command))
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise StageError(f"명령을 시작할 수 없습니다: {error}") from error

    assert process.stdout is not None
    for line in process.stdout:
        LOGGER.info("[%s][%s] %s", model, stage, line.rstrip())
    returncode = process.wait()
    if returncode != 0:
        raise StageError(f"명령이 종료 코드 {returncode}로 실패했습니다.")


def existing_artifact(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _remove_partial(path: Path, model: str, stage: str) -> None:
    if path.exists():
        LOGGER.warning("[%s][%s] 이전 미완성 파일 제거: %s", model, stage, path)
        path.unlink()


def find_quantizer(config: dict[str, Any]) -> Path:
    paths = resolve_paths(config)
    root = paths["llama_cpp"]
    configured = paths.get("llama_quantize")
    candidates = [
        configured,
        root / "build" / "bin" / "Release" / "llama-quantize.exe",
        root / "build" / "bin" / "llama-quantize.exe",
        root / "build" / "Release" / "bin" / "llama-quantize.exe",
        root / "llama-quantize.exe",
        root / "build" / "bin" / "llama-quantize",
        root / "llama-quantize",
    ]
    unique_candidates = list(dict.fromkeys(item for item in candidates if item is not None))
    for candidate in unique_candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(item) for item in unique_candidates)
    raise StageError(f"llama-quantize 실행 파일을 찾지 못했습니다:\n  - {searched}")


def validate_llama_cpp(config: dict[str, Any]) -> tuple[Path, Path]:
    root = resolve_paths(config)["llama_cpp"]
    converter = root / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise StageError(f"변환 스크립트를 찾지 못했습니다: {converter}")
    return converter, find_quantizer(config)


def _matching_download_marker(marker: Path, model: dict[str, Any]) -> dict[str, Any] | None:
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        value.get("hf_repo") == model["hf_repo"]
        and value.get("requested_revision") == str(model["revision"])
    ):
        return value
    return None


def download_model(model: dict[str, Any], model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    stage = "download"
    marker_path = model_dir / DOWNLOAD_MARKER
    marker = _matching_download_marker(marker_path, model)
    if marker is not None and (model_dir / "config.json").is_file():
        LOGGER.info("[%s][%s] 완료된 다운로드가 있어 건너뜁니다: %s", model["name"], stage, model_dir)
        return {
            "status": "skipped",
            "path": str(model_dir.resolve()),
            "directory_size_bytes": directory_size(model_dir),
            "resolved_revision": marker.get("resolved_revision"),
        }

    if marker_path.exists() and marker is None:
        raise StageError(
            f"기존 다운로드 마커의 저장소 또는 revision이 현재 설정과 다릅니다: {marker_path}. "
            "모델 name을 변경하거나 기존 디렉터리를 확인해 주세요."
        )

    LOGGER.info(
        "[%s][%s] Hugging Face 다운로드 시작: %s @ %s (hf_transfer=%s)",
        model["name"],
        stage,
        model["hf_repo"],
        model["revision"],
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"],
    )
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise StageError(
            "huggingface_hub/hf_transfer가 설치되지 않았습니다. "
            "pip install -r requirements.txt를 실행해 주세요."
        ) from error

    token = os.environ.get("HF_TOKEN") or None
    download_config = config.get("download", {})
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = HfApi().model_info(
            repo_id=model["hf_repo"], revision=str(model["revision"]), token=token
        )
        snapshot_download(
            repo_id=model["hf_repo"],
            revision=str(model["revision"]),
            repo_type="model",
            local_dir=model_dir,
            token=token,
            max_workers=int(download_config.get("max_workers", 8)),
            ignore_patterns=download_config.get("ignore_patterns"),
        )
    except Exception as error:
        raise StageError(f"Hugging Face 다운로드 실패: {type(error).__name__}: {error}") from error

    if not (model_dir / "config.json").is_file():
        raise StageError(f"다운로드 후 config.json을 찾지 못했습니다: {model_dir}")

    marker = {
        "hf_repo": model["hf_repo"],
        "requested_revision": str(model["revision"]),
        "resolved_revision": info.sha,
        "completed_at_utc": utc_now(),
    }
    write_json(marker_path, marker)
    size = directory_size(model_dir)
    LOGGER.info("[%s][%s] 다운로드 완료: %s bytes", model["name"], stage, size)
    return {
        "status": "success",
        "path": str(model_dir.resolve()),
        "directory_size_bytes": size,
        "resolved_revision": info.sha,
    }


def convert_model(
    model: dict[str, Any], model_dir: Path, output: Path, converter: Path
) -> dict[str, Any]:
    stage = "convert_f16"
    if existing_artifact(output):
        LOGGER.info("[%s][%s] 기존 파일이 있어 건너뜁니다: %s", model["name"], stage, output)
        return {"status": "skipped", "artifact": artifact_info(output)}

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    _remove_partial(partial, model["name"], stage)
    command = [
        sys.executable,
        str(converter),
        str(model_dir),
        "--outfile",
        str(partial),
        "--outtype",
        "f16",
    ]
    try:
        run_command(command, model["name"], stage)
        if not existing_artifact(partial):
            raise StageError(f"명령은 성공했지만 출력 파일이 생성되지 않았습니다: {partial}")
        partial.replace(output)
    except Exception:
        _remove_partial(partial, model["name"], stage)
        raise

    result = {"status": "success", "artifact": artifact_info(output)}
    LOGGER.info(
        "[%s][%s] F16 변환 완료: %s bytes",
        model["name"],
        stage,
        result["artifact"]["size_bytes"],
    )
    return result


def quantize_model(
    model: dict[str, Any],
    f16_path: Path,
    output: Path,
    quantization: str,
    quantize_type: str,
    quantizer: Path,
    threads: int,
) -> dict[str, Any]:
    stage = f"quantize:{quantization}"
    if existing_artifact(output):
        LOGGER.info("[%s][%s] 기존 파일이 있어 건너뜁니다: %s", model["name"], stage, output)
        return {"status": "skipped", "artifact": artifact_info(output)}

    partial = output.with_name(output.stem + ".partial" + output.suffix)
    _remove_partial(partial, model["name"], stage)
    command = [
        str(quantizer),
        str(f16_path),
        str(partial),
        quantize_type,
        str(threads),
    ]
    try:
        run_command(command, model["name"], stage)
        if not existing_artifact(partial):
            raise StageError(f"명령은 성공했지만 출력 파일이 생성되지 않았습니다: {partial}")
        partial.replace(output)
    except Exception:
        _remove_partial(partial, model["name"], stage)
        raise

    result = {"status": "success", "artifact": artifact_info(output)}
    LOGGER.info(
        "[%s][%s] 양자화 완료: %s bytes",
        model["name"],
        stage,
        result["artifact"]["size_bytes"],
    )
    return result


def execute_stage(
    model_name: str,
    stage: str,
    action: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    started = time.perf_counter()
    started_at = utc_now()
    try:
        record = action()
        failed = False
    except Exception as error:
        LOGGER.exception("[%s][%s] 실패: %s", model_name, stage, error)
        record = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failed = True
    record["started_at_utc"] = started_at
    record["duration_seconds"] = round(time.perf_counter() - started, 3)
    return record, failed


def blocked_stage(reason: str) -> dict[str, Any]:
    return {"status": "blocked", "reason": reason}


def dry_run(config: dict[str, Any]) -> None:
    paths = resolve_paths(config)
    print(f"HF_HUB_ENABLE_HF_TRANSFER={os.environ['HF_HUB_ENABLE_HF_TRANSFER']}")
    print(f"models={paths['models']}")
    print(f"gguf={paths['gguf']}")
    print(f"llama.cpp={paths['llama_cpp']}")
    total = 0
    for model in config["models"]:
        print(f"{model['name']}: {model['hf_repo']} @ {model['revision']}")
        for quantization in model["quantizations"]:
            print(f"  - {quantization}")
            total += 1
    print(f"총 실험 조합: {total}개")


def run_pipeline(config: dict[str, Any], run_name: str | None, verbose: bool) -> int:
    run_dir = create_run_directory(config, run_name)
    report_path = run_dir / "model-preparation.json"
    failures_path = run_dir / "failures.jsonl"
    setup_logging(run_dir / "prepare.log", verbose)
    LOGGER.info("모델 준비 실행 디렉터리: %s", run_dir)

    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "hf_transfer_enabled": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1",
        "models": [],
        "failures": [],
    }
    write_json(report_path, report)

    try:
        converter, quantizer = validate_llama_cpp(config)
    except Exception as error:
        LOGGER.exception("[preflight][llama.cpp] 실패: %s", error)
        preflight_record = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        persist_failure(
            failures_path, report, "__pipeline__", "preflight:llama.cpp", preflight_record
        )
        report["status"] = "failed"
        report["preflight_error"] = str(error)
        report["finished_at_utc"] = utc_now()
        write_json(report_path, report)
        return 1

    paths = resolve_paths(config)
    presets = config["quantization"]["presets"]
    threads = int(config["runtime"].get("threads", os.cpu_count() or 1))
    any_failure = False

    for model in config["models"]:
        model_name = model["name"]
        model_dir = paths["models"] / model_name
        output_dir = paths["gguf"] / model_name
        f16_path = output_dir / f"{model_name}-F16.gguf"
        model_record: dict[str, Any] = {
            "name": model_name,
            "hf_repo": model["hf_repo"],
            "requested_revision": str(model["revision"]),
            "status": "running",
            "stages": {},
            "quantizations": [],
        }
        report["models"].append(model_record)
        write_json(report_path, report)

        download, failed = execute_stage(
            model_name,
            "download",
            lambda model=model, model_dir=model_dir: download_model(model, model_dir, config),
        )
        model_record["stages"]["download"] = download
        if failed:
            any_failure = True
            persist_failure(failures_path, report, model_name, "download", download)
            model_record["stages"]["convert_f16"] = blocked_stage("download 단계 실패")
            for quantization in model["quantizations"]:
                model_record["quantizations"].append(
                    {"name": quantization, **blocked_stage("download 단계 실패")}
                )
            model_record["status"] = "failed"
            write_json(report_path, report)
            continue

        conversion, failed = execute_stage(
            model_name,
            "convert_f16",
            lambda model=model, model_dir=model_dir, f16_path=f16_path: convert_model(
                model, model_dir, f16_path, converter
            ),
        )
        model_record["stages"]["convert_f16"] = conversion
        if failed:
            any_failure = True
            persist_failure(failures_path, report, model_name, "convert_f16", conversion)
            for quantization in model["quantizations"]:
                model_record["quantizations"].append(
                    {"name": quantization, **blocked_stage("convert_f16 단계 실패")}
                )
            model_record["status"] = "failed"
            write_json(report_path, report)
            continue

        model_failed = False
        for quantization in model["quantizations"]:
            quantize_type = presets[quantization]
            output = output_dir / f"{model_name}-{quantization}.gguf"
            quantized, failed = execute_stage(
                model_name,
                f"quantize:{quantization}",
                lambda model=model, output=output, quantization=quantization, quantize_type=quantize_type: quantize_model(
                    model,
                    f16_path,
                    output,
                    quantization,
                    quantize_type,
                    quantizer,
                    threads,
                ),
            )
            model_record["quantizations"].append({"name": quantization, **quantized})
            if failed:
                any_failure = True
                model_failed = True
                persist_failure(
                    failures_path,
                    report,
                    model_name,
                    f"quantize:{quantization}",
                    quantized,
                )
            write_json(report_path, report)

        model_record["status"] = "failed" if model_failed else "success"
        write_json(report_path, report)

    report["status"] = "partial_failure" if any_failure else "success"
    report["finished_at_utc"] = utc_now()
    write_json(report_path, report)
    LOGGER.info("모델 준비 종료: %s", report["status"])
    return 1 if any_failure else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="설정된 Hugging Face 모델을 다운로드하고 GGUF로 변환·양자화합니다."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--run-name", help="results 아래에 생성할 실행 이름")
    parser.add_argument("--verbose", action="store_true", help="상세 로그를 콘솔에도 표시")
    parser.add_argument(
        "--dry-run", action="store_true", help="다운로드나 변환 없이 실험 계획만 출력"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        dry_run(config)
        return
    raise SystemExit(run_pipeline(config, args.run_name, args.verbose))


if __name__ == "__main__":
    main()
