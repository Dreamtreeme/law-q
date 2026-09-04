from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiment.yaml"


class ConfigError(ValueError):
    """실험 설정이 유효하지 않을 때 발생합니다."""


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """YAML 설정을 읽고 필수 필드 및 실험 조합을 검증합니다."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError("설정 파일의 최상위 값은 매핑이어야 합니다.")

    for key in ("paths", "runtime", "quantization", "models"):
        if key not in config:
            raise ConfigError(f"필수 설정이 없습니다: {key}")

    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ConfigError("models는 하나 이상의 모델 목록이어야 합니다.")

    presets = config["quantization"].get("presets", {})
    seen_names: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise ConfigError("각 models 항목은 매핑이어야 합니다.")
        for key in ("name", "hf_repo", "revision", "quantizations"):
            if key not in model:
                raise ConfigError(f"모델 항목에 필수 값이 없습니다: {key}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(model["name"])):
            raise ConfigError(
                f"모델 name에는 영문자, 숫자, 점, 밑줄, 하이픈만 사용할 수 있습니다: "
                f"{model['name']}"
            )
        if model["name"] in seen_names:
            raise ConfigError(f"중복 모델 이름: {model['name']}")
        seen_names.add(model["name"])
        if not model["quantizations"]:
            raise ConfigError(f"양자화 수준이 비어 있습니다: {model['name']}")
        unknown = set(model["quantizations"]) - set(presets)
        if unknown:
            raise ConfigError(
                f"{model['name']}에 정의되지 않은 양자화 프리셋이 있습니다: "
                f"{', '.join(sorted(unknown))}"
            )

    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent)
    return config


def resolve_paths(config: dict[str, Any]) -> dict[str, Path]:
    """설정의 상대 경로를 설정 파일 기준 절대 경로로 변환합니다."""
    base = Path(config["_project_root"])
    return {
        name: (base / value).resolve() if not Path(value).is_absolute() else Path(value)
        for name, value in config["paths"].items()
    }


def experiment_matrix(config: dict[str, Any]) -> list[dict[str, str]]:
    """모델과 양자화 수준의 실행 가능한 평면 목록을 반환합니다."""
    presets = config["quantization"]["presets"]
    return [
        {
            "model": model["name"],
            "hf_repo": model["hf_repo"],
            "revision": str(model["revision"]),
            "quantization": quantization,
            "quantize_type": presets[quantization],
        }
        for model in config["models"]
        for quantization in model["quantizations"]
    ]


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    """환경 조회 명령을 실패 허용 방식으로 실행합니다."""
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return {
            "ok": completed.returncode == 0,
            "value": completed.stdout.strip() or completed.stderr.strip(),
            "returncode": completed.returncode,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "value": str(error), "returncode": None}


def _git_state(repo: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    dirty = _run(["git", "status", "--porcelain"], cwd=repo)
    return {
        "path": str(repo),
        "commit": commit["value"] if commit["ok"] else None,
        "dirty": bool(dirty["value"]) if dirty["ok"] else None,
        "error": None if commit["ok"] else commit["value"],
    }


def _gpu_info() -> dict[str, Any]:
    query = (
        "name,driver_version,memory.total,compute_cap"
    )
    result = _run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result["ok"]:
        return {"available": False, "gpus": [], "error": result["value"]}

    gpus = []
    for index, line in enumerate(result["value"].splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            gpus.append(
                {
                    "index": index,
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total_mib": int(parts[2]),
                    "compute_capability": parts[3],
                }
            )
    return {"available": True, "gpus": gpus, "error": None}


def capture_environment(config: dict[str, Any]) -> dict[str, Any]:
    """재현성에 필요한 실행 환경 정보를 수집합니다."""
    now_utc = datetime.now(timezone.utc)
    paths = resolve_paths(config)
    config_path = Path(config["_config_path"])
    config_bytes = config_path.read_bytes()

    return {
        "captured_at_utc": now_utc.isoformat(),
        "captured_at_local": now_utc.astimezone().isoformat(),
        "timezone": str(now_utc.astimezone().tzinfo),
        "config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "experiment_count": len(experiment_matrix(config)),
        },
        "llama_cpp": _git_state(paths["llama_cpp"]),
        "project_git": _git_state(Path(config["_project_root"])),
        "gpu": _gpu_info(),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


def create_run_directory(
    config: dict[str, Any], run_name: str | None = None
) -> Path:
    """결과 폴더를 만들고 환경 정보와 설정 스냅샷을 기록합니다."""
    results_dir = resolve_paths(config)["results"]
    run_id = run_name or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = capture_environment(config)
    with (run_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    shutil.copy2(config["_config_path"], run_dir / "experiment.yaml")
    return run_dir
