from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, load_config
from scoring import EvaluationError, load_dataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_counts(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for part in value.split(","):
        key, separator, count = part.partition(":")
        if not separator:
            raise ValueError("유형별 문항 수는 1:24,2:12 형식이어야 합니다.")
        result[int(key.strip())] = int(count.strip())
    if set(result) != {1, 2, 3, 4} or any(count < 0 for count in result.values()):
        raise ValueError("유형 1~4의 0 이상 문항 수를 모두 지정해야 합니다.")
    return result


def validate_artifacts(
    dataset_path: Path, expected_counts: dict[int, int]
) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    records = load_dataset(dataset_path)
    actual_counts = Counter(int(record["type"]) for record in records)
    issues = []
    for item_type in range(1, 5):
        expected = expected_counts[item_type]
        actual = actual_counts[item_type]
        if actual != expected:
            issues.append(f"유형 {item_type}: 기대 {expected}문항, 실제 {actual}문항")

    dataset_root = dataset_path.parent.resolve()
    documents: dict[str, dict[str, Any]] = {}
    for record in records:
        relative = Path(str(record["context_doc"]))
        if relative.is_absolute():
            issues.append(f"{record['id']}: context_doc는 상대 경로여야 합니다.")
            continue
        document_path = (dataset_root / relative).resolve()
        if document_path != dataset_root and dataset_root not in document_path.parents:
            issues.append(f"{record['id']}: context_doc가 평가셋 디렉터리를 벗어납니다.")
            continue
        key = relative.as_posix()
        if key in documents:
            continue
        if not document_path.is_file():
            issues.append(f"{record['id']}: 참조 문서가 없습니다: {key}")
            continue
        try:
            text = document_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            issues.append(f"{record['id']}: UTF-8 문서를 읽지 못했습니다: {key}: {error}")
            continue
        documents[key] = {
            "path": key,
            "size_bytes": document_path.stat().st_size,
            "characters": len(text),
            "sha256": sha256(document_path),
        }

    return {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "status": "success" if not issues else "failed",
        "dataset": {
            "path": dataset_path.name,
            "sha256": sha256(dataset_path),
            "questions": len(records),
            "expected_type_counts": {str(k): v for k, v in expected_counts.items()},
            "actual_type_counts": {
                str(item_type): actual_counts[item_type] for item_type in range(1, 5)
            },
        },
        "documents": documents,
        "issues": issues,
    }


def assert_dataset_lock(
    dataset_path: Path,
    lock_path: Path,
    configured_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """현재 평가셋과 문서가 기존 잠금 파일의 해시와 같은지 확인합니다."""
    dataset_path = dataset_path.resolve()
    lock_path = lock_path.resolve()
    if not lock_path.is_file():
        raise ValueError(f"평가셋 잠금 파일이 없습니다: {lock_path}")
    try:
        locked = json.loads(lock_path.read_text(encoding="utf-8-sig"))
        expected_counts = {
            int(key): int(value)
            for key, value in locked["dataset"]["expected_type_counts"].items()
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"평가셋 잠금 파일 형식이 잘못됐습니다: {lock_path}") from error
    if locked.get("status") != "success":
        raise ValueError("실패 상태의 평가셋 잠금 파일은 사용할 수 없습니다.")
    if configured_counts is not None and configured_counts != expected_counts:
        raise ValueError(
            "설정의 유형별 문항 수가 잠금 파일과 다릅니다: "
            f"config={configured_counts}, lock={expected_counts}"
        )

    current = validate_artifacts(dataset_path, expected_counts)
    mismatches = list(current["issues"])
    if current["dataset"]["sha256"] != locked["dataset"].get("sha256"):
        mismatches.append("평가셋 SHA-256이 잠금 이후 변경됐습니다.")
    locked_documents = locked.get("documents", {})
    if set(current["documents"]) != set(locked_documents):
        mismatches.append("참조 문서 목록이 잠금 이후 변경됐습니다.")
    else:
        for name, document in current["documents"].items():
            if document["sha256"] != locked_documents[name].get("sha256"):
                mismatches.append(f"참조 문서 SHA-256이 변경됐습니다: {name}")
    if mismatches:
        raise ValueError("평가셋 잠금 검증 실패: " + " | ".join(mismatches))
    return current


def main() -> None:
    parser = argparse.ArgumentParser(
        description="평가셋 구성과 참조 문서를 검증하고 재현성 잠금 파일을 만듭니다."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--dataset", help="설정의 evaluation.dataset 대체")
    parser.add_argument("--expected-counts", help="예: 1:24,2:12,3:12,4:12")
    parser.add_argument("--output", help="검증 JSON 또는 잠금 파일 위치")
    parser.add_argument(
        "--verify-lock",
        action="store_true",
        help="잠금 파일을 다시 쓰지 않고 현재 평가셋·문서 해시를 검증",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    root = Path(config["_project_root"])
    dataset_value = args.dataset or config["evaluation"]["dataset"]
    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    configured_counts = config["evaluation"].get("expected_type_counts", {})
    try:
        expected_counts = (
            parse_counts(args.expected_counts)
            if args.expected_counts
            else {int(key): int(value) for key, value in configured_counts.items()}
        )
        if set(expected_counts) != {1, 2, 3, 4}:
            raise ValueError("evaluation.expected_type_counts에 유형 1~4가 모두 필요합니다.")
        if args.verify_lock:
            configured_lock = config["evaluation"].get(
                "dataset_lock", str(dataset_path.parent / "dataset.lock.json")
            )
            lock_path = Path(args.output or configured_lock)
            if not lock_path.is_absolute():
                lock_path = root / lock_path
            report = assert_dataset_lock(dataset_path, lock_path, expected_counts)
            print(f"평가셋 잠금 검증: success ({lock_path.resolve()})")
            return
        report = validate_artifacts(dataset_path, expected_counts)
    except (EvaluationError, OSError, ValueError) as error:
        parser.error(str(error))

    output = Path(args.output) if args.output else dataset_path.parent / "dataset.lock.json"
    if not output.is_absolute():
        output = root / output
    write_json_atomic(output.resolve(), report)
    for issue in report["issues"]:
        print(f"[failed] {issue}")
    print(
        f"평가셋 검증: {report['status']} / {report['dataset']['questions']}문항 / "
        f"{len(report['documents'])}문서 ({output.resolve()})"
    )
    raise SystemExit(0 if report["status"] == "success" else 1)


if __name__ == "__main__":
    main()
