from __future__ import annotations

import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCORING_BY_TYPE = {
    1: "keyword_any",
    2: "keyword_ratio",
    3: "json_field",
    4: "refusal",
}


class EvaluationError(ValueError):
    """평가셋이나 예측 결과 형식이 올바르지 않을 때 발생합니다."""


def normalize_text(value: str) -> str:
    """유니코드·대소문자·공백 차이를 제거해 포함 여부를 안정적으로 비교합니다."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", "", normalized)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """JSONL 또는 JSON 배열 파일을 읽습니다."""
    source = Path(path)
    try:
        if source.suffix.casefold() == ".jsonl":
            records: list[dict[str, Any]] = []
            with source.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise EvaluationError(
                            f"{source}:{line_number} JSON 파싱 실패: {error.msg}"
                        ) from error
                    if not isinstance(record, dict):
                        raise EvaluationError(
                            f"{source}:{line_number} 각 레코드는 JSON 객체여야 합니다."
                        )
                    records.append(record)
            return records

        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except OSError as error:
        raise EvaluationError(f"파일을 읽을 수 없습니다: {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise EvaluationError(f"{source} JSON 파싱 실패: {error.msg}") from error

    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise EvaluationError(f"{source}는 JSON 객체 배열이어야 합니다.")
    return value


def validate_dataset(records: list[dict[str, Any]]) -> None:
    required = {
        "id",
        "type",
        "context_doc",
        "question",
        "answer_keywords",
        "scoring",
    }
    seen_ids: set[str] = set()
    for index, item in enumerate(records, start=1):
        missing = required - set(item)
        if missing:
            raise EvaluationError(
                f"평가 항목 {index}에 필수 필드가 없습니다: {', '.join(sorted(missing))}"
            )
        item_id = item["id"]
        if not isinstance(item_id, str) or not re.fullmatch(r"L[0-9]{3,}", item_id):
            raise EvaluationError(f"평가 항목 {index}의 id 형식이 잘못됐습니다: {item_id!r}")
        if item_id in seen_ids:
            raise EvaluationError(f"중복 평가 id: {item_id}")
        seen_ids.add(item_id)

        item_type = item["type"]
        if item_type not in SCORING_BY_TYPE:
            raise EvaluationError(f"{item_id}: type은 1~4여야 합니다.")
        scoring = item["scoring"]
        expected_scoring = SCORING_BY_TYPE[item_type]
        if scoring != expected_scoring:
            raise EvaluationError(
                f"{item_id}: type {item_type}의 scoring은 {expected_scoring}이어야 합니다."
            )
        if not isinstance(item["context_doc"], str) or not item["context_doc"].strip():
            raise EvaluationError(f"{item_id}: context_doc는 빈 문자열일 수 없습니다.")
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise EvaluationError(f"{item_id}: question은 빈 문자열일 수 없습니다.")

        keywords = item["answer_keywords"]
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) and keyword for keyword in keywords
        ):
            raise EvaluationError(f"{item_id}: answer_keywords는 문자열 배열이어야 합니다.")
        if len(keywords) != len(set(keywords)):
            raise EvaluationError(f"{item_id}: answer_keywords에 중복 값이 있습니다.")
        if scoring != "json_field" and not keywords:
            raise EvaluationError(f"{item_id}: {scoring}에는 answer_keywords가 필요합니다.")

        if scoring == "json_field":
            fields = item.get("answer_fields")
            if not isinstance(fields, dict) or not fields:
                raise EvaluationError(f"{item_id}: json_field에는 answer_fields가 필요합니다.")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    records = read_records(path)
    if not records:
        raise EvaluationError("평가셋이 비어 있습니다.")
    validate_dataset(records)
    return records


def load_predictions(path: str | Path) -> dict[str, str]:
    records = read_records(path)
    predictions: dict[str, str] = {}
    for index, item in enumerate(records, start=1):
        item_id = item.get("id")
        response = item.get("response")
        if not isinstance(item_id, str) or not item_id:
            raise EvaluationError(f"예측 항목 {index}: id가 필요합니다.")
        if item_id in predictions:
            raise EvaluationError(f"중복 예측 id: {item_id}")
        if not isinstance(response, str):
            raise EvaluationError(f"예측 {item_id}: response는 문자열이어야 합니다.")
        predictions[item_id] = response
    return predictions


def keyword_matches(response: str, keywords: list[str]) -> list[str]:
    normalized_response = normalize_text(response)
    return [
        keyword
        for keyword in keywords
        if normalize_text(keyword) in normalized_response
    ]


def _strip_json_fence(response: str) -> str:
    stripped = response.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else stripped


def parse_json_response(response: str) -> Any:
    return json.loads(
        _strip_json_fence(response),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"허용되지 않는 JSON 상수: {value}")
        ),
    )


def get_field(value: dict[str, Any], path: str) -> tuple[bool, Any]:
    """직접 키를 우선하고, 없으면 점 표기 중첩 경로를 조회합니다."""
    if path in value:
        return True, value[path]
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isfinite(float(expected)) and actual == expected
    if isinstance(actual, str) and isinstance(expected, str):
        return normalize_text(actual) == normalize_text(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            values_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            values_equal(actual[key], expected[key]) for key in expected
        )
    return actual == expected


def score_item(item: dict[str, Any], response: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": item["id"],
        "type": item["type"],
        "scoring": item["scoring"],
        "score": 0.0,
        "missing_prediction": response is None,
        "json_parse_failed": False,
    }
    if response is None:
        result["reason"] = "prediction_missing"
        return result

    scoring = item["scoring"]
    if scoring in {"keyword_any", "refusal"}:
        matched = keyword_matches(response, item["answer_keywords"])
        result["score"] = 1.0 if matched else 0.0
        result["matched_keywords"] = matched
        result["total_keywords"] = len(item["answer_keywords"])
        return result

    if scoring == "keyword_ratio":
        matched = keyword_matches(response, item["answer_keywords"])
        total = len(item["answer_keywords"])
        result["score"] = round(len(matched) / total, 6)
        result["matched_keywords"] = matched
        result["matched_count"] = len(matched)
        result["total_keywords"] = total
        return result

    try:
        parsed = parse_json_response(response)
    except (json.JSONDecodeError, ValueError) as error:
        result["json_parse_failed"] = True
        result["reason"] = "json_parse_failed"
        result["parse_error"] = str(error)
        return result

    if not isinstance(parsed, dict):
        result["reason"] = "json_root_not_object"
        result["field_results"] = []
        result["matched_fields"] = 0
        result["total_fields"] = len(item["answer_fields"])
        return result

    field_results = []
    for field, expected in item["answer_fields"].items():
        exists, actual = get_field(parsed, field)
        matched = exists and values_equal(actual, expected)
        field_results.append(
            {
                "field": field,
                "matched": matched,
                "exists": exists,
                "expected": expected,
                "actual": actual if exists else None,
            }
        )
    matched_count = sum(field["matched"] for field in field_results)
    result["score"] = round(matched_count / len(field_results), 6)
    result["field_results"] = field_results
    result["matched_fields"] = matched_count
    result["total_fields"] = len(field_results)
    return result


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    total_score = sum(float(result["score"]) for result in results)
    return {
        "count": count,
        "total_score": round(total_score, 6),
        "average_score": round(total_score / count, 6) if count else None,
        "percent": round(total_score / count * 100, 2) if count else None,
        "full_credit_count": sum(result["score"] == 1.0 for result in results),
        "zero_score_count": sum(result["score"] == 0.0 for result in results),
        "missing_predictions": sum(result["missing_prediction"] for result in results),
        "json_parse_failures": sum(result["json_parse_failed"] for result in results),
    }


def score_dataset(
    dataset: list[dict[str, Any]], predictions: dict[str, str]
) -> dict[str, Any]:
    results = [score_item(item, predictions.get(item["id"])) for item in dataset]
    dataset_ids = {item["id"] for item in dataset}
    extra_prediction_ids = sorted(set(predictions) - dataset_ids)

    by_type = {
        str(item_type): _aggregate(
            [result for result in results if result["type"] == item_type]
        )
        for item_type in sorted(SCORING_BY_TYPE)
    }
    by_scoring = {
        scoring: _aggregate(
            [result for result in results if result["scoring"] == scoring]
        )
        for scoring in SCORING_BY_TYPE.values()
    }
    summary = _aggregate(results)
    summary.update(
        {
            "dataset_items": len(dataset),
            "prediction_items": len(predictions),
            "extra_predictions": len(extra_prediction_ids),
            "extra_prediction_ids": extra_prediction_ids,
        }
    )
    return {
        "schema_version": 1,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "by_type": by_type,
        "by_scoring": by_scoring,
        "items": results,
    }
