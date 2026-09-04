from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, load_config
from scoring import EvaluationError, load_dataset


REVIEW_FIELDS = [
    "review_id",
    "candidate_alias",
    "question_id",
    "context_doc",
    "context_text",
    "question",
    "response",
    "auto_score",
    "matched_keywords",
    "human_score_0_to_2",
    "logical_error_yes_no",
    "reviewer_notes",
]


class ManualReviewError(ValueError):
    """블라인드 검토 자료가 불완전하거나 잘못되었을 때 발생합니다."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def select_top_combinations(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    successful = [row for row in rows if row.get("status") == "success"]
    if len(successful) < top_k:
        raise ManualReviewError(
            f"성공 조합이 {len(successful)}개뿐이라 상위 {top_k}개를 선택할 수 없습니다."
        )
    try:
        successful.sort(
            key=lambda row: (
                -float(row["overall_score"]),
                row["model"],
                row["quantization"],
            )
        )
    except (KeyError, ValueError) as error:
        raise ManualReviewError("results.csv의 종합 점수를 읽을 수 없습니다.") from error
    return successful[:top_k]


def jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_id = str(record["id"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ManualReviewError(f"{path}:{line_number} JSONL 형식 오류") from error
            if record_id in records:
                raise ManualReviewError(f"{path}: 중복 ID {record_id}")
            records[record_id] = record
    return records


def safe_document(dataset_path: Path, relative_value: str) -> Path:
    root = dataset_path.parent.resolve()
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ManualReviewError(f"context_doc는 상대 경로여야 합니다: {relative_value}")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ManualReviewError(f"context_doc가 평가셋 경로를 벗어납니다: {relative_value}")
    if not resolved.is_file():
        raise ManualReviewError(f"참조 문서가 없습니다: {resolved}")
    return resolved


def export_review(
    results_csv: Path,
    dataset_path: Path,
    output_dir: Path,
    top_k: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    if top_k < 1:
        raise ManualReviewError("top-k는 1 이상이어야 합니다.")
    results_csv = results_csv.resolve()
    dataset_path = dataset_path.resolve()
    selected = select_top_combinations(read_csv(results_csv), top_k)
    type2_items = [item for item in load_dataset(dataset_path) if int(item["type"]) == 2]
    if not type2_items:
        raise ManualReviewError("유형 2 문항이 없습니다.")

    shuffled = list(selected)
    random.Random(seed).shuffle(shuffled)
    alias_by_key = {
        (row["model"], row["quantization"]): f"C{index:02d}"
        for index, row in enumerate(shuffled, start=1)
    }
    review_rows: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    run_dir = results_csv.parent

    for combination in selected:
        model = combination["model"]
        quantization = combination["quantization"]
        alias = alias_by_key[(model, quantization)]
        combo_dir = run_dir / "evaluations" / model / quantization
        response_path = combo_dir / "responses.jsonl"
        score_path = combo_dir / "scores.json"
        if not response_path.is_file() or not score_path.is_file():
            raise ManualReviewError(f"평가 원본이 없습니다: {model}/{quantization}")
        responses = jsonl_by_id(response_path)
        score_document = json.loads(score_path.read_text(encoding="utf-8-sig"))
        scores = {str(item["id"]): item for item in score_document.get("items", [])}
        mapping[alias] = {
            "model": model,
            "quantization": quantization,
            "overall_score": float(combination["overall_score"]),
            "responses_sha256": sha256(response_path),
            "scores_sha256": sha256(score_path),
        }
        for item in type2_items:
            item_id = str(item["id"])
            response = responses.get(item_id)
            score = scores.get(item_id)
            if response is None or response.get("status") != "success" or score is None:
                raise ManualReviewError(f"{model}/{quantization}/{item_id}: 응답 또는 점수 누락")
            document_path = safe_document(dataset_path, str(item["context_doc"]))
            review_rows.append(
                {
                    "review_id": f"{alias}-{item_id}",
                    "candidate_alias": alias,
                    "question_id": item_id,
                    "context_doc": item["context_doc"],
                    "context_text": document_path.read_text(encoding="utf-8-sig"),
                    "question": item["question"],
                    "response": response.get("response", ""),
                    "auto_score": score.get("score", ""),
                    "matched_keywords": " | ".join(score.get("matched_keywords", [])),
                    "human_score_0_to_2": "",
                    "logical_error_yes_no": "",
                    "reviewer_notes": "",
                }
            )

    # 후보가 연속해서 보이지 않도록 문항별 후보 순서를 고정 seed로 섞습니다.
    random.Random(seed + 1).shuffle(review_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "type2-blind-review.csv"
    key_path = output_dir / "type2-blinding-key.json"
    write_csv(review_path, REVIEW_FIELDS, review_rows)
    key = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "seed": seed,
        "top_k": top_k,
        "expected_reviews": len(type2_items) * top_k,
        "results_csv": str(results_csv),
        "results_csv_sha256": sha256(results_csv),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "aliases": mapping,
        "rubric": {
            "0": "핵심 법리·조건이 틀렸거나 결론이 모순됨",
            "1": "일부 조건 또는 논리가 맞지만 중요한 누락·오류가 있음",
            "2": "문서 근거, 조건 적용과 결론이 모두 타당함",
            "logical_error_yes_no": "yes 또는 no",
        },
    }
    write_json(key_path, key)
    return {"review": review_path, "key": key_path, "rows": len(review_rows)}


def summarize_review(review_path: Path, key_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_csv(review_path)
    key = json.loads(key_path.read_text(encoding="utf-8-sig"))
    expected = int(key["expected_reviews"])
    if len(rows) != expected:
        raise ManualReviewError(f"검토 행 수가 다릅니다: 기대 {expected}, 실제 {len(rows)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        review_id = row.get("review_id", "")
        if not review_id or review_id in seen:
            raise ManualReviewError(f"빈 값 또는 중복 review_id: {review_id}")
        seen.add(review_id)
        alias = row.get("candidate_alias", "")
        if alias not in key["aliases"]:
            raise ManualReviewError(f"알 수 없는 후보 별칭: {alias}")
        try:
            human_score = int(row.get("human_score_0_to_2", ""))
        except ValueError as error:
            raise ManualReviewError(f"{review_id}: 수동 점수가 비어 있거나 정수가 아닙니다.") from error
        if human_score not in {0, 1, 2}:
            raise ManualReviewError(f"{review_id}: 수동 점수는 0, 1, 2 중 하나여야 합니다.")
        logical_error = row.get("logical_error_yes_no", "").strip().casefold()
        if logical_error not in {"yes", "no"}:
            raise ManualReviewError(f"{review_id}: 논리 오류는 yes 또는 no여야 합니다.")
        grouped[alias].append(
            {
                "human_score": human_score,
                "auto_score": float(row["auto_score"]),
                "logical_error": logical_error == "yes",
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for alias, values in grouped.items():
        identity = key["aliases"][alias]
        count = len(values)
        summary_rows.append(
            {
                "candidate_alias": alias,
                "model": identity["model"],
                "quantization": identity["quantization"],
                "reviews": count,
                "auto_score_mean": round(sum(v["auto_score"] for v in values) / count, 6),
                "human_score_mean_0_to_2": round(
                    sum(v["human_score"] for v in values) / count, 6
                ),
                "human_score_percent": round(
                    50 * sum(v["human_score"] for v in values) / count, 2
                ),
                "logical_error_count": sum(v["logical_error"] for v in values),
            }
        )
    summary_rows.sort(key=lambda row: (-row["human_score_percent"], row["candidate_alias"]))
    fields = [
        "candidate_alias",
        "model",
        "quantization",
        "reviews",
        "auto_score_mean",
        "human_score_mean_0_to_2",
        "human_score_percent",
        "logical_error_count",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "type2-manual-summary.csv"
    json_path = output_dir / "type2-manual-summary.json"
    write_csv(csv_path, fields, summary_rows)
    report = {
        "schema_version": 1,
        "summarized_at_utc": utc_now(),
        "review_csv_sha256": sha256(review_path),
        "rows": len(rows),
        "combinations": summary_rows,
        "note": "수동 점수는 자동 점수를 대체하지 않는 별도 지표입니다.",
    }
    write_json(json_path, report)
    return {"csv": csv_path, "json": json_path, "rows": len(rows)}


def resolve_dataset(config: dict[str, Any], override: str | None) -> Path:
    value = override or config["evaluation"]["dataset"]
    path = Path(value)
    return path if path.is_absolute() else Path(config["_project_root"]) / path


def main() -> None:
    parser = argparse.ArgumentParser(description="유형 2 응답 블라인드 수동 검토 자료를 관리합니다.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="상위 조합의 블라인드 검토 CSV 생성")
    export_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    export_parser.add_argument("--input", required=True, help="본 실험 results.csv")
    export_parser.add_argument("--dataset", help="평가셋 경로 대체")
    export_parser.add_argument("--output-dir", help="기본값: results.csv 옆 manual-review")
    export_parser.add_argument("--top-k", type=int, default=3)
    export_parser.add_argument("--seed", type=int, default=42)
    summary_parser = subparsers.add_parser("summarize", help="작성된 수동 검토 CSV 집계")
    summary_parser.add_argument("--review", required=True)
    summary_parser.add_argument("--key", required=True)
    summary_parser.add_argument("--output-dir")
    args = parser.parse_args()

    try:
        if args.command == "export":
            config = load_config(args.config)
            results_csv = Path(args.input).resolve()
            output_dir = (
                Path(args.output_dir).resolve()
                if args.output_dir
                else results_csv.parent / "manual-review"
            )
            result = export_review(
                results_csv,
                resolve_dataset(config, args.dataset),
                output_dir,
                args.top_k,
                args.seed,
            )
            print(f"블라인드 검토 CSV: {result['review']} ({result['rows']}행)")
            print(f"별도 보관할 블라인딩 키: {result['key']}")
        else:
            review_path = Path(args.review).resolve()
            output_dir = (
                Path(args.output_dir).resolve()
                if args.output_dir
                else review_path.parent
            )
            result = summarize_review(
                review_path, Path(args.key).resolve(), output_dir
            )
            print(f"수동 검토 집계: {result['csv']} ({result['rows']}행)")
    except (
        ManualReviewError,
        EvaluationError,
        OSError,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
