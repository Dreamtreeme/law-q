from __future__ import annotations

import argparse
import json
from pathlib import Path

from scoring import EvaluationError, load_dataset, load_predictions, score_dataset


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="한국어 법률 QA 응답을 채점합니다.")
    parser.add_argument("--dataset", required=True, help="평가셋 JSONL 또는 JSON")
    parser.add_argument("--predictions", required=True, help="예측 결과 JSONL 또는 JSON")
    parser.add_argument("--output", required=True, help="채점 리포트 JSON 경로")
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.dataset)
        predictions = load_predictions(args.predictions)
        report = score_dataset(dataset, predictions)
        write_report(Path(args.output), report)
    except EvaluationError as error:
        parser.error(str(error))

    summary = report["summary"]
    print(f"전체 점수: {summary['total_score']}/{summary['count']} ({summary['percent']}%)")
    for item_type, result in report["by_type"].items():
        percent = "N/A" if result["percent"] is None else f"{result['percent']}%"
        print(f"유형 {item_type}: {result['total_score']}/{result['count']} ({percent})")
    print(f"JSON 파싱 실패: {summary['json_parse_failures']}")
    print(f"누락 예측: {summary['missing_predictions']}")
    print(f"결과 파일: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

