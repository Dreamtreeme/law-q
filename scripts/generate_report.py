from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common import DEFAULT_CONFIG, load_config, resolve_paths


SCORE_FIELDS = ["type1_score", "type2_score", "type3_score", "type4_score"]


class ReportError(ValueError):
    """결과 리포트를 만들 수 없을 때 발생합니다."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return statistics.fmean(available) if available else None


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ReportError(f"결과 CSV를 찾지 못했습니다: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ReportError(f"결과 CSV가 비어 있습니다: {path}")
    required = {"model", "quantization", "status"}
    missing = required - set(rows[0])
    if missing:
        raise ReportError(f"결과 CSV에 필수 열이 없습니다: {', '.join(sorted(missing))}")
    return rows


def resolve_input(
    config: dict[str, Any], input_path: str | None, run_name: str | None
) -> Path:
    if input_path:
        path = Path(input_path)
        return path.resolve() if path.is_absolute() else (Path(config["_project_root"]) / path).resolve()
    results_dir = resolve_paths(config)["results"]
    if run_name:
        return (results_dir / run_name / "results.csv").resolve()
    candidates = sorted(
        results_dir.glob("*/results.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise ReportError("results 아래에서 results.csv를 찾지 못했습니다.")
    return candidates[0].resolve()


def normalize_quantization(value: str) -> str:
    normalized = value.upper().replace("-", "_")
    if normalized in {"FP16", "F16"}:
        return "F16"
    if normalized in {"Q8", "Q8_0"}:
        return "Q8_0"
    return normalized


def quantization_order(value: str) -> tuple[int, str]:
    normalized = normalize_quantization(value)
    order = {"Q4_K_M": 4, "Q4": 4, "Q5_K_M": 5, "Q5": 5, "Q8_0": 8, "F16": 16}
    if normalized in order:
        return order[normalized], normalized
    digits = "".join(character for character in normalized if character.isdigit())
    return (int(digits) if digits else 999), normalized


def comprehensive_rows(rows: list[dict[str, str]], lengths: list[int]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item: dict[str, Any] = {
            "model": row["model"],
            "quantization": row["quantization"],
            "status": row.get("status", ""),
            "file_size_bytes": as_float(row.get("gguf_size_bytes")),
            "file_size_gib": (
                round(as_float(row.get("gguf_size_bytes")) / (1024**3), 3)
                if as_float(row.get("gguf_size_bytes")) is not None
                else None
            ),
            "max_vram_mib": as_float(row.get("max_vram_used_mib")),
            "ttft_seconds": as_float(row.get("ttft_mean_seconds")),
            "type1_score": as_float(row.get("type1_score")),
            "type2_score": as_float(row.get("type2_score")),
            "type3_score": as_float(row.get("type3_score")),
            "type4_score": as_float(row.get("type4_score")),
            "overall_score": as_float(row.get("overall_score")),
        }
        for length in lengths:
            item[f"pp_{length}_mean_tps"] = as_float(row.get(f"pp_{length}_mean_tps"))
            item[f"tg_{length}_mean_tps"] = as_float(row.get(f"tg_{length}_mean_tps"))
        output.append(item)
    return output


def choose_baselines(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    baselines: dict[str, dict[str, str]] = {}
    for model, candidates in grouped.items():
        scored_candidates = [
            row
            for row in candidates
            if any(
                as_float(row.get(field)) is not None
                for field in [*SCORE_FIELDS, "overall_score"]
            )
        ]
        f16 = next(
            (
                row
                for row in scored_candidates
                if normalize_quantization(row["quantization"]) == "F16"
            ),
            None,
        )
        q8 = next(
            (
                row
                for row in scored_candidates
                if normalize_quantization(row["quantization"]) == "Q8_0"
            ),
            None,
        )
        if f16 is not None:
            baselines[model] = f16
        elif q8 is not None:
            baselines[model] = q8
    return baselines


def quantization_losses(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    baselines = choose_baselines(rows)
    details: list[dict[str, Any]] = []
    excluded_models = sorted({row["model"] for row in rows} - set(baselines))
    for row in rows:
        baseline = baselines.get(row["model"])
        if baseline is None:
            continue
        detail: dict[str, Any] = {
            "model": row["model"],
            "quantization": normalize_quantization(row["quantization"]),
            "baseline": normalize_quantization(baseline["quantization"]),
        }
        for field in [*SCORE_FIELDS, "overall_score"]:
            baseline_value = as_float(baseline.get(field))
            value = as_float(row.get(field))
            detail[f"{field}_drop_pp"] = (
                round((baseline_value - value) * 100, 4)
                if baseline_value is not None and value is not None
                else None
            )
        details.append(detail)

    grouped_details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped_details[detail["quantization"]].append(detail)
    aggregate: list[dict[str, Any]] = []
    for quantization, candidates in sorted(
        grouped_details.items(), key=lambda item: quantization_order(item[0])
    ):
        baseline_mix = Counter(candidate["baseline"] for candidate in candidates)
        item: dict[str, Any] = {
            "quantization": quantization,
            "models_compared": len({candidate["model"] for candidate in candidates}),
            "baseline_mix": ", ".join(
                f"{name}:{count}" for name, count in sorted(baseline_mix.items())
            ),
        }
        for field in [*SCORE_FIELDS, "overall_score"]:
            item[f"{field}_drop_pp"] = mean(
                candidate[f"{field}_drop_pp"] for candidate in candidates
            )
            if item[f"{field}_drop_pp"] is not None:
                item[f"{field}_drop_pp"] = round(item[f"{field}_drop_pp"], 4)
        aggregate.append(item)
    return aggregate, details, excluded_models


def group_similar_vram(
    rows: list[dict[str, str]], tolerance_mib: float
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates = []
    missing = []
    for row in rows:
        vram = as_float(row.get("max_vram_used_mib"))
        if vram is None:
            missing.append(f"{row['model']}/{row['quantization']}")
        else:
            candidates.append((vram, row))
    candidates.sort(key=lambda item: item[0])

    groups: list[list[tuple[float, dict[str, str]]]] = []
    current: list[tuple[float, dict[str, str]]] = []
    group_min = 0.0
    for candidate in candidates:
        if not current or candidate[0] - group_min <= tolerance_mib:
            if not current:
                group_min = candidate[0]
            current.append(candidate)
        else:
            groups.append(current)
            current = [candidate]
            group_min = candidate[0]
    if current:
        groups.append(current)

    output = []
    for group_number, group in enumerate(groups, start=1):
        vram_values = [item[0] for item in group]

        def score_order(item: tuple[float, dict[str, str]]) -> tuple[bool, float, str]:
            score = as_float(item[1].get("overall_score"))
            return score is None, -(score if score is not None else 0.0), item[1]["model"]

        for vram, row in sorted(
            group,
            key=score_order,
        ):
            output.append(
                {
                    "group": group_number,
                    "group_size": len(group),
                    "vram_range_mib": f"{min(vram_values):.0f}-{max(vram_values):.0f}",
                    "model": row["model"],
                    "quantization": row["quantization"],
                    "max_vram_mib": vram,
                    "overall_score": as_float(row.get("overall_score")),
                    "ttft_seconds": as_float(row.get("ttft_mean_seconds")),
                    "pp_2048_mean_tps": as_float(row.get("pp_2048_mean_tps")),
                    "tg_2048_mean_tps": as_float(row.get("tg_2048_mean_tps")),
                }
            )
    return output, missing


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["no_data"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _svg_document(elements: list[str], title: str) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600" viewBox="0 0 1000 600">',
            f"<title>{html.escape(title)}</title>",
            '<rect width="1000" height="600" fill="white"/>',
            '<g font-family="Arial, sans-serif" fill="#111827">',
            *elements,
            "</g>",
            "</svg>",
            "",
        ]
    )


def _write_svg(path: Path, elements: list[str], title: str) -> None:
    path.write_text(_svg_document(elements, title), encoding="utf-8")


def _chart_axes(title: str, x_label: str, y_label: str) -> list[str]:
    elements = [
        f'<text x="500" y="28" font-size="20" text-anchor="middle" font-weight="bold">{html.escape(title)}</text>',
        '<line x1="90" y1="520" x2="960" y2="520" stroke="#374151" stroke-width="1.5"/>',
        '<line x1="90" y1="55" x2="90" y2="520" stroke="#374151" stroke-width="1.5"/>',
        f'<text x="525" y="580" font-size="14" text-anchor="middle">{html.escape(x_label)}</text>',
        f'<text x="22" y="290" font-size="14" text-anchor="middle" transform="rotate(-90 22 290)">{html.escape(y_label)}</text>',
    ]
    for value in range(0, 101, 20):
        y = 520 - value * 4.65
        elements.extend(
            [
                f'<line x1="90" y1="{y:.1f}" x2="960" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="80" y="{y + 4:.1f}" font-size="11" text-anchor="end">{value}</text>',
            ]
        )
    return elements


def plot_accuracy_speed(
    rows: list[dict[str, str]], output: Path, speed_length: int
) -> int:
    points = []
    for row in rows:
        accuracy = as_float(row.get("overall_score"))
        speed = as_float(row.get(f"tg_{speed_length}_mean_tps"))
        if accuracy is not None and speed is not None:
            points.append((speed, accuracy * 100, row))
    colors = {"Q4_K_M": "#2563eb", "Q5_K_M": "#16a34a", "Q8_0": "#ea580c", "F16": "#7c3aed"}
    elements = _chart_axes(
        "Accuracy vs. Generation Speed",
        f"Token generation speed at context {speed_length} (tokens/s)",
        "Overall accuracy (%)",
    )
    if points:
        speeds = [point[0] for point in points]
        minimum = min(speeds)
        maximum = max(speeds)
        padding = max((maximum - minimum) * 0.08, maximum * 0.03, 1.0)
        x_min = max(0.0, minimum - padding)
        x_max = maximum + padding
        if x_max == x_min:
            x_max = x_min + 1.0
        for index in range(6):
            value = x_min + (x_max - x_min) * index / 5
            x = 90 + 870 * index / 5
            elements.extend(
                [
                    f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="520" stroke="#f3f4f6"/>',
                    f'<text x="{x:.1f}" y="540" font-size="11" text-anchor="middle">{value:.1f}</text>',
                ]
            )
        for speed, accuracy, row in points:
            quantization = normalize_quantization(row["quantization"])
            x = 90 + (speed - x_min) / (x_max - x_min) * 870
            y = 520 - max(0.0, min(100.0, accuracy)) * 4.65
            label = html.escape(f"{row['model']} / {row['quantization']}")
            elements.extend(
                [
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors.get(quantization, "#64748b")}" opacity="0.9"><title>{label}: {accuracy:.1f}%, {speed:.2f} t/s</title></circle>',
                    f'<text x="{x + 8:.1f}" y="{y - 7:.1f}" font-size="10">{label}</text>',
                ]
            )
    else:
        elements.append('<text x="525" y="290" font-size="16" text-anchor="middle" fill="#6b7280">No complete accuracy/speed data</text>')
    _write_svg(output, elements, "Accuracy vs. Generation Speed")
    return len(points)


def plot_quantization_scores(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        quantization = normalize_quantization(row["quantization"])
        for field in SCORE_FIELDS:
            value = as_float(row.get(field))
            if value is not None:
                grouped[quantization][field].append(value * 100)
    quantizations = sorted(grouped, key=quantization_order)
    elements = _chart_axes(
        "Type Scores by Quantization", "Quantization", "Mean score (%)"
    )
    x_positions = {
        quantization: (
            525.0
            if len(quantizations) == 1
            else 120 + index * 810 / (len(quantizations) - 1)
        )
        for index, quantization in enumerate(quantizations)
    }
    for quantization, x in x_positions.items():
        elements.append(
            f'<text x="{x:.1f}" y="542" font-size="12" text-anchor="middle">{html.escape(quantization)}</text>'
        )
    plotted = 0
    colors = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed"]
    for type_number, field in enumerate(SCORE_FIELDS, start=1):
        coordinates = []
        for quantization in quantizations:
            values = grouped[quantization].get(field, [])
            if values:
                score = statistics.fmean(values)
                coordinates.append((x_positions[quantization], 520 - score * 4.65, score))
        if coordinates:
            color = colors[type_number - 1]
            points_attribute = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coordinates)
            elements.append(
                f'<polyline points="{points_attribute}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
            for x, y, score in coordinates:
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>Type {type_number}: {score:.1f}%</title></circle>'
                )
            legend_x = 710 + (type_number - 1) % 2 * 120
            legend_y = 75 + (type_number - 1) // 2 * 24
            elements.extend(
                [
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-size="12">Type {type_number}</text>',
                ]
            )
            plotted += 1
    if not plotted:
        elements.append('<text x="525" y="290" font-size="16" text-anchor="middle" fill="#6b7280">No type score data</text>')
    _write_svg(output, elements, "Type Scores by Quantization")
    return {"quantizations": quantizations, "plotted_type_series": plotted}


def fmt(value: Any, digits: int = 2, percent: bool = False) -> str:
    number = as_float(value)
    if number is None:
        return "N/A"
    if percent:
        number *= 100
    return f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escape(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(
    input_path: Path,
    comprehensive: list[dict[str, Any]],
    losses: list[dict[str, Any]],
    memory_groups: list[dict[str, Any]],
    excluded_baselines: list[str],
    missing_vram: list[str],
    lengths: list[int],
    speed_length: int,
    tolerance_mib: float,
) -> str:
    combo_headers = ["Model", "Quant", "Size GiB", "VRAM MiB", "TTFT s"]
    combo_headers += [f"PP{length}" for length in lengths]
    combo_headers += [f"TG{length}" for length in lengths]
    combo_headers += ["T1", "T2", "T3", "T4", "Overall", "Status"]
    combo_rows = []
    for row in comprehensive:
        values = [
            row["model"],
            row["quantization"],
            fmt(row["file_size_gib"], 3),
            fmt(row["max_vram_mib"], 0),
            fmt(row["ttft_seconds"], 3),
        ]
        values += [fmt(row.get(f"pp_{length}_mean_tps"), 2) for length in lengths]
        values += [fmt(row.get(f"tg_{length}_mean_tps"), 2) for length in lengths]
        values += [fmt(row.get(field), 1, percent=True) for field in SCORE_FIELDS]
        values += [fmt(row["overall_score"], 1, percent=True), row["status"]]
        combo_rows.append(values)

    loss_headers = ["Quant", "Models", "Baseline mix", "T1 drop pp", "T2 drop pp", "T3 drop pp", "T4 drop pp", "Overall drop pp"]
    loss_rows = [
        [
            row["quantization"],
            row["models_compared"],
            row["baseline_mix"],
            fmt(row["type1_score_drop_pp"]),
            fmt(row["type2_score_drop_pp"]),
            fmt(row["type3_score_drop_pp"]),
            fmt(row["type4_score_drop_pp"]),
            fmt(row["overall_score_drop_pp"]),
        ]
        for row in losses
    ]
    memory_headers = ["Group", "Range MiB", "Model", "Quant", "VRAM MiB", "Overall %", "TTFT s", "PP2048", "TG2048"]
    memory_rows = [
        [
            row["group"],
            row["vram_range_mib"],
            row["model"],
            row["quantization"],
            fmt(row["max_vram_mib"], 0),
            fmt(row["overall_score"], 1, percent=True),
            fmt(row["ttft_seconds"], 3),
            fmt(row["pp_2048_mean_tps"]),
            fmt(row["tg_2048_mean_tps"]),
        ]
        for row in memory_groups
    ]

    notes = []
    if excluded_baselines:
        notes.append("- F16과 Q8 기준이 모두 없어 손실 계산에서 제외: " + ", ".join(excluded_baselines))
    if missing_vram:
        notes.append("- VRAM 측정값이 없어 메모리 그룹에서 제외: " + ", ".join(missing_vram))
    if not notes:
        notes.append("- 누락된 기준 또는 VRAM 값이 없습니다.")

    return f"""# 한국어 법률 QA 모델·양자화 실험 리포트

- 생성 시각(UTC): {utc_now()}
- 입력 결과: `{input_path}`
- 조합 수: {len(comprehensive)}

## 1. 조합별 종합 표

점수는 백분율, PP/TG는 tokens/s입니다.

{markdown_table(combo_headers, combo_rows)}

## 2. 양자화 수준별 손실 표

각 모델에서 F16을 우선 기준으로 사용하고, F16이 없으면 Q8을 기준으로 사용했습니다. 값은 기준 대비 평균 점수 하락폭(percentage points)이며 음수는 기준보다 높은 점수입니다.

{markdown_table(loss_headers, loss_rows)}

## 3. 동일 메모리 예산 비교

최대 VRAM 차이가 그룹 최솟값에서 {tolerance_mib:.0f} MiB 이내인 조합끼리 묶었습니다.

{markdown_table(memory_headers, memory_rows)}

## 그래프

### 정확도 vs 속도

속도 축은 컨텍스트 {speed_length}의 TG 속도입니다.

![Accuracy vs speed](accuracy-vs-speed.svg)

### 양자화 수준별 유형 점수

각 양자화 수준에서 사용 가능한 모델의 유형별 평균입니다.

![Type scores by quantization](type-scores-by-quantization.svg)

## 데이터 품질 메모

{chr(10).join(notes)}
"""


def generate_report(
    input_path: Path, output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    rows = read_results(input_path)
    report_config = config.get("report", {})
    lengths = [int(value) for value in config["benchmark"]["prompt_lengths"]]
    speed_length = int(report_config.get("speed_context_length", 2048))
    if speed_length not in lengths:
        raise ReportError(
            f"report.speed_context_length={speed_length}가 benchmark.prompt_lengths에 없습니다."
        )
    tolerance_mib = float(report_config.get("similar_vram_tolerance_mib", 512))
    if tolerance_mib < 0:
        raise ReportError("report.similar_vram_tolerance_mib는 0 이상이어야 합니다.")

    output_dir.mkdir(parents=True, exist_ok=True)
    comprehensive = comprehensive_rows(rows, lengths)
    losses, loss_details, excluded_baselines = quantization_losses(rows)
    memory_groups, missing_vram = group_similar_vram(rows, tolerance_mib)
    scatter_count = plot_accuracy_speed(
        rows, output_dir / "accuracy-vs-speed.svg", speed_length
    )
    line_metadata = plot_quantization_scores(
        rows, output_dir / "type-scores-by-quantization.svg"
    )

    write_csv(output_dir / "combination-summary.csv", comprehensive)
    write_csv(output_dir / "quantization-loss.csv", losses)
    write_csv(output_dir / "quantization-loss-detail.csv", loss_details)
    write_csv(output_dir / "memory-budget-groups.csv", memory_groups)
    analysis = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "input": str(input_path.resolve()),
        "configuration": {
            "prompt_lengths": lengths,
            "speed_context_length": speed_length,
            "similar_vram_tolerance_mib": tolerance_mib,
            "baseline_policy": "F16, otherwise Q8_0",
        },
        "combination_summary": comprehensive,
        "quantization_loss": losses,
        "quantization_loss_detail": loss_details,
        "memory_budget_groups": memory_groups,
        "data_quality": {
            "models_without_f16_or_q8_baseline": excluded_baselines,
            "combinations_without_vram": missing_vram,
            "scatter_points": scatter_count,
            **line_metadata,
        },
    }
    write_json(output_dir / "analysis.json", analysis)
    markdown = build_markdown(
        input_path,
        comprehensive,
        losses,
        memory_groups,
        excluded_baselines,
        missing_vram,
        lengths,
        speed_length,
        tolerance_mib,
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(markdown, encoding="utf-8")
    return {"report": report_path, "analysis": analysis}


def main() -> None:
    parser = argparse.ArgumentParser(description="실험 결과 CSV를 분석해 표와 그래프 리포트를 만듭니다.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--input", help="통합 파이프라인 results.csv")
    parser.add_argument("--run-name", help="results 하위 실행 이름")
    parser.add_argument("--output-dir", help="리포트 출력 디렉터리")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        input_path = resolve_input(config, args.input, args.run_name)
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else input_path.parent / "report"
        )
        result = generate_report(input_path, output_dir, config)
    except (OSError, ReportError) as error:
        parser.error(str(error))
    print(f"리포트: {result['report'].resolve()}")
    print(f"분석 JSON: {(output_dir / 'analysis.json').resolve()}")


if __name__ == "__main__":
    main()
