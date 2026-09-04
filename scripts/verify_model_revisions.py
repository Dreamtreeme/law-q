from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, load_config


FULL_SHA = re.compile(r"[0-9a-f]{40}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_revisions(
    config: dict[str, Any], *, strict: bool = True, api: Any | None = None
) -> dict[str, Any]:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()

    records = []
    token = os.environ.get("HF_TOKEN") or None
    for model in config["models"]:
        repo = str(model["hf_repo"])
        requested = str(model["revision"])
        record: dict[str, Any] = {
            "model": str(model["name"]),
            "hf_repo": repo,
            "requested_revision": requested,
            "resolved_revision": None,
            "status": "failed",
            "error": None,
        }
        if strict and FULL_SHA.fullmatch(requested) is None:
            record["error"] = "strict 모드에서는 40자리 소문자 Git SHA가 필요합니다."
            records.append(record)
            continue
        try:
            info = api.model_info(repo_id=repo, revision=requested, token=token)
            resolved = str(info.sha)
            record["resolved_revision"] = resolved
            if FULL_SHA.fullmatch(requested) and resolved != requested:
                record["error"] = (
                    f"요청 SHA와 HF가 해석한 SHA가 다릅니다: {requested} != {resolved}"
                )
            else:
                record["status"] = "success"
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)

    failures = [record for record in records if record["status"] != "success"]
    return {
        "schema_version": 1,
        "checked_at_utc": utc_now(),
        "strict": strict,
        "status": "success" if not failures else "failed",
        "verified_models": len(records) - len(failures),
        "failed_models": len(failures),
        "models": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="설정된 Hugging Face revision이 실제로 존재하고 정확히 해석되는지 검증합니다."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument(
        "--output", default="results/revision-verification.json", help="검증 JSON 출력"
    )
    parser.add_argument(
        "--allow-moving-revision",
        action="store_true",
        help="main 같은 이동 가능한 revision을 허용",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    report = verify_revisions(config, strict=not args.allow_moving_revision)
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(config["_project_root"]) / output
    write_json_atomic(output.resolve(), report)
    for record in report["models"]:
        print(
            f"[{record['status']}] {record['model']}: "
            f"{record['requested_revision']} -> {record['resolved_revision'] or record['error']}"
        )
    print(f"검증 결과: {report['status']} ({output.resolve()})")
    raise SystemExit(0 if report["status"] == "success" else 1)


if __name__ == "__main__":
    main()
