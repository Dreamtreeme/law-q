from __future__ import annotations

import argparse
import json

from common import DEFAULT_CONFIG, capture_environment, create_run_directory, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="실험 실행 환경 정보를 기록합니다.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--run-name", help="results 아래에 생성할 실행 이름")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="파일을 만들지 않고 환경 정보를 표준 출력으로 표시",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stdout:
        print(json.dumps(capture_environment(config), ensure_ascii=False, indent=2))
        return

    run_dir = create_run_directory(config, args.run_name)
    print(run_dir)


if __name__ == "__main__":
    main()

