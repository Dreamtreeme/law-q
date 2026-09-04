from __future__ import annotations

import argparse
import json

from common import DEFAULT_CONFIG, experiment_matrix, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="설정에서 실험 조합을 출력합니다.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 설정 파일")
    parser.add_argument("--json", action="store_true", help="JSON 형식으로 출력")
    args = parser.parse_args()

    matrix = experiment_matrix(load_config(args.config))
    if args.json:
        print(json.dumps(matrix, ensure_ascii=False, indent=2))
        return

    print(f"실험 조합: {len(matrix)}개")
    for index, item in enumerate(matrix, start=1):
        print(
            f"{index:>2}. {item['model']} @ {item['revision']} / "
            f"{item['quantization']}"
        )


if __name__ == "__main__":
    main()

