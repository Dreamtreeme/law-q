# Bllossom 3B Q4 종단 스모크 요약

이 문서는 Git에서 제외되는 최종 원본 `results/smoke-locked-20260905/`의 재현 가능한 요약이다. 합성 4문항으로 파이프라인만 검증했으며 모델 품질 결론에는 사용하지 않는다.

## 실행 결과

| Model | Quant | GGUF GiB | VRAM MiB | TTFT s | 응답 s | PP512 | PP2048 | PP4096 | TG512 | TG2048 | TG4096 | Overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bllossom-3b | Q4_K_M | 1.881 | 3713 | 0.035098 | 0.207121 | 10042.76 | 9747.92 | 9180.45 | 232.00 | 214.02 | 198.80 | 66.67% |

유형별 점수는 T1 100%, T2 0%, T3 66.67%, T4 100%다. 모든 평가 요청과 벤치마크는 성공했다.

## 원본 무결성

| 원본 파일 | SHA-256 |
| --- | --- |
| `results.csv` | `dcb7b149dc59a63ff41bdde37ed1973539bd047c46883ecffaa7a803e34add5c` |
| `responses.jsonl` | `7ebbfba0f8a60f540e1ce9c8c4fabf63a7f551651abdad9b298368a4ae25c07f` |
| `scores.json` | `601f6d20f1b355d86bc60327faac830ba90c88b0c031b8bd3e6ffeab860f9326` |
| 벤치마크 `summary.json` | `d3f424df811996bcf6883e6c0cfd740f05e8aef7b9d653d7ac66f9e0a198e349` |
| 생성 `REPORT.md` | `625ca87ef920f01e296146f6ad4a15963c5cc58ba49710f626a2e94e159d2a9a` |

실행 설정 SHA-256은 `d408d09483f8636515ae109b966169d9e4e75bc441c3ced7e36f0eae355586bc`, llama.cpp 커밋은 `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`다.
