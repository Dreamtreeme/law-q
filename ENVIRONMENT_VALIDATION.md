# 실행 환경 검증 결과

## 최종 검증

- 시각: 2026-09-05 03:30 KST (`2026-09-04T18:30:30Z`)
- 운영체제: Windows 11 `10.0.26200`, AMD64
- Python: 3.13.14, 프로젝트 `.venv` 사용
- CPU: Intel64 Family 6 Model 151, 논리 CPU 20개
- GPU: NVIDIA GeForce RTX 3080, 10,240 MiB, Compute Capability 8.6
- NVIDIA 드라이버: 591.86
- 프로젝트 Git: `f99e5b0160ec75b27a2784d226d92a922b118d54`에서 실행, 작업 트리 변경 있음
- 원격 저장소: `https://github.com/Dreamtreeme/law-q.git`

## llama.cpp CUDA 런타임

공식 릴리스 `b10809`를 `C:\Users\psg\Desktop\개발\llama.cpp`에 고정했다.

| 항목 | 값 |
| --- | --- |
| Git 커밋 | `5266f24da75dc449bd56cbed7addb9c8e4a6a73e` |
| 버전/빌드 | `0.4.0-dev` / 10809 |
| 툴체인 | Clang 20.1.8 for Windows x86_64 |
| CUDA 패키지 | 12.4 |
| CUDA backend | 로드 성공 |
| llama.cpp가 인식한 VRAM | 10,239 MiB |

실행 파일 SHA-256은 [`runtime-lock.json`](runtime-lock.json)에 기록했다. `llama-server`, `llama-quantize`, `llama-bench`의 실행과 RTX 3080 CUDA 인식을 모두 확인했다.

## Python 패키지

| 패키지 | 버전 |
| --- | --- |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |
| huggingface_hub | 0.36.2 |
| torch | 2.11.0+cpu |
| transformers | 4.57.6 |
| sentencepiece | 0.2.2 |

llama.cpp의 변환기 requirements에 고정된 NumPy 1.26은 Python 3.13 wheel과 호환되지 않아, 프로젝트는 [`requirements-convert.txt`](requirements-convert.txt)의 NumPy 2.5.2 조합을 사용한다. 실제 Bllossom 변환으로 호환성을 검증했다.

## 모델·설정 검증

- 4개 모델 revision은 모두 HF API에서 요청 SHA와 응답 SHA가 정확히 일치했다.
- 본 설정은 4개 모델, 11개 양자화 조합으로 로드된다.
- 현재 본 설정 SHA-256: `cfd5bce8d8171e93f9937295c49bc91782fc913aa433cbf03ff78a7dffcedcc2`
- 현재 스모크 설정 SHA-256: `d408d09483f8636515ae109b966169d9e4e75bc441c3ced7e36f0eae355586bc`
- 스모크 평가셋: 4문항·4문서, 유형별 1문항, 잠금 파일 생성 성공

## 종단 실행 검증

Bllossom 3B Q4_K_M으로 다운로드 → F16 변환 → 양자화 → 서버 평가 → VRAM 측정 → 벤치마크 → 리포트의 전 경로를 실행했다.

| 산출물 | 결과 |
| --- | ---: |
| 원본 모델 디렉터리 | 6,442,823,749 bytes |
| F16 GGUF | 6,433,688,192 bytes |
| Q4_K_M GGUF | 2,019,377,792 bytes |
| 최대 VRAM | 3,713 MiB |
| 평가 요청 | 4/4 성공 |
| VRAM 표본 | 5개, 실패 0개 |
| 벤치마크 | PP/TG × 512/2048/4096 × 3회 성공 |
| 프로세스 정리 | 실행 종료 후 `llama-server` 0개 |
| 재개 | 완료 조합 1개 건너뛰기 성공 |

상세 실행 기록과 발견한 오류는 [`docs/SMOKE_TEST_LOG.md`](docs/SMOKE_TEST_LOG.md)에, 재현 가능한 요약은 [`reports/smoke/README.md`](reports/smoke/README.md)에 남겼다.
