# 실행 환경 검증 결과

## 검증 일시

- UTC: 2026-09-04 09:31:21
- 로컬 시간: 2026-09-04 18:31:21 (대한민국 표준시, UTC+09:00)

## 검증 요약

| 항목 | 결과 | 상태 |
|---|---|---|
| 운영체제 | Windows 11 (`10.0.26200`) | 정상 |
| 시스템 아키텍처 | AMD64 | 정상 |
| Python | 3.13.14, 64-bit | 정상 |
| 프로젝트 가상환경 | `.venv` | 정상 |
| PyYAML | 6.0.3 | 정상 |
| GPU | NVIDIA GeForce RTX 3080 | 정상 |
| GPU VRAM | 10,240 MiB | 정상 |
| NVIDIA 드라이버 | 591.86 | 정상 |
| CUDA Compute Capability | 8.6 | 정상 |
| 논리 CPU 수 | 20 | 정상 |
| 실험 설정 로드 | 성공 | 정상 |
| 실험 조합 생성 | 11개 | 정상 |
| Python 문법 검사 | 통과 | 정상 |
| llama.cpp 커밋 조회 | 저장소 경로를 찾지 못함 | 확인 필요 |
| 프로젝트 Git 커밋 조회 | 아직 Git 저장소가 아님 | 확인 필요 |

## Python 환경

- 설치 경로: `C:\Users\psg\AppData\Local\Programs\Python\Python313\python.exe`
- 버전: `Python 3.13.14`
- 가상환경: `C:\Users\psg\Desktop\개발\law-q\.venv`
- 설치된 필수 패키지: `PyYAML 6.0.3`

시스템 PATH에서는 `python` 명령을 찾을 수 없으며, `C:\Windows\py.exe` 런처도 설치된 Python을 자동으로 인식하지 못했다. 프로젝트 내부 가상환경의 Python 실행 파일을 직접 사용했을 때는 정상 실행됐다.

## GPU 환경

`nvidia-smi` 조회 결과:

```text
GPU: NVIDIA GeForce RTX 3080
VRAM: 10240 MiB
Driver: 591.86
Compute Capability: 8.6
```

## 프로젝트 스크립트 검증

다음 파일의 Python 문법 검사가 통과했다.

- `scripts/common.py`
- `scripts/capture_env.py`
- `scripts/plan_experiments.py`

`scripts/plan_experiments.py` 실행 결과, `experiment.yaml`에서 총 11개의 모델·양자화 실험 조합이 정상 생성됐다.

```text
 1. qwen2.5-7b-instruct / Q4_K_M
 2. qwen2.5-7b-instruct / Q5_K_M
 3. qwen2.5-7b-instruct / Q8_0
 4. exaone-3.5-7.8b-instruct / Q4_K_M
 5. exaone-3.5-7.8b-instruct / Q5_K_M
 6. exaone-3.5-7.8b-instruct / Q8_0
 7. eeve-instruct-10.8b / Q4_K_M
 8. eeve-instruct-10.8b / Q5_K_M
 9. bllossom-3b / Q4_K_M
10. bllossom-3b / Q5_K_M
11. bllossom-3b / Q8_0
```

환경 정보 수집 함수에서 다음 항목이 정상 기록되는 것을 확인했다.

- UTC 및 로컬 실행 시각
- 설정 파일 경로와 SHA-256
- 실험 조합 수
- GPU 모델, 드라이버, VRAM, Compute Capability
- OS, Python, CPU 정보
- llama.cpp 및 프로젝트 Git 커밋과 작업 트리 상태

검증 당시 `experiment.yaml`의 SHA-256은 다음과 같다.

```text
e9b270200a2c5e1e49a212b4e243a07f21244689246d2854e00dc7f32feede43
```

## 확인이 필요한 항목

### llama.cpp 경로

현재 `experiment.yaml`에는 다음 경로가 지정되어 있다.

```yaml
paths:
  llama_cpp: ../llama.cpp
```

검증 시 해당 위치에서 llama.cpp 저장소를 찾지 못해 커밋 해시가 기록되지 않았다. 실제 llama.cpp 저장소 위치로 값을 변경하면 이후 실행부터 커밋 해시와 작업 트리 상태가 자동 기록된다.

### 프로젝트 Git 상태

검증 당시 프로젝트 디렉토리는 Git 저장소가 아니었다. Git 저장소로 초기화하면 프로젝트 커밋과 dirty 상태도 자동 기록된다.
