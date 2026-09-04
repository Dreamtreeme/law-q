# Korean Legal QA Quantization Lab

10GB VRAM 환경에서 한국어 법률 문서 QA용 모델과 GGUF 양자화 조합을 비교하기 위한 실험 프로젝트입니다.

## 구조

```text
.
├── experiment.yaml       # 실험 대상과 모든 경로/실행 옵션의 단일 설정 지점
├── models/               # Hugging Face 원본 모델
├── gguf/                 # 변환 및 양자화된 GGUF
├── eval/                 # 평가셋
├── results/              # 실행별 환경 정보와 결과
└── scripts/
    ├── common.py         # 설정, 실험 매트릭스, 환경 기록 공통 함수
    ├── capture_env.py    # 실행 환경 기록 CLI
    ├── prepare_models.py # HF 다운로드, F16 변환 및 양자화
    └── plan_experiments.py
```

## 시작하기

Python 3.10 이상이 필요합니다.

```powershell
$pythonExe = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
& $pythonExe -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\plan_experiments.py
python scripts\capture_env.py
```

`experiment.yaml`의 `models` 목록과 각 모델의 `quantizations`만 수정하면 실험 매트릭스가 바뀝니다. 경로와 llama.cpp 실행 옵션도 같은 파일에서 관리합니다.

`capture_env.py`는 `results/<실행시각>/`에 다음 파일을 생성합니다.

- `environment.json`: llama.cpp 및 프로젝트 커밋, dirty 여부, GPU/드라이버/VRAM, 실행 시각, OS/Python 정보, 설정 SHA-256
- `experiment.yaml`: 해당 실행에 실제 사용된 설정 스냅샷

본 실험에서는 재현성을 위해 각 모델의 `revision: main`을 Hugging Face 커밋 SHA로 교체하는 것을 권장합니다.

## 모델 다운로드 및 GGUF 생성

먼저 `experiment.yaml`의 `paths.llama_cpp`와 필요하면 `paths.llama_quantize`를 실제 llama.cpp 위치에 맞춥니다. 비공개 또는 gated 모델은 `HF_TOKEN` 환경 변수도 설정해야 합니다.

```powershell
python scripts\prepare_models.py --dry-run
python scripts\prepare_models.py
```

스크립트는 `HF_HUB_ENABLE_HF_TRANSFER=1`을 `huggingface_hub` import 전에 설정합니다. `models/<모델명>/`에 원본을 내려받고 `gguf/<모델명>/`에 F16과 설정된 양자화 파일을 만듭니다. 완료된 다운로드와 0바이트보다 큰 기존 GGUF는 건너뛰며, 미완성 GGUF는 `.partial.gguf`로 분리합니다.

`huggingface_hub` 1.x에서는 `hf_transfer` 지원이 제거되었으므로, 이 프로젝트는 요청한 전송 방식을 유지하기 위해 `huggingface_hub>=0.28,<0.32`를 사용합니다.

각 실행의 `results/<실행시각>/`에는 다음 기록이 남습니다.

- `prepare.log`: 모델명과 단계명이 포함된 상세 로그
- `model-preparation.json`: 단계별 성공/실패/건너뛰기, 소요 시간, HF 실제 revision, 생성 파일의 실제 byte/GiB 크기
- `failures.jsonl`: 각 실패의 모델명, 단계, 오류 유형·메시지, traceback을 즉시 추가 기록한 append-only 로그(실패가 있을 때 생성)
- `environment.json` 및 `experiment.yaml`: 실행 환경과 설정 스냅샷

한 모델의 다운로드나 F16 변환이 실패하면 그 모델의 후속 단계는 `blocked`로 기록하고 다음 모델로 계속 진행합니다. 개별 양자화 실패도 다른 양자화 작업을 막지 않으며, 하나라도 실패하면 프로세스는 종료 코드 1을 반환합니다.

## 평가셋과 채점

평가 항목 스키마와 예시는 `eval/`에 있습니다. 채점기는 JSONL과 JSON 배열을 모두 지원합니다.

```powershell
python scripts\score_answers.py `
  --dataset eval\sample_questions.jsonl `
  --predictions eval\sample_predictions.jsonl `
  --output results\sample-scores.json
```

지원 방식은 `keyword_any`, `keyword_ratio`, `json_field`, `refusal`입니다. 결과 JSON에는 문항별 점수와 매칭 근거, 유형별 점수, 채점 방식별 점수, 전체 점수, JSON 파싱 실패 및 누락·추가 예측 수가 포함됩니다. 자세한 형식은 `eval/README.md`를 참고하세요.

## llama-server 평가 실행

`experiment.yaml`의 `runtime.gpu_layers`에서 GPU 오프로드 레이어 수를 조정하고, `evaluation.dataset`에 실제 평가셋을 지정합니다. 샘플링은 같은 설정의 `temperature: 0.0`과 `seed`를 모든 문항에 적용합니다.

```powershell
python scripts\run_evaluation.py --dry-run
python scripts\run_evaluation.py
```

특정 평가셋으로 실행 계획만 확인할 수도 있습니다.

```powershell
python scripts\run_evaluation.py `
  --dataset eval\sample_questions.jsonl `
  --dry-run
```

각 모델·양자화 조합마다 별도 `llama-server`를 시작하고 `/health`가 HTTP 200과 `status=ok`를 반환한 뒤 문항을 순차 실행합니다. 응답은 스트리밍으로 받아 첫 콘텐츠 도착 시간(TTFT)과 전체 응답 시간을 측정합니다. 동시에 `nvidia-smi`를 주기적으로 호출하여 전체 원본 샘플과 GPU별·전체 최대 VRAM을 기록합니다.

조합별 결과는 `results/<실행시각>/evaluations/<모델>/<양자화>/` 아래에 저장됩니다.

- `responses.jsonl`: 질문, 응답 원문, TTFT, 전체 시간, sampling 값, API usage
- `scores.json`: 문항·유형·전체 채점 결과
- `server.log`: llama-server 표준 출력 및 오류
- `vram-samples.jsonl`: 시각별 GPU VRAM 원본 측정값
- `vram-summary.json`: 최대 VRAM과 샘플링 오류
- `summary.json`: 서버 시작·종료와 조합 실행 요약

정상 경로에서는 조합 종료 후 interrupt 신호를 보내며, 제한 시간 안에 끝나지 않으면 terminate와 프로세스 트리 강제 종료를 차례로 적용합니다. 평가 요청이나 문맥 파일 처리 중 예외가 발생해도 `finally`에서 서버와 VRAM 샘플러를 종료합니다.

## llama-bench 속도 측정

`experiment.yaml`의 `benchmark`에서 프롬프트 길이, 생성 토큰 수, 반복 횟수, batch 크기와 시간 제한을 관리합니다. `repetitions`는 3 이상만 허용합니다.

```powershell
python scripts\run_benchmarks.py --dry-run
python scripts\run_benchmarks.py
```

각 GGUF에 대해 다음 두 단계를 별도로 실행합니다.

- Prompt processing: `-p 512,2048,4096 -n 0`
- Token generation: `-p 0 -n 128 -d 512,2048,4096`

TG의 `-d`는 해당 길이만큼 KV cache를 미리 채워 실제 컨텍스트 깊이에 따른 생성 속도 변화를 측정합니다. `--no-warmup`을 전달하지 않아 llama-bench의 기본 워밍업이 각 본 측정 전에 실행되며, 통계에는 `-r`로 지정한 본 반복값만 사용합니다.

실행별 `benchmark-results.json`에는 PP/TG, 프롬프트 길이, tokens/s 평균·표준편차·원본 반복값, latency 평균·표준편차, llama.cpp 빌드 커밋과 backend가 저장됩니다. 조합별 원본 stdout JSON, stderr 로그, 요약 및 실패 기록은 `benchmarks/<모델>/<양자화>/` 아래에 남습니다.

## 단계 2~5 통합 실행

원본 모델 다운로드(단계 1)가 끝난 뒤 다음 명령으로 GGUF 준비, 평가, 속도 벤치마크와 CSV 집계를 순서대로 실행합니다.

```powershell
python scripts\run_pipeline.py --run-name experiment-001
```

모든 모델·양자화 조합을 설정 순서대로 처리하며 콘솔에 `[현재/전체]` 진행률을 표시합니다. 조합이 끝날 때마다 `results/<run-name>/results.csv` 전체를 임시 파일에 쓰고 원자적으로 교체하므로 중단 전 완료 결과가 보존됩니다. 한 단계 또는 조합이 실패해도 다음 조합으로 계속 진행합니다. 평가가 실패해도 독립적인 속도 벤치마크는 시도하며, GGUF 준비가 실패한 경우에만 두 후속 단계가 `blocked`가 됩니다.

중단된 실행은 이름을 지정해 재개할 수 있습니다.

```powershell
python scripts\run_pipeline.py --run-name experiment-001 --resume
```

`--run-name` 없이 `--resume`만 사용하면 가장 최근 `pipeline-*` 실행을 선택합니다. CSV에서 `status=success`인 조합은 건너뛰고, 실패·부분 실패·중단 조합은 다시 실행합니다. 재시도 시 해당 조합의 이전 응답·VRAM 원본은 새 시도의 값으로 교체됩니다.

CSV에는 단계별 상태, 오류, 정확도와 유형별 점수, TTFT·응답시간, 최대 VRAM, 512/2048/4096별 PP·TG 평균 및 표준편차가 한 행에 저장됩니다. 상세 원본은 같은 실행 폴더의 `evaluations/`, `benchmarks/`, `failures.jsonl`, `pipeline.log`에서 확인할 수 있습니다.

## 결과 분석 리포트

통합 실행 결과를 Markdown 표와 그래프로 변환합니다.

```powershell
python scripts\generate_report.py --run-name experiment-001
```

`--input`으로 CSV를 직접 지정할 수 있으며 아무 옵션도 주지 않으면 가장 최근 `results/*/results.csv`를 사용합니다. 기본 출력 위치는 입력 CSV 옆의 `report/`입니다.

```powershell
python scripts\generate_report.py `
  --input results\experiment-001\results.csv `
  --output-dir results\experiment-001\report
```

생성 파일:

- `REPORT.md`: 조합별 종합 표, 양자화 손실 표, 동일 VRAM 그룹 표와 그래프
- `analysis.json`: 모든 계산 결과와 데이터 누락 정보
- `combination-summary.csv`, `quantization-loss.csv`, `quantization-loss-detail.csv`, `memory-budget-groups.csv`
- `accuracy-vs-speed.svg`, `type-scores-by-quantization.svg`

양자화 손실은 같은 모델의 F16을 우선 기준으로 사용하고, F16 결과가 없으면 Q8을 기준으로 계산합니다. 둘 다 없는 모델은 손실 표에서 제외하고 리포트의 데이터 품질 메모에 기록합니다. 동일 메모리 그룹 허용치는 `experiment.yaml`의 `report.similar_vram_tolerance_mib`에서 조정할 수 있습니다.
