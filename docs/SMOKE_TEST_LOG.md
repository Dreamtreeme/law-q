# 스모크 테스트 실행 기록

## 2026-09-04 사전 검증 실행

- 실행: `validation-pipeline`
- 결과: 11개 조합 모두 단계 2에서 실패
- 원인: `../llama.cpp/convert_hf_to_gguf.py`가 존재하지 않음
- 의미: 결과 CSV와 실패 JSONL의 보존은 확인했으나 실제 모델 실행은 검증하지 못함

## Bllossom Q4 종단 스모크

### 모델 준비

- 실행: `smoke-prepare-20260905`
- HF revision: `e68fbb0d9c2a4031b0d61b14014eac1a4810ac2e` 일치 확인
- 원본: 6,442,823,749 bytes
- F16: 6,433,688,192 bytes, SHA-256 `a6afd2307ae67129b554c60c04ffa010c3cf9299f410ba08acd4c22ff21ac1f3`
- Q4_K_M: 2,019,377,792 bytes, SHA-256 `f67461ae9a05de4db9727a96cd03f5e85a7320cfec9556f27b8620628c6326cf`
- 관찰: 저장소가 Xet Storage를 사용하지만 `hf_xet`가 없어 `hf_transfer` 대신 일반 HTTP로 자동 폴백했다. 다운로드 자체는 성공했다.
- 관찰: 변환기가 Bllossom tokenizer의 `<|begin_of_text|>` separator token을 알 수 없다는 경고를 남겼으나 GGUF 변환과 채팅 템플릿 기록은 성공했다.

### 1차 종단 실행 — 부분 실패

- 실행: `smoke-bllossom-q4-20260905`
- 결과: 평가 2/4 성공, 벤치마크 성공, 조합 상태 `partial_failure`
- 증상: 한국어 SSE 이벤트의 JSON 문자열이 중간에서 잘려 `JSONDecodeError` 발생
- 원인: llama-server 응답에 SSE charset이 없을 때 `requests.iter_lines(decode_unicode=True)`가 ISO-8859-1로 먼저 디코딩했다. UTF-8 continuation byte `0x85` 등이 Unicode 줄 구분자로 해석되어 한 이벤트가 분할됐다.
- 수정: HTTP 스트림을 bytes 상태로 줄 분리한 다음 완성된 SSE 행만 UTF-8로 디코딩하도록 변경하고 회귀 테스트를 추가했다.

### `--resume` 재실행 — 성공

- 평가: 요청 4/4 성공, 실패 0, 전체 자동점수 66.67%
- 유형 점수: 유형1 100%, 유형2 0%, 유형3 66.67%, 유형4 100%
- 지연: 평균 TTFT 0.034925초, 평균 전체 응답 0.208197초
- VRAM: 표본 5개, 수집 실패 0, 최대 3,713 MiB
- PP 평균: 512=10,051.43, 2048=9,789.51, 4096=9,168.55 tokens/s
- TG 평균: 512=231.41, 2048=213.94, 4096=198.82 tokens/s
- 각 벤치마크 조건은 워밍업 뒤 3회 측정했고 평균과 표준편차를 저장했다.
- 종료 뒤 `llama-server` 프로세스가 남지 않았다.
- 두 번째 `--resume`은 완료 조합을 실행하지 않고 `skipped_by_resume: 1`을 기록했다.
- Markdown·CSV·JSON·SVG 리포트가 모두 생성됐다.

### 평가셋 잠금 적용 후 재검증

- 실행: `smoke-locked-20260905`
- 현재 스모크 설정과 `dataset.lock.json`의 해시 검증을 먼저 통과했다.
- 전 단계 1/1 성공, 평가 요청 4/4 성공, VRAM 표본 5개, 최대 3,713 MiB였다.
- 평균 TTFT 0.035098초, 평균 전체 응답 0.207121초였다.
- PP 평균: 512=10,042.76, 2048=9,747.92, 4096=9,180.45 tokens/s
- TG 평균: 512=232.00, 2048=214.02, 4096=198.80 tokens/s
- 같은 설정으로 `--resume`을 다시 실행해 1개 완료 조합이 건너뛰어지는 것을 확인했다.
- 현재 본 설정으로 `final-preflight-20260905`를 실행했을 때 실제 60문항이 없음을 `dataset_preflight` 실패로 기록하고, 모델 작업은 0개만 수행했다.

스모크 데이터는 합성 문서 네 개로 파이프라인만 검증한 결과다. 이 정확도를 모델 선정 근거로 사용하지 않는다.
