# 평가셋 형식

평가셋과 예측 결과는 한 줄에 JSON 객체 하나를 쓰는 JSONL을 기본으로 사용합니다. JSON 배열로 저장된 `.json` 파일도 채점기가 읽을 수 있습니다.

## 평가 항목

공통 필드는 다음과 같습니다.

```json
{
  "id": "L001",
  "type": 1,
  "context_doc": "근로기준법_53조.txt",
  "question": "...",
  "answer_keywords": ["12시간", "12"],
  "scoring": "keyword_any"
}
```

- `id`: 중복되지 않는 문항 ID. `L`과 세 자리 이상의 숫자로 구성합니다.
- `type`: 1=사실확인, 2=조건적용, 3=정보추출, 4=함정.
- `context_doc`: `eval/` 기준 문서 파일명 또는 상대 경로.
- `question`: 모델에 전달할 질문.
- `answer_keywords`: `keyword_any`, `keyword_ratio`, `refusal`에서 찾을 표현 목록.
- `scoring`: `keyword_any`, `keyword_ratio`, `json_field`, `refusal` 중 하나.
- `answer_fields`: `json_field`에서만 필수인 기대 JSON 필드와 값.

전체 JSON Schema는 `evaluation-item.schema.json`에 있습니다.

### json_field 예시

```json
{
  "id": "L003",
  "type": 3,
  "context_doc": "판결문.txt",
  "question": "사건 정보를 JSON으로 추출하라.",
  "answer_keywords": [],
  "answer_fields": {
    "사건번호": "2026가단1234",
    "원고": "김민수",
    "인용금액": 5000000
  },
  "scoring": "json_field"
}
```

`answer_fields`는 각 필드를 동일한 비중으로 채점합니다. 중첩 객체는 `당사자.원고` 같은 점 표기 경로도 사용할 수 있습니다.

## 예측 결과

```json
{"id":"L001","response":"합의하면 1주 12시간까지 연장할 수 있습니다."}
```

문항별 `id`와 모델의 원문 응답인 `response`가 필요합니다. 예측이 누락된 문항은 0점으로 계산하고, 평가셋에 없는 추가 예측은 무시하되 개수를 리포트에 기록합니다.

## 채점

```powershell
python scripts\score_answers.py `
  --dataset eval\sample_questions.jsonl `
  --predictions eval\sample_predictions.jsonl `
  --output results\sample-scores.json
```

