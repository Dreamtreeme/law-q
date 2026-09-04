# 모델 라이선스 검토

검토일: 2026-09-05

> 이 문서는 기술적 후보 선정을 위한 검토 기록이며 법률 자문이 아니다. 실제 납품 전에는 별도 법률 검토가 필요하다.

## 배포 후보 판정

| 모델 | 라이선스 | 법률사무소 납품 후보 | 판단 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | Apache-2.0 | 가능 | 라이선스·저작권·NOTICE 의무 준수 필요 |
| EXAONE-3.5-7.8B-Instruct | EXAONE AI Model License Agreement 1.1 - NC | 제외 | 모델·파생물·출력의 상업적 사용은 별도 계약 없이 금지됨 |
| EEVE-Instruct-10.8B | Apache-2.0 | 가능 | EEVE와 기반 모델의 고지 조건을 함께 확인 |
| Bllossom-3B | Llama 3.2 Community License | 조건부 가능 | Llama 고지·브랜딩·Acceptable Use Policy 준수 필요 |

EXAONE은 연구용 성능 비교에는 남기되, 최종 리포트의 배포 가능 모델 순위에서는 제외한다. 별도 상업 라이선스를 취득한 경우에만 재검토한다.

Bllossom은 법률가의 내부 검토 보조와 최종 법률 판단을 대신하는 서비스를 구분한다. Llama 3.2 Acceptable Use Policy의 무허가 전문행위 제한을 제품 요구사항에 반영한다.

## 고정 원문과 검증값

| 자료 | 고정 출처 | SHA256 |
|---|---|---|
| Qwen LICENSE | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/a09a35458c702b33eeacc393d103063234e8bc28/LICENSE | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| EXAONE LICENSE | https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct/raw/553ea250b9a5317231459279d5847d6cf955b9aa/LICENSE | `294fd63925d81e9a11a2bd71fefecbfefb0d1a329eef7de28b96e49b6da4f67e` |
| EEVE 모델 카드 | https://huggingface.co/yanolja/YanoljaNEXT-EEVE-Instruct-10.8B/raw/b0c26460bf14cd825a3c4df2363cdfd63c47fb96/README.md | `73cd82fb504fc5bc6a4e187fc777c9cfe5c2d675863de8059546b57e1920b1fb` |
| Bllossom 모델 카드 | https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B/raw/e68fbb0d9c2a4031b0d61b14014eac1a4810ac2e/README.md | `e4f66078ee50adbeaae0dbe7bca7cbf058b6ef517045bb6cfdcf6883f26d3a6b` |
| Llama 3.2 LICENSE | https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE | `8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74` |
| Llama 3.2 Use Policy | https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/USE_POLICY.md | `40e2777d7faa6beaf98400654170f414d8ab29b921b5163ad4ea0a1d39894201` |

Llama 원문 URL은 모델 revision에 묶이지 않으므로 본 실험 검토 시점의 SHA256을 함께 기록한다.

## EEVE Q8 제외

EEVE 10.8B Q8은 가중치만 약 11GB 수준으로 예상되어 10GB VRAM에서 전체 GPU 오프로드와 4096 컨텍스트를 동시에 만족하기 어렵다. CPU 오프로드는 다른 조합과 속도 조건이 달라지므로 기본 실험 행렬에서 제외한다.
