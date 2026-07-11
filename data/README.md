# 최종 금융/경제 예측 데이터셋

## 파일

- `finance_econ_tasks_final.jsonl`

이 파일이 최종 사용 대상 데이터셋입니다. 포함 조건은 다음과 같습니다.

1. `Qwen/Qwen3-4B` judge가 금융/경제 관련 질문으로 판정한 row
2. `answer`가 존재하는 row
3. `ForecastBench`를 제외한 row

`ForecastBench`는 금융/경제로 판정된 row가 일부 있었지만, 현재 파일 안에 resolved answer가 없어서 최종 데이터셋에서는 제거했습니다.

## Row 수

```text
finance_econ_tasks_final.jsonl: 9,678 rows
```

소스별 구성:

```text
Daily Oracle:   6,175
OpenForesight:  3,048
BTF-2:            455
ForecastBench:      0
```

정답 및 날짜 상태:

```text
answer가 있는 row:              9,678 / 9,678
forecast_date가 있는 row:       9,675 / 9,678
resolution_date가 있는 row:     9,223 / 9,678
answer는 있지만 resolution_date가 없는 row: 455
```

`resolution_date`가 없는 455개 row는 모두 BTF-2에서 온 row이며, `answer`는 포함되어 있습니다.

## 필터링 방식

초기 후보셋은 다음 네 소스에서 만들어졌습니다.

```text
BTF-2
OpenForesight
Daily Oracle
ForecastBench
```

LLM judge는 중복을 줄인 judge unit 단위로 실행했습니다. Judge에는 예측 시점에서 볼 수 있는 도메인 문맥만 전달했습니다.

Judge 입력 필드:

```text
source_dataset
question
question_type
choices
background
resolution_criteria
forecast_date
raw_category
source_url
```

Judge에는 다음 필드를 전달하지 않았습니다.

```text
answer
resolution_date
extra
```

Judge 출력 형식:

```json
{"is_finance_econ": true, "confidence": 0.95}
```

최종 파일은 다음 순서로 만들었습니다.

1. `judge_uid` 기준으로 judge 결과를 task-level row에 join
2. `is_finance_econ == true`인 row만 유지
3. resolved answer가 없는 `ForecastBench` row 제거

## Judge 결과 요약

Judge unit 기준:

```text
총 judge unit: 71,078
finance/econ true: 11,138
finance/econ false: 59,940
invalid/missing judge result: 0
```

Task-level 기준:

```text
ForecastBench 제거 전 finance/econ row: 14,883
제거한 ForecastBench row:              5,205
최종 row:                              9,678
```

최종 파일의 confidence 분포:

```text
0.90:                   7
0.95:               9,131
0.98:                   3
0.99:                   9
0.999:                  4
0.9999999999999999:     3
1.0:                  521
```

## 최종 데이터 형태

`finance_econ_tasks_final.jsonl`은 JSONL 형식입니다. 한 줄이 하나의 forecasting task입니다.

주요 컬럼:

```text
task_uid: row 단위 고유 ID
judge_uid: LLM judge에 사용한 고유 ID
source_dataset: 원본 데이터셋 이름
question: 예측 질문
question_type: binary, multiple_choice, open_ended 등 질문 형식
choices: 선택지가 있는 경우의 보기
background: 질문 배경 정보
resolution_criteria: 정답 판정 기준
forecast_date: 예측 시점 또는 질문 기준 시점
resolution_date: 정답이 확인된 날짜
answer: 정답
answer_type: 정답 형식
source_url: 원문 URL
raw_category: 원본 카테고리
extra: 원본 데이터셋별 추가 정보
is_finance_econ: 금융/경제 관련 여부
confidence: judge 판정 confidence
```

예시 row는 다음과 같은 구조입니다.

```json
{
  "task_uid": "...",
  "source_dataset": "daily_oracle",
  "question": "...",
  "forecast_date": "2024-01-01",
  "resolution_date": "2024-01-15",
  "answer": "...",
  "is_finance_econ": true,
  "confidence": 0.95
}
```
