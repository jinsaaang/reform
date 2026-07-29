# HGF 실행 및 데이터 안내

## 프로젝트 루트

모든 공개 코드와 데이터는 `study_kakao/` 안에서 완결된다. 다른 폴더의
코드나 데이터를 참조하지 않는다.

```text
study_kakao/
├── README.md
├── experiments.md
├── instruction.md
├── artifacts/
│   ├── exemplars/
│   └── semantic_lessons/
├── configs/
├── data/
│   ├── questions/
│   ├── memory_bank/
│   ├── dags/
│   └── evidence/
│       ├── e0/
│       └── e1/
├── experiments/
├── src/hgf/
└── tests/
```

## 고정 입력

- Memory questions: 200개
- Test questions: 100개
- 최종 DAG memory bank: 200개
- E0 question-only evidence DB: 100개
- E1 factor-guided evidence DB: 100개
- Worked exemplar: test question별 1개, 총 100개

Worked exemplar와 memory-bank 선택은 `artifacts/`와 `data/`에 고정되어
있으며 forecasting 실행 중 다시 생성하지 않는다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

API 호출이 필요한 실행에서는 프로젝트 루트의 `.env`에
`OPENROUTER_API_KEY`를 설정한다. `.env`는 Git에 포함하지 않는다.

## 검증

```bash
PYTHONPATH=src python -m hgf.verify
pytest -q
```

검증은 질문 100개, memory bank 200개, 고정 exemplar 100개, E0/E1 evidence
DB 및 공개 소스의 독립성을 확인한다.

## HGF 실행

```bash
PYTHONPATH=src python -m hgf.runner \
  --model google/gemini-2.5-flash-lite \
  --limit 100 \
  --workers 4
```

기본 결과는 `runs/hgf/`에 저장된다.

## HGF와 여섯 baseline 실행

```bash
PYTHONPATH=src python experiments/run_main_table.py
```

비교 방법은 다음 일곱 개다.

1. Search Only
2. Factor Memory
3. Case Memory
4. Text Memory
5. Direct DAG
6. Prospective DAG
7. HGF

## 논문 실험

```bash
PYTHONPATH=src python experiments/run_all_paper_experiments.py --dry-run
PYTHONPATH=src python experiments/run_all_paper_experiments.py
```

세부 실행법과 산출물 구조는 `experiments/EXPERIMENTS_USAGE.md`를 따른다.
실행 결과와 로그는 `runs/` 아래에 저장되며 Git에는 포함하지 않는다.
