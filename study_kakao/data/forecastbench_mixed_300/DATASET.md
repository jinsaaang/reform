# ForecastBench Mixed-Domain 300

This directory contains a reproducible 200-memory / 100-test selection from
ForecastBench.

## Composition

| Split | Structured dataset | Market/event | Total |
|---|---:|---:|---:|
| Memory | 100 | 100 | 200 |
| Test | 50 | 50 | 100 |

Structured sources are FRED, Yahoo Finance, DBnomics, ACLED, and Wikipedia.
Market/event sources are Polymarket, Manifold, Metaculus, and INFER.

Only resolved, scalar binary questions are included. ForecastBench combination
questions are excluded. A `(source, source_question_id)` family appears at most
once in a split and never appears in both memory and test.

## Temporal split

- Memory forecast round: `2024-07-21`
- Latest memory resolution: `2025-03-28`
- Test forecast round: `2025-03-30`
- Latest retained test resolution: `2026-07-21`

Thus every memory outcome is available before the test forecast round.

## Files

- `memory_questions.jsonl`: 200 resolved memory questions, including labels.
- `test_questions.jsonl`: 100 public forecasting questions without labels.
- `test_answers.jsonl`: isolated answer key for evaluation only.
- `manifest.json`: pinned source commit, quotas, counts, checksums, and checks.
- `worldreasoner_memory_questions.jsonl`: resolved memory questions converted to
  the input contract used by the hindsight DAG builder.
- `worldreasoner_memory_questions.manifest.json`: conversion counts, checksums,
  domain assignments, and the resolution timestamp policy.

Never provide `test_answers.jsonl` to a forecaster.

## Rebuild

From the repository root:

```powershell
python study_kakao/experiments/build_forecastbench_mixed_300.py
```

The builder uses only the Python standard library and deterministic seed `27`.

## Build hindsight DAGs

First convert the 200 resolved memory questions:

```powershell
study_kakao/.venv/Scripts/python.exe `
  study_kakao/experiments/prepare_forecastbench_dag_questions.py
```

Then run the strict, resumable DAG supervisor with mixed-domain search queries:

```powershell
refactored_exempler/research/data_construction/.venv/Scripts/python.exe `
  refactored_exempler/research/forecaster/memory/build.py `
  --experiment-dir study_kakao/runs/forecastbench_mixed_300_dags `
  --memory-questions study_kakao/data/forecastbench_mixed_300/worldreasoner_memory_questions.jsonl `
  --search-query-mode original `
  --min-graph-events 7 `
  --min-graph-edges 6
```

Only the resolved memory split is eligible for hindsight DAG construction. The
public test split intentionally has no labels and must not be passed to this
builder.

ForecastBench supplies a target resolution date, not a separately observed
publication timestamp for the answer. The adapter therefore uses 23:59:59 UTC
on that target date as the evidence cutoff and records this approximation in
every question's metadata and in the conversion manifest.

## Source and license

The source data is the
[ForecastBench datasets repository](https://github.com/forecastingresearch/forecastbench-datasets),
distributed under CC BY-SA 4.0. The exact source commit is recorded in
`manifest.json`.
