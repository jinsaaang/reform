# Pre-Judge Forecasting Candidate Set

This directory contains a unified candidate set for later finance/economics filtering.

## Files

- `judge_units.jsonl`: rows to send to the LLM judge.
- `task_index.jsonl`: forecasting task rows for later evaluation.
- `unified_tasks.jsonl`: one task-level file with judge context, answers, and empty judge output columns.
- `manifest.json`: source counts and build summary.
- `raw_cache/`: downloaded source parquet files used to build the set.

## Judge Output Contract

Append only these two columns to `judge_units.jsonl` results:

```text
is_finance_econ
confidence
```

`is_finance_econ` should be a yes/no boolean. `confidence` should be a number between 0 and 1.

## Row Counts

```text
judge_units.jsonl: 71,078 rows
task_index.jsonl:  88,031 rows
unified_tasks.jsonl: 88,031 rows
```

Source breakdown:

```text
BTF-2:           1,417 judge rows /  1,417 task rows
OpenForesight:  55,301 judge rows / 55,301 task rows
Daily Oracle:    8,832 judge rows /  8,832 task rows
ForecastBench:   5,528 judge rows / 22,481 task rows
```

Daily Oracle was pre-filtered to `category == "Economics & Business"` before inclusion.

ForecastBench is snapshot data. The task index keeps unique `(id, freeze_datetime)` snapshots, while judge rows are deduplicated by normalized question text so repeated forecast dates for the same query are judged once.

## Judge Input Columns

All sources are mapped into the same judge schema:

```text
judge_uid
source_dataset
source_id
source_split
question
question_type
choices
background
resolution_criteria
forecast_date
raw_category
source_url
representative_task_uid
task_count
```

Answers, post-resolution explanations, SOTA rationales, and future article text are excluded from judge input.

## Join Logic

After judging, join labels back to `task_index.jsonl` on:

```text
judge_uid
```

This propagates one ForecastBench domain decision to all task snapshots for the same normalized question.

If you want one-file workflow instead, use `unified_tasks.jsonl` and fill:

```text
is_finance_econ
confidence
```

For judging from `unified_tasks.jsonl`, pass only the non-answer context fields to the LLM judge: `question`, `question_type`, `choices`, `background`, `resolution_criteria`, `forecast_date`, `raw_category`, and `source_url`.
