# Final Finance/Economics Forecasting Dataset

## Files

- `finance_econ_tasks_final.jsonl`

This is the final filtered dataset intended for use. It contains forecasting tasks that:

1. were judged as finance/economics relevant by `Qwen/Qwen3-4B`;
2. have an `answer`;
3. exclude `ForecastBench`, because those rows do not currently include resolved answers.

## Row Count

```text
finance_econ_tasks_final.jsonl: 9,678 rows
```

Source breakdown:

```text
Daily Oracle:   6,175
OpenForesight:  3,048
BTF-2:            455
ForecastBench:      0
```

Answer/date availability:

```text
Rows with answer:                    9,678 / 9,678
Rows with forecast_date:             9,675 / 9,678
Rows with resolution_date:           9,223 / 9,678
Rows with answer but no resolution_date: 455
```

The 455 rows without `resolution_date` are all from BTF-2 and still include answers.

## Filtering Process

The pre-judge candidate set was built from four sources:

```text
BTF-2
OpenForesight
Daily Oracle
ForecastBench
```

The judge was run on `data/prejudge/judge_units.jsonl`, which contains deduplicated judge units. The model received only forecast-time/domain context:

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

The model did not receive `answer`, `resolution_date`, or `extra`.

Judge output schema:

```json
{"is_finance_econ": true, "confidence": 0.95}
```

The final checked-in file was produced by:

1. joining judge labels back to task-level rows by `judge_uid`;
2. keeping rows where `is_finance_econ == true`;
3. removing all `ForecastBench` rows because their answers are currently missing.

## Judge Result Summary

Judge-unit level:

```text
Total judge units: 71,078
Finance/econ true: 11,138
Finance/econ false: 59,940
Invalid/missing judge results: 0
```

Task-level before removing ForecastBench:

```text
Finance/econ rows: 14,883
ForecastBench rows removed: 5,205
Final rows: 9,678
```

Confidence distribution in the final file:

```text
0.90:                   7
0.95:               9,131
0.98:                   3
0.99:                   9
0.999:                  4
0.9999999999999999:     3
1.0:                  521
```
