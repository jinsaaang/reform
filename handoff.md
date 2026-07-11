# Handoff: Finance/Economics Filtering for Forecasting Candidate Set

## Current Goal

We built a unified candidate set from four public forecasting sources and now need to filter it into a finance/economics forecasting dataset using an LLM judge.

Judge model target:

```text
Qwen3-4B via vLLM
4x NVIDIA A5000 available
gpu_memory_utilization = 0.90
Do not use the GPUs if another process is already running on them.
```

The judge output should be intentionally simple:

```text
is_finance_econ: boolean
confidence: float between 0 and 1
```

## Repository State

Root:

```text
/Users/suhwan/Desktop/research/kakaobank
```

Important files:

```text
plan.md
handoff.md
kakao_0706 (1).pdf
scripts/build_prejudge_set.py
data/prejudge/unified_tasks.jsonl
data/prejudge/judge_units.jsonl
data/prejudge/task_index.jsonl
data/prejudge/manifest.json
data/prejudge/README.md
```

`plan.md` is the renamed research/plan document.

## Built Dataset

The easiest file to work from is:

```text
data/prejudge/unified_tasks.jsonl
```

It has one row per forecasting task.

```text
total rows: 88,031
size: ~140 MB
```

Source breakdown:

```text
BTF-2:           1,417 rows
OpenForesight:  55,301 rows
Daily Oracle:    8,832 rows
ForecastBench:  22,481 rows
```

Daily Oracle was pre-filtered to:

```text
category == "Economics & Business"
```

ForecastBench is snapshot data. The same question can appear at multiple `freeze_datetime` values. We keep those as separate task rows because each row represents a different forecast-time cutoff.

## Unified File Schema

Each row in `data/prejudge/unified_tasks.jsonl` has:

```text
task_uid
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
resolution_date
answer
answer_type
source_url
raw_category
representative_task_uid
task_count_for_judge_uid
extra
is_finance_econ
confidence
```

The final two columns currently contain `null`. They are placeholders for judge output.

Important:

- Do not send `answer`, `resolution_date`, or `extra` to the judge.
- Do not send post-resolution explanations or future article text.
- Judge only from forecast-time/domain context.

Recommended judge input fields:

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

## Judge Unit Choice

There are two possible workflows.

### Recommended Workflow: Unique Judge Units

Use:

```text
data/prejudge/judge_units.jsonl
```

Rows:

```text
71,078
```

This avoids duplicate judge calls for ForecastBench repeated snapshots. After judging, join results back to `unified_tasks.jsonl` using:

```text
judge_uid
```

This is the recommended path.

### Simpler Workflow: Direct Task-Level Judging

Use:

```text
data/prejudge/unified_tasks.jsonl
```

Rows:

```text
88,031
```

This is simpler but will repeat some ForecastBench questions across dates. It is acceptable if throughput is not an issue.

## Judge Definition

Return `true` if the question is directly about financial markets, economics, macro indicators, monetary policy, corporate financial performance, commodities/energy markets, FX, credit, rates, inflation, labor market macro releases, or prediction-market questions about financial/economic outcomes.

Return `false` for general business/career/marketing/workplace/product questions unless the question directly concerns financial performance, markets, macroeconomic outcomes, investment, prices, rates, currencies, credit, commodities, or firm earnings.

Examples of `true`:

```text
Will the Fed cut rates at the next FOMC?
Will CPI YoY exceed 3.0%?
Will Apple revenue beat consensus?
Will WTI crude close above $90?
Will USD/JPY be above 160?
Will the S&P 500 close higher by year-end?
Will unemployment exceed 4.5%?
```

Examples of `false`:

```text
Will a tennis player reach a final?
Will a court issue a gag order?
Will a company launch a new product name?
Will workers be advised to change career strategy?
Will a political leader visit a country?
```

Borderline rule:

If the question is only broadly "business" but not finance/economics/market relevant, mark `false`.

## Judge Prompt

Use a strict JSON-only response. Suggested system prompt:

```text
You are a dataset filtering judge. Decide whether a forecasting question belongs in a finance/economics forecasting dataset.

Return JSON only with exactly these keys:
{"is_finance_econ": boolean, "confidence": number}

Mark true only when the question is directly about financial markets, economics, macro indicators, monetary policy, corporate financial performance, commodities/energy markets, FX, credit, rates, inflation, labor market macro releases, or financial/economic prediction markets.

Mark false for general business, career, marketing, technology, politics, legal, sports, health, entertainment, or geopolitics unless the question directly asks about a financial/economic/market outcome.

Confidence must be between 0 and 1.
```

Suggested user payload:

```json
{
  "source_dataset": "...",
  "question": "...",
  "question_type": "...",
  "choices": null,
  "background": "...",
  "resolution_criteria": "...",
  "forecast_date": "...",
  "raw_category": "...",
  "source_url": "..."
}
```

Expected output:

```json
{"is_finance_econ": true, "confidence": 0.94}
```

## GPU Safety Check

Before starting vLLM, check that GPUs are free.

```bash
nvidia-smi
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Do not launch vLLM if any unrelated process is already using the A5000 GPUs.

Also check utilization/memory:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

## vLLM Launch

For Qwen3-4B on 4x A5000, the practical throughput setup is four single-GPU replicas instead of one 4-GPU tensor-parallel server.

Launch one vLLM server per GPU:

```bash
export MODEL_NAME="Qwen/Qwen3-4B"

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwen3-4b-gpu0

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 --port 8001 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwen3-4b-gpu1

CUDA_VISIBLE_DEVICES=2 vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 --port 8002 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwen3-4b-gpu2

CUDA_VISIBLE_DEVICES=3 vllm serve "$MODEL_NAME" \
  --host 0.0.0.0 --port 8003 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --served-model-name qwen3-4b-gpu3
```

Run these in separate tmux panes/sessions, or write a small launcher script. If one GPU is occupied, skip that GPU and only launch on the free ones.

Endpoints:

```text
http://localhost:8000/v1/chat/completions
http://localhost:8001/v1/chat/completions
http://localhost:8002/v1/chat/completions
http://localhost:8003/v1/chat/completions
```

## Append-Only Output

Do not rewrite `unified_tasks.jsonl` during inference. That is risky for crashes.

Use append-only sidecar results:

```text
data/judged/judge_results.jsonl
```

Each completed judge call should append one line:

```json
{
  "judge_uid": "...",
  "is_finance_econ": true,
  "confidence": 0.94,
  "model": "Qwen/Qwen3-4B",
  "endpoint": "http://localhost:8000",
  "created_at": "2026-07-11T..."
}
```

Crash-safety rules:

- Open the output in append mode.
- Write exactly one JSON object per completed row.
- Flush and `fsync` every small batch, for example every 10 or 50 rows.
- On restart, read existing `judge_results.jsonl`, collect completed `judge_uid`s, and skip them.
- Keep invalid/failed model outputs in a separate file:

```text
data/judged/judge_errors.jsonl
```

Do not modify the source JSONL while inference is running.

## Parallel Inference Plan

Recommended input:

```text
data/prejudge/judge_units.jsonl
```

Recommended output:

```text
data/judged/judge_results.jsonl
```

Process:

```text
1. Load completed judge_uid values from judge_results.jsonl if it exists.
2. Stream judge_units.jsonl line by line.
3. Skip completed judge_uid values.
4. Send requests concurrently to the four vLLM endpoints.
5. Validate that model output parses as JSON with only:
   - is_finance_econ
   - confidence
6. Append successful rows to judge_results.jsonl immediately.
7. Append failures to judge_errors.jsonl.
8. After all judge_units are done, join results back to unified_tasks.jsonl on judge_uid.
```

Suggested generation parameters:

```json
{
  "temperature": 0,
  "max_tokens": 64
}
```

## Final Join

After judging, create:

```text
data/judged/unified_tasks_judged.jsonl
data/judged/finance_econ_tasks.jsonl
```

Join key:

```text
judge_uid
```

For each row in `unified_tasks.jsonl`, attach:

```text
is_finance_econ
confidence
```

Then write filtered rows:

```text
is_finance_econ == true
```

Optionally require a confidence threshold:

```text
confidence >= 0.7
```

But keep the full judged file so thresholding can be changed later.

## Notes

- `judge_units.jsonl` is smaller and avoids duplicate ForecastBench judging.
- `unified_tasks.jsonl` is the final all-in-one task table.
- ForecastBench answers are mostly not in `unified_tasks.jsonl` yet; many rows are future/unresolved or need separate resolution joining.
- BTF-2, OpenForesight, and Daily Oracle already include answers.
- The filtering task is only domain filtering. Do not evaluate forecast correctness at this stage.
