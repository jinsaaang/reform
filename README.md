# ReForm

Forecasting over a current-instantiated exact topology, guided by hindsight
structure rather than hindsight answers.

A forecaster reading only current evidence cannot know which causal relations in
its domain have held before. A forecaster handed a past outcome learns the
answer, not the reasoning. ReForm takes the third path: it retrieves the
*structure* of how earlier events in the same family resolved — outcome-redacted
— re-tests each relation against current evidence, and forecasts from the
relations that survive.

Historical outcomes, option mappings, and probabilities never reach the
forecaster. The implementation does not pool probabilities, adjust a posterior,
anchor to a baseline prediction, or select a result by its score.

### Naming

The method is **ReForm** in writing and `hgf` in code — same method, two names.
The code identifier is left alone because two of those strings, the
structured-output schema name and the factor-memory view label, are sent to the
model; renaming them would change the prompt and make new runs incomparable with
everything already recorded. The rest are frozen benchmark contracts.

| In the paper | In this repository |
| --- | --- |
| ReForm | package `hgf/`, launcher `scripts/run_hgf.py` |
| ReForm result row | method key `procedural_topology_hgf_canonical` |
| ReForm schemas | every `hgf_*` schema string |

## Install

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENROUTER_API_KEY=YOUR_KEY
```

The only third-party runtime dependencies are `openai`, `pydantic`, and
`python-dotenv`. SQLite comes from the standard library.

## Run

The benchmark ships with the repository, so `--dataset-root .` runs against it
directly. Validate first; this makes no API call.

```bash
python3 scripts/validate_inputs.py --dataset-root .
```

```bash
python3 scripts/run_hgf.py \
  --dataset-root . \
  --model google/gemini-2.5-flash-lite \
  --provider google-ai-studio \
  --output-dir runs/gemini_seed0 --workers 20 --limit 100
```

```bash
python3 scripts/run_baselines.py \
  --dataset-root . \
  --model google/gemini-2.5-flash-lite \
  --output-dir runs/gemini_baselines --workers 20 --limit 100
```

Both launchers accept `--dry-run`, which prints the resolved command and exits
without creating an output directory or calling an API.

### Model settings

Each model carries its own provider route and reasoning setting.
`openai/gpt-5*` models omit `temperature` automatically.

| Model | Provider | Reasoning effort | `--disable-native-reasoning` |
| --- | --- | --- | --- |
| `google/gemini-2.5-flash-lite` | `google-ai-studio` | medium | |
| `openai/gpt-5-mini` | `openai` | medium | |
| `deepseek/deepseek-v3.2` | `atlas-cloud` | medium | |
| `qwen/qwen-plus-2025-07-28` | `alibaba` | medium | yes |
| `meta-llama/llama-4-maverick` | `deepinfra` | none | yes |

Two settings suppress reasoning and they act at different points.
`--reasoning-effort none` never builds the field; `--disable-native-reasoning`
strips it from the outgoing request just before the provider policy is attached.
Either one alone results in a request without a `reasoning` field, and passing
both is redundant rather than contradictory — Qwen is registered with effort
`medium` and the flag set, so its requests carry no reasoning field despite the
effort value.

```bash
python3 scripts/run_hgf.py \
  --dataset-root . \
  --model meta-llama/llama-4-maverick --provider deepinfra \
  --reasoning-effort none --disable-native-reasoning \
  --output-dir runs/llama_seed0 --workers 20 --limit 100
```

### Evidence selection

Evidence is selected at run time by a deterministic ranking in
`hgf/hgf/exemplar.py`, shared by ReForm and every baseline: keyword overlap with
the target, weighted toward title and entity matches, a bonus for official
sources, and a small recency term. No model call is involved, so the same
question yields the same evidence on every run and for every model.

### Resuming

A run writes one file per case, and both launchers refuse an existing
`--output-dir` so a fresh run cannot be mixed into an old one. Pass `--resume`
to continue: successful cases are read back from disk without an API call, and
only the missing and previously failed ones run.

```bash
python3 scripts/run_baselines.py --dataset-root . \
  --model google/gemini-2.5-flash-lite \
  --output-dir runs/gemini_baselines --workers 20 --limit 100 --resume
```

Expect some cases to fail on any pass — truncated structured output, transport
errors, or a payload the repair loop could not fix. Seeds are derived per
question and stage, so a case that fails on a contract violation tends to fail
the same way at the same `--run-seed`; give it a different seed for an
independent attempt.

## Method

ReForm forecasts in six stages.

1. Read the current cutoff-safe E1 evidence and write an evidence ledger
   covering target semantics, baseline, current drivers, counterevidence, and
   missing information.
2. Retrieve up to three outcome-redacted Blueprints from earlier resolved events
   with the same `family_id` and `target_metric`, only after current evidence
   has been read.
3. Route the Blueprint paths relevant to the current ledger. The original node
   order and edge relations are not rewritten.
4. Instantiate those paths with current evidence. Supported and contradicted
   relations can inform the forecast; unverified historical relations are not
   treated as current facts.
5. Produce a reasoning trace from the verified partial topology, current factors
   missing from history, counterevidence, uncertainty, and an answer-free worked
   reasoning check.
6. Map direction and magnitude to the public answer options in a separate
   boundary call that emits one probability distribution.

The Blueprint is neither a prior forecast nor a chain that must be followed. It
is a reusable expert structure that helps the model check and connect current
evidence.

### Controls

- Every article must precede the forecast cutoff.
- Historical events must resolve before the current forecast cutoff.
- Retrieval requires an exact event family and target metric match.
- Historical outcomes and probabilities must not enter ReForm.
- No method may read another method's prediction.
- No probability pooling, posterior adjustment, or result-conditioned retry.
- A successful run retains current evidence IDs, retrieved Blueprint IDs,
  reasoning, raw requests and responses, token usage, cost, and latency.

## Baselines

| `--methods` key | Name | Group | Description |
|---|---|---|---|
| `direct_forecast` | Direct Forecast | No-memory | Retrieval-augmented forecasting from current evidence. |
| `structured_reasoning` | Structured Reasoning | No-memory | Builds a forecast DAG from current evidence, without resolved-event memory. |
| `factor_memory` | Factor Memory | Resolved-event memory | Compact historical factors without the relations between them. |
| `principle_memory` | Principle Memory | Resolved-event memory | Answer-free forecasting principles distilled into text. |
| `case_memory` | Case Memory | Resolved-event memory | A complete resolved episode: question, evidence, outcome, reasoning. |
| `structure_memory` | Structure Memory | Resolved-event memory | An outcome-redacted past DAG used as-is, without instantiation against current evidence. |

All six share the model, target contract, output validator, question IDs, and
probability scorer with ReForm. `direct_forecast`, `structured_reasoning`,
`case_memory`, `principle_memory`, and `structure_memory` read E0;
`factor_memory` and ReForm read E1. E0 and E1 are frozen evidence databases, not
model predictions. Preserve this contract in comparisons, or report a uniform
evidence design as a separate experiment.

ReForm is not one of the six. It is a separate implementation under
`hgf/hgf_e2e_topology`, always run through `scripts/run_hgf.py`.

## Reasoning evaluation

`eval/reasoning_judge/` scores the reasoning behind each forecast, independently
of whether the forecast was right. An LLM judge grades every trace on five
dimensions with the ground truth and all metrics withheld; correctness is joined
back afterwards, which separates a method that argued its way to the answer from
one that guessed it. `RUBRIC.md` holds the rubric, `verdicts/` the scored
traces, `joined.json` the scores joined to accuracy and Brier.

```bash
python3 eval/reasoning_judge/build_packets.py <question-ids.json>
python3 eval/reasoning_judge/validate.py
python3 eval/reasoning_judge/aggregate.py
```

## Repository layout

| Path | Contents |
|---|---|
| `hgf/hgf` | Shared utilities, the six baselines, and benchmark validation. |
| `hgf/hgf_e2e_topology` | The ReForm method. |
| `hgf/hgf_e2e_topology_sidecar` | Records every API call alongside the run. |
| `hgf/hgf_e2e_topology_provider_pinned` | Pins the OpenRouter provider. |
| `data`, `artifacts` | The benchmark: questions, evidence, DAGs, Blueprints, exemplars. |
| `eval/reasoning_judge` | Reasoning-quality study. |
| `scripts/` | Launchers and benchmark validation. |

`hgf/` is the single `PYTHONPATH` root. Each script is a thin launcher: it puts
that root on `PYTHONPATH` and runs the corresponding module with `python -m`.
There is no build step; `pip install .` is not supported.

## Benchmark inputs

The layout below is the contract for pointing the launchers at a different
benchmark. The launchers assume historical DAGs have already been converted to
the canonical topology-preserving Blueprint format, and do not build DAGs.

```text
BENCHMARK_ROOT/
  data/
    questions/{test_questions.jsonl, memory_questions.jsonl, selection.json}
    evidence/{e0,e1}/<target_question_id>.sqlite
    memory_bank/manifest.json
  artifacts/
    hgf/blueprints/{manifest.json, cases/*.json}
    hgf/exemplars/{manifest.json, cases/*.json}
    baselines/factor_memory/{manifest.json, cases/*.json}
```

ReForm requires E1 plus the Blueprint and exemplar artifacts under
`artifacts/hgf`. The six baselines require E0, E1 for Factor Memory, the raw DAG
manifest, and the factor-memory artifacts.

**Questions.** JSON Lines accepted by the Pydantic `Question` model. Each target
and historical event needs a stable ID, question text, answer options,
resolution time, and metadata carrying at least these fields, either directly or
under `finance`, `finfactorbench`, or `benchmark`:

```json
{
  "family_id": "recurring_event_family",
  "target_metric": "metric_name",
  "category": "category_name",
  "forecast_date_options": ["2026-01-15"]
}
```

Historical events used as memory must resolve before the target forecast cutoff.
`selection.json` has one field: `{"question_ids": ["target_001"]}`.

**Evidence databases.** Each SQLite file needs an `articles` table with columns
`id, title, source, published_date, content, collected_for_question_id`. The
runtime independently rejects articles at or after the target cutoff.

**Blueprints.** Manifest schema `hgf_blueprint_manifest_v1`, cases
`hgf_blueprint_topology_v2`. The manifest must cover every row in
`memory_questions.jsonl`. A Blueprint preserves checkpoint IDs, directed edges,
conditional paths, lag, support, confidence, target bridge, and source
provenance, while excluding the historical answer from the forecast payload.

**Answer-free worked exemplars.** One per historical memory question, carrying a
historical ID in `retrieved_memory_question_id`, `source_question_id`, or
`memory_question_id`, plus a `worked_exemplar` object. The method strips the
historical estimate, option mapping, evidence article IDs, and forecast-time
evidence before building the reasoning check.

## Tests

```bash
python3 -m pytest
```

## License

MIT. See [LICENSE](LICENSE).
