# ReFoRM

**Re**curring-Event **Fo**recasting with **R**easoning **M**emory.

A forecaster reading only current evidence cannot know which causal relations in
its domain have held before. A forecaster handed a past outcome learns the
answer, not the reasoning. ReFoRM takes the third path: it retrieves the
*structure* of how earlier events in the same family resolved — outcome-redacted
— re-tests each relation against current evidence, and forecasts from the
relations that survive.

Historical outcomes, option mappings, and probabilities never reach the
forecaster. The implementation does not pool probabilities, adjust a posterior,
anchor to a baseline prediction, or select a result by its score.

### Naming

The method is **ReFoRM** in writing and `hgf` in code — same method, two names.
The code identifier is left alone because two of those strings, the
structured-output schema name and the factor-memory view label, are sent to the
model; renaming them would change the prompt and make new runs incomparable with
everything already recorded. The rest are frozen benchmark contracts.

| In the paper | In this repository |
| --- | --- |
| ReFoRM | package `hgf/`, launcher `scripts/run_hgf.py` |
| ReFoRM result row | method key `procedural_topology_hgf_canonical` |
| ReFoRM schemas | every `hgf_*` schema string |

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
`hgf/hgf/exemplar.py`, shared by ReFoRM and every baseline: keyword overlap with
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

ReFoRM forecasts in six stages.

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
- Historical outcomes and probabilities must not enter ReFoRM.
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
probability scorer with ReFoRM. `direct_forecast`, `structured_reasoning`,
`case_memory`, `principle_memory`, and `structure_memory` read E0;
`factor_memory` and ReFoRM read E1. E0 and E1 are frozen evidence databases, not
model predictions. Preserve this contract in comparisons, or report a uniform
evidence design as a separate experiment.

ReFoRM is not one of the six. It is a separate implementation under
`hgf/hgf_e2e_topology`, always run through `scripts/run_hgf.py`.

## Reasoning evaluation

Brier scores the distribution a method emits, not how it got there. On a
three-way question, a method that invents a threshold-crossing quantity is right
about a third of the time, and on those questions its Brier is
indistinguishable from a method that reasoned its way to the same interval.
`eval/reasoning_judge/` scores the reasoning itself and then crosses it against
the outcome.

**Protocol.** One judge per question, each an isolated agent, grading all seven
traces of that question. The judge sees the question, its options and cutoff,
the evidence bank each trace actually read, and each trace's `reasoning`,
`forecast` and `probabilities`. It does **not** see the ground truth or any
metric — a judge that knows the answer reads correct traces charitably and hunts
for flaws in wrong ones, which collapses the one cell that answers the question.
Correctness is joined back afterwards by `aggregate.py`.

Because the benchmark does not give every method the same evidence, each packet
carries both banks, every trace names its own, and grading is against that
trace's own bank plus its retrieved memory — never against a bank it was not
given, and never rewarding bank size.

### Rubric

Five dimensions, integers 1–5, anchors defined at 1, 3 and 5. Full text in
`eval/reasoning_judge/RUBRIC.md`.

| Dimension | The check | 1 | 5 |
| --- | --- | --- | --- |
| `evidence_grounding` | Do claims trace to the cited evidence? | Invents or misquotes figures | Every figure traceable, citations match the source |
| `logical_validity` | Do premises reach the conclusion? | Non-sequitur or self-contradiction | Each step follows; competing considerations resolved |
| `prediction_alignment` | Does the reasoning entail the option chosen? | Reasoning points elsewhere | Uniquely selects it, and excludes the neighbours |
| `probability_justification` | Does the mass match declared uncertainty? | Calls a claim unsupported, then bets on it | Support level, admitted gaps and mass all agree |
| `mechanism_specificity` | Real mechanism or filler? | Fits any question on any indicator | Traces named drivers through to the target |

`prediction_alignment` is the central test: the judge reads the reasoning,
decides what it implies, and only then compares against the emitted option.

Six binary flags catch categorical defects an average would hide —
`unsupported_magnitude_leap`, `hallucinated_number`, `internal_contradiction`,
`boilerplate_only`, `post_hoc_option_fit`, and `admits_own_gap`. The last is a
**positive** signal: without it the rubric cannot tell honesty from hedging,
since a trace that says it cannot establish the magnitude and keeps its mass
flat is doing the right thing. Every score requires a verbatim quote, so any
judgement can be audited by hand.

### What the scores look like in practice

One question, one evidence bank, seven traces. `v3_googl_revenue_growth_acceleration_2025_12_31`
asks whether Alphabet's Q4 2025 YoY revenue growth clears 15.95 percent, and the
bank contains no Q4 forecast — only a Q3 figure.

| Method | Score | Ground | Valid | Align | Prob | Mech | Decisive quote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **ReFoRM** | **4.2** | 5 | 4 | 4 | 3 | 5 | "Derived from Q3 2025 performance (16% YoY growth) and identified positive growth drivers (AI, Cloud, Advertising), tempered by the absence of specific Q4 2025 forecasts" |
| Factor Memory | 3.6 | 5 | 3 | 4 | 2 | 4 | "No evidence-supported magnitude distinguishes this option; the contract-centered abstention is intentionally broad." |
| Direct Forecast | 2.2 | 3 | 2 | 2 | 2 | 2 | "Broad range reflecting positive trends and AI-driven growth, but lacking specific Q4 2025 forecasts." |
| Structure Memory | 1.0 | 1 | 1 | 1 | 1 | 1 | "The broad estimate [-20, 20] has significant overlap with the 'no' interval, including its central estimate (0), making it the most supported option by the estimate." |

The spread is not about who had better evidence — all four read the same bank.
ReFoRM anchors to the one quantity that exists and names the gap the quantity
cannot close. Structure Memory invents a ±20 percentage-point band wide enough
to contain any answer, then treats the band's midpoint as support for the option
it picked. Direct Forecast notices the same missing forecast ReFoRM does but
draws nothing from it.

The failure mode at the bottom of the distribution is consistent. Most
correct-but-unreasoned traces are a validation failure falling into a fallback
that cites nothing and declares an even split, after which the forecast stage
supplies a central estimate and commits most of the mass to it:

> "No evidence-supported point estimate; retain a broad neutral range."
> — then assigns 0.764 to one option.

> "Central estimate (0.25) falls within the 'yes' interval."
> — after the reasoning block states no boundary crossing is supported.

Both were scored 1.0–1.4 and both were correct. Counting them as successes is
exactly what this evaluation is designed to prevent.

### Running it

```bash
python3 eval/reasoning_judge/build_packets.py <question-ids.json>
python3 eval/reasoning_judge/validate.py
python3 eval/reasoning_judge/aggregate.py
```

`build_packets.py` writes one ground-truth-free packet per question and refuses
to emit one whose trace cites evidence outside its own bank. Judging is the
manual step: each packet goes to one judge with `RUBRIC.md`, and the verdict
lands in `verdicts/qNN.json`. `validate.py` rejects malformed, truncated or
incomplete verdicts — a retried judge overwrites its own file, and a write cut
off midway still parses far enough to fool a shallow check. `aggregate.py` joins
the verdicts against the withheld metrics; `joined.json` holds all 700 scored
traces with their quotes.

## Repository layout

| Path | Contents |
|---|---|
| `hgf/hgf` | Shared utilities, the six baselines, and benchmark validation. |
| `hgf/hgf_e2e_topology` | The ReFoRM method. |
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

ReFoRM requires E1 plus the Blueprint and exemplar artifacts under
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
