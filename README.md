# ReForm

Forecasting over a current-instantiated exact topology, guided by hindsight
structure rather than hindsight answers.

A forecaster reading only current evidence has no way to know which causal
relations in its domain have held before. A forecaster handed a past outcome
learns the answer, not the reasoning. ReForm takes the third path: it retrieves
the *structure* of how earlier events in the same family resolved —
outcome-redacted — re-tests each relation against current evidence, and
forecasts from the relations that survive.

Concretely, the method organizes current cutoff-safe evidence first, then
retrieves outcome-redacted Blueprints from resolved events in the same event
family, instantiates their useful subgraphs against that evidence, and uses the
verified partial topology as a reasoning scaffold. The forecaster may add
current factors absent from history and must weigh counterevidence before a
separate boundary call maps direction and magnitude to one probability
distribution.

Historical outcomes, option mappings, and probabilities never reach the
forecaster. The implementation does not pool probabilities, adjust a posterior,
anchor to a baseline prediction, or select a result by its score.

### Naming

The method is **ReForm** in writing and `hgf` in code. Same method, two names.

| In the paper | In this repository |
| --- | --- |
| ReForm | package `hgf/`, launcher `scripts/run_hgf.py` |
| ReForm result row | method key `procedural_topology_hgf_canonical` |
| ReForm case file | `procedural_topology_hgf_canonical.json` |
| ReForm schemas | every `hgf_*` schema string |

The code identifier stays as it is. Two of those strings — the
structured-output schema name and the factor-memory view label — are sent to the
model, so renaming them would change the prompt and make new runs incomparable
with every run already recorded. The rest are frozen benchmark contracts that
would require rewriting the dataset. Nothing downstream needs the code to say
ReForm; read the table above and use the paper's name in write-ups.

---

## Results

Gemini 2.5 Flash Lite, 100 questions, three seeds, with evidence selected by the
deterministic ranking described under [Evidence selection](#evidence-selection).

| Method | Accuracy | Brier | NLL |
| --- | ---: | ---: | ---: |
| Direct Forecast | 0.523±0.021 | 0.2349±0.0076 | 1.0405±0.0342 |
| Structured Reasoning | 0.387±0.031 | 0.2852±0.0148 | 1.2099±0.0555 |
| Factor Memory | 0.523±0.012 | 0.2351±0.0050 | 1.0007±0.0182 |
| Principle Memory | 0.460±0.035 | 0.2595±0.0152 | 1.1164±0.0670 |
| Case Memory | 0.453±0.042 | 0.2617±0.0076 | 1.1605±0.0494 |
| Structure Memory | 0.483±0.012 | 0.2502±0.0050 | 1.0937±0.0026 |
| **ReForm** | **0.530±0.000** | **0.2123±0.0000** | **0.9141±0.0000** |

ReForm reduces Brier by 9.6 percent against the strongest baseline. Its zero
standard deviation is a property of the configuration, not a claim of
stability: with `temperature=0` and a pinned provider the decode is greedy, so
the run seed cannot change the output. The baselines vary because they are not
provider-pinned, which makes their spread routing noise rather than sampling
noise.

### Does the accuracy come from the reasoning?

Brier scores the distribution a method emits, not how it got there. On a
three-way question, a method that invents a threshold-crossing quantity is right
about a third of the time, and on those questions its Brier is
indistinguishable from a method that reasoned its way to the same interval.

To separate the two, an LLM judge (Claude Opus 5, one judge per question) scored
all 700 seed-0 traces on five dimensions with **the ground truth and every
metric withheld**, and correctness was joined back afterwards. Withholding is
load-bearing: a judge that knows the answer reads correct traces charitably and
hunts for flaws in wrong ones, which collapses the cell that answers the
question. Full protocol in [Reasoning evaluation](#reasoning-evaluation).

| Method | Reasoning score | Correct-but-unreasoned |
| --- | ---: | ---: |
| **ReForm** | **3.73** | **0% (0/53)** |
| Factor Memory | 3.17 | 19% (10/53) |
| Principle Memory | 2.97 | 30% (13/44) |
| Case Memory | 2.96 | 24% (10/42) |
| Direct Forecast | 2.93 | 24% (13/54) |
| Structured Reasoning | 2.86 | 29% (12/42) |
| Structure Memory | 2.86 | 37% (18/49) |

Crossing the blind score against correctness separates the two ways of being
right. Of the 53 questions ReForm answered correctly, **none** came from a trace
the judge scored at or below 2.5; every baseline has 10 to 18 such cases. ReForm
also holds the largest wrong-but-reasoned count, 20 against 0–4 for the
baselines — the expected shape for a method whose errors come from the evidence
rather than from the argument.

| Method | Right, reasoned | Right, unreasoned | Wrong, reasoned | Wrong, unreasoned |
| --- | ---: | ---: | ---: | ---: |
| **ReForm** | **19** | **0** | **20** | 1 |
| Factor Memory | 11 | 10 | 4 | 12 |
| Principle Memory | 6 | 13 | 1 | 11 |
| Structure Memory | 5 | 18 | 0 | 18 |
| Direct Forecast | 3 | 13 | 1 | 11 |
| Structured Reasoning | 1 | 12 | 1 | 16 |
| Case Memory | 1 | 10 | 0 | 15 |

ReForm leads all five rubric dimensions and takes the top score, alone or tied, in
59 of 100 questions against 23 for the next method. Paired by question, its
narrowest margin — 0.57 over Factor Memory — is 6.5 standard errors.

---

## Method

ReForm forecasts in six stages.

1. Read the current cutoff-safe E1 evidence and write an evidence ledger
   covering target semantics, baseline, current drivers, counterevidence, and
   missing information.
2. Retrieve up to three outcome-redacted Blueprints from earlier resolved
   events with the same `family_id` and `target_metric`. Retrieval happens only
   after current evidence has been read.
3. Route the Blueprint paths relevant to the current ledger. The original node
   order and edge relations are not rewritten.
4. Instantiate those paths with current evidence. Supported and contradicted
   relations can inform the forecast; unverified historical relations are not
   treated as current facts.
5. Produce a reasoning trace from the verified partial topology, current factors
   missing from history, counterevidence, uncertainty, and an answer-free worked
   reasoning check. The exemplar supplies reasoning form, not a historical
   answer or probability.
6. Map the model's direction and magnitude judgment to the public answer options
   in a separate boundary call that emits one probability distribution.

The Blueprint is therefore neither a prior forecast nor a chain that must be
followed. It is a reusable expert structure that helps the model check and
connect current evidence.

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

The six baselines split by whether they read resolved-event memory at all.

| `--methods` key | Name | Group | Description |
|---|---|---|---|
| `direct_forecast` | Direct Forecast | No-memory | Standard retrieval-augmented forecasting from current evidence. |
| `structured_reasoning` | Structured Reasoning | No-memory | Builds a forecast DAG from current evidence, still without resolved-event memory. |
| `factor_memory` | Factor Memory | Resolved-event memory | Retrieves compact historical factors without the relations between them. |
| `principle_memory` | Principle Memory | Resolved-event memory | Retrieves answer-free forecasting principles distilled into text. |
| `case_memory` | Case Memory | Resolved-event memory | Retrieves a complete resolved episode: question, evidence, outcome, reasoning. |
| `structure_memory` | Structure Memory | Resolved-event memory | Retrieves an outcome-redacted past DAG and uses it as-is, without instantiating it against current evidence. |

All six share the model, target contract, output validator, question IDs, and
probability scorer with ReForm. `direct_forecast`, `structured_reasoning`,
`case_memory`, `principle_memory`, and `structure_memory` read E0;
`factor_memory` and ReForm read E1. E0 and E1 are frozen evidence databases, not
model predictions. Preserve this contract in comparisons, or report a uniform
evidence design as a separate experiment.

Structure Memory and Case Memory inputs need the same semantic leakage audit
before their numbers are reportable. The launcher does not silently rewrite
user-provided DAG or case artifacts.

ReForm is not one of the six. It is a separate implementation under
`hgf/hgf_e2e_topology`, always run through `scripts/run_hgf.py`.

---

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

Each registered model carries its own provider route and reasoning setting.
Models with no native reasoning control take `--reasoning-effort none`, which
omits the reasoning field from the request rather than sending a low value;
`--disable-native-reasoning` records that choice in the run manifest.
`openai/gpt-5*` models omit `temperature` automatically.

| Model | ReForm provider | Reasoning effort | Native reasoning |
| --- | --- | --- | --- |
| `google/gemini-2.5-flash-lite` | `google-ai-studio` | medium | on |
| `openai/gpt-5-mini` | `openai` | medium | on |
| `deepseek/deepseek-v3.2` | `atlas-cloud` | medium | on |
| `meta-llama/llama-4-maverick` | `deepinfra` | none | off |
| `minimax/minimax-m2.5` | `auto-latency` | medium | on |

```bash
python3 scripts/run_hgf.py \
  --dataset-root . \
  --model meta-llama/llama-4-maverick --provider deepinfra \
  --reasoning-effort none --disable-native-reasoning \
  --output-dir runs/llama_seed0 --workers 20 --limit 100
```

### Evidence selection

Evidence is selected at run time by a deterministic ranking in
`hgf/hgf/exemplar.py`, shared by ReForm and every baseline. Each article in the
cutoff-safe pool is scored by keyword overlap with the target, weighted toward
title and entity matches, with a bonus for official sources and a small recency
term; the top-scoring articles are kept.

No model call is involved, so the same question yields the same evidence on
every run and for every model. Differences between methods are therefore
differences in reasoning, not in what each method was allowed to read.

### Resuming

A run writes one file per case, and both launchers refuse an existing
`--output-dir` so a fresh run cannot be mixed into an old one. Pass `--resume`
to continue instead: cases already recorded as successful are read back from
their case file without an API call, and only the missing and previously failed
ones run.

```bash
python3 scripts/run_baselines.py --dataset-root . \
  --model google/gemini-2.5-flash-lite \
  --output-dir runs/gemini_baselines --workers 20 --limit 100 --resume
```

Expect some cases to fail on any given pass. Models truncate structured output,
providers return transport errors, and the forecast-contract validator rejects a
payload the repair loop could not fix. A run reporting a handful of failures is
normal; re-running with `--resume` picks up exactly those.

Seeds are derived per question and stage, so a case that fails on a contract
violation tends to fail the same way on every retry at the same `--run-seed`.
Give that case a different seed for an independent attempt:

```bash
python3 scripts/run_hgf.py --dataset-root . \
  --model google/gemini-2.5-flash-lite --provider google-ai-studio \
  --output-dir runs/gemini_seed0 --resume \
  --question-ids THE_FAILING_ID --run-seed 1
```

---

## Reasoning evaluation

`eval/reasoning_judge/` holds the study behind the second results table.

**Setup.** One judge per question, each an isolated agent with no memory of
other questions, grading all seven traces of that question. The judge sees the
question, its options and cutoff, the full evidence bank each trace actually
read, and each trace's `reasoning`, `forecast`, and `probabilities`. It does not
see the ground truth or any metric; correctness is joined afterwards by a script
the judge never runs. Packets are checked structurally for leakage before
dispatch.

Because the benchmark does not give every method the same evidence, each packet
carries both banks, every trace names its own, and the rubric grades each trace
against its own bank plus its retrieved memory — never against a bank it was not
given, and never rewarding bank size. Method names are visible: ReForm emits a
distinctive schema, so blinding was not achievable without discarding the
content under test. The rubric compensates by scoring checkable properties
rather than impressions and by instructing judges that verbosity, jargon, and
structural formatting earn nothing.

**Rubric.** Five dimensions scored 1–5 with anchors at 1, 3, and 5:
evidence grounding, logical validity, prediction alignment, probability
justification, mechanism specificity. Prediction alignment is the central test —
the judge reads the reasoning, decides what it implies, and only then compares
against the emitted option. Six binary flags record categorical defects an
average would hide, including `admits_own_gap` as a *positive* signal, without
which the rubric cannot distinguish honesty from hedging. Every score requires a
verbatim quote. Full text in `eval/reasoning_judge/RUBRIC.md`.

**Reading the numbers.** Reasoning scores come from seed 0 while Acc/Brier/NLL
average three seeds, so the columns should not be correlated within a row.
Within-question scores are not independent — one judge scored all seven traces —
so the paired-by-question comparison is the appropriate test and pooled standard
errors would be optimistic. Two questions were graded twice by accident and
low-scoring traces moved by up to 11 points on the 25-point scale, so no claim
should rest on a single trace's score. These scores are a strong model's
judgement of each trace, not ground truth about reasoning quality; every score
carries a quote so a human can audit it.

```bash
python3 eval/reasoning_judge/build_packets.py <question-ids.json>
python3 eval/reasoning_judge/validate.py
python3 eval/reasoning_judge/aggregate.py
```

`build_packets.py` writes one ground-truth-free packet per question. Judging is
the manual step: each packet goes to one judge with `RUBRIC.md`, and the verdict
lands in `verdicts/qNN.json`. `validate.py` rejects malformed, truncated, or
incomplete verdicts before they reach the aggregate. All 700 scored traces with
their decisive quotes are in `joined.json`.

---

## Repository layout

| Path | Contents |
|---|---|
| `hgf/hgf` | Shared utilities, the six baselines, and benchmark validation. |
| `hgf/hgf_e2e_topology` | The ReForm method. |
| `hgf/hgf_e2e_topology_sidecar` | Records every API call alongside the run. |
| `hgf/hgf_e2e_topology_provider_pinned` | Pins the OpenRouter provider. |
| `data`, `artifacts` | The benchmark: questions, evidence, DAGs, Blueprints, exemplars. |
| `eval/reasoning_judge` | Reasoning-quality study: rubric, verdicts, aggregation. |
| `scripts/run_hgf.py` | Run ReForm on a compatible benchmark. |
| `scripts/run_baselines.py` | Run the six baselines. |
| `scripts/validate_inputs.py` | Check benchmark artifacts without an API call. |

`hgf/` is the single `PYTHONPATH` root and holds the importable packages. Each
script is a thin launcher: it puts that root on `PYTHONPATH` and runs the
corresponding module with `python -m`. There is no build step and nothing to
install; `pip install .` is not supported.

## Benchmark inputs

The layout below is the contract for pointing the launchers at a different
benchmark. The launchers assume historical DAGs have already been converted to
the canonical topology-preserving Blueprint format, and do not build DAGs.

```text
BENCHMARK_ROOT/
  data/
    questions/
      test_questions.jsonl
      memory_questions.jsonl
      selection.json
    evidence/
      e0/<target_question_id>.sqlite
      e1/<target_question_id>.sqlite
    memory_bank/
      manifest.json
  artifacts/
    hgf/
      blueprints/{manifest.json, cases/*.json}
      exemplars/{manifest.json, cases/*.json}
    baselines/
      factor_memory/{manifest.json, cases/*.json}
```

ReForm requires E1 plus the Blueprint and exemplar artifacts under `artifacts/hgf`.
The six baselines require E0, E1 for Factor Memory, the raw DAG manifest, and
the factor-memory artifacts; they do not read `artifacts/hgf`.

**Questions.** Both question files are JSON Lines accepted by the Pydantic
`Question` model. Each target and historical event needs a stable ID, question
text, answer options, resolution time, and metadata carrying at least these
fields, either directly or under `finance`, `finfactorbench`, or `benchmark`:

```json
{
  "family_id": "recurring_event_family",
  "target_metric": "metric_name",
  "category": "category_name",
  "forecast_date_options": ["2026-01-15"]
}
```

Historical events used as memory must resolve before the target forecast cutoff.
`selection.json` has one field: `{"question_ids": ["target_001", "target_002"]}`.

**Evidence databases.** Each SQLite file needs an `articles` table with columns
`id, title, source, published_date, content, collected_for_question_id`. The
runtime independently rejects articles at or after the target cutoff.

**Blueprints.** Manifest schema `hgf_blueprint_manifest_v1`, cases
`hgf_blueprint_topology_v2`. The manifest must cover every row in
`memory_questions.jsonl`, and its optional canonical hashes must match. A
Blueprint preserves checkpoint IDs, directed edges, conditional paths, lag,
support, confidence, target bridge, and source provenance, while excluding the
historical answer from the forecast payload.

**Answer-free worked exemplars.** One per historical memory question. The loader
accepts JSON files carrying a historical ID in `retrieved_memory_question_id`,
`source_question_id`, or `memory_question_id`, plus a `worked_exemplar` object.
The method strips the historical estimate, option mapping, evidence article IDs,
and forecast-time evidence before building the reasoning check.

## Tests

```bash
python3 -m pytest
```

## License

MIT. See [LICENSE](LICENSE).
