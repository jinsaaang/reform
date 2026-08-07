# Procedural Topology HGF

Hindsight-Guided Forecasting over a current-instantiated exact topology.

The method first organizes current cutoff-safe evidence. It then retrieves
outcome-redacted Blueprints from resolved events in the same event family,
instantiates their useful subgraphs with current evidence, and uses the
verified partial topology as a reasoning scaffold. The forecaster may add
current factors that are absent from history and must consider counterevidence
before mapping the resulting direction and magnitude assessment to a single
probability forecast.

Historical outcomes, option mappings, and probabilities are never passed to the
forecaster. The implementation does not pool probabilities, adjust a posterior,
anchor to a baseline prediction, or choose a result by its score.

## Layout

| Path | Contents |
|---|---|
| `hgf/hgf` | Shared utilities, the six baselines, and benchmark validation. |
| `hgf/hgf_e2e_topology` | The HGF method. |
| `hgf/hgf_e2e_topology_sidecar` | Records every API call alongside the run. |
| `hgf/hgf_e2e_topology_provider_pinned` | Pins the OpenRouter provider. |
| `data`, `artifacts` | The benchmark: questions, evidence, DAGs, Blueprints, exemplars. |
| `scripts/run_hgf.py` | Run HGF on a compatible benchmark. |
| `scripts/run_baselines.py` | Run the six baselines. |
| `scripts/validate_inputs.py` | Check benchmark artifacts without an API call. |

`hgf/` is the single `PYTHONPATH` root and holds the four importable packages.
Each script is a thin launcher: it puts that root on `PYTHONPATH` and runs the
corresponding module with `python -m`. There is no build step and nothing to
install; `pip install .` is not supported.

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

## Benchmark inputs

The benchmark ships with the repository, so `--dataset-root .` runs against it
directly. The layout below is also the contract for pointing the launchers at a
different benchmark; the launchers assume historical DAGs have already been
converted to the canonical topology-preserving Blueprint format, and do not
build DAGs.

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
      blueprints/
        manifest.json
        cases/*.json
      exemplars/
        manifest.json
        cases/*.json
    baselines/
      factor_memory/
        manifest.json
        cases/*.json
```

HGF requires E1 plus the Blueprint and exemplar artifacts under
`artifacts/hgf`. The six baselines require E0, E1 for Factor Memory, the raw
DAG manifest, and the factor-memory artifacts; they do not read
`artifacts/hgf`.

### Questions

Both question files are JSON Lines accepted by the Pydantic `Question` model.
Each target and historical event needs a stable ID, question text, answer
options, resolution time, and metadata. The metadata must carry at least these
fields, either directly or under `finance`, `finfactorbench`, or `benchmark`:

```json
{
  "family_id": "recurring_event_family",
  "target_metric": "metric_name",
  "category": "category_name",
  "forecast_date_options": ["2026-01-15"]
}
```

Historical events used as memory must resolve before the target forecast
cutoff. `selection.json` has one field:

```json
{"question_ids": ["target_001", "target_002"]}
```

### Evidence databases

Each SQLite file needs an `articles` table with these columns:

```text
id, title, source, published_date, content, collected_for_question_id
```

The runtime independently rejects articles at or after the target cutoff.

### Blueprints

The Blueprint manifest schema is `hgf_blueprint_manifest_v1` and every case
uses `hgf_blueprint_topology_v2`. The manifest must cover every row in
`memory_questions.jsonl`, and its optional canonical hashes must match. A
Blueprint preserves checkpoint IDs, directed edges, conditional paths, lag,
support, confidence, target bridge, and source provenance, while excluding the
historical answer from the forecast payload.

### Answer-free worked exemplars

The exemplar directory needs one worked exemplar per historical memory
question. The loader accepts JSON files carrying a historical ID in
`retrieved_memory_question_id`, `source_question_id`, or `memory_question_id`,
plus a `worked_exemplar` object. The method strips the historical estimate,
option mapping, evidence article IDs, and forecast-time evidence before it
builds the reasoning check.

## Run

Validate the benchmark first; this makes no API call.

```bash
python3 scripts/validate_inputs.py --dataset-root .
```

Run HGF:

```bash
python3 scripts/run_hgf.py \
  --dataset-root . \
  --model google/gemini-2.5-flash-lite \
  --provider google-ai-studio \
  --output-dir runs/gemini_seed0 \
  --workers 20 \
  --limit 100
```

Run the baselines:

```bash
python3 scripts/run_baselines.py \
  --dataset-root . \
  --model google/gemini-2.5-flash-lite \
  --output-dir runs/gemini_baselines \
  --workers 20 \
  --limit 100
```

Both launchers accept `--dry-run`, which prints the resolved command and exits
without creating an output directory or calling an API. The method reranks
current evidence and retrieves compatible Blueprints at runtime from the
artifacts you supply.

### Resuming

A run writes one file per case, and both launchers refuse an existing
`--output-dir` so that a fresh run cannot be mixed into an old one. Pass
`--resume` to continue instead: cases already recorded as successful are read
back from their case file without an API call, and only the missing and
previously failed ones run.

```bash
python3 scripts/run_baselines.py --dataset-root . \
  --model google/gemini-2.5-flash-lite \
  --output-dir runs/gemini_baselines --workers 20 --limit 100 --resume
```

Expect some cases to fail on any given pass. The models truncate structured
output and providers return transport errors, so a run that reports a handful
of failures is normal; re-running with `--resume` picks up exactly those.

## Method

HGF forecasts in six stages.

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
5. Produce a reasoning trace from the verified partial topology, current
   factors missing from history, counterevidence, uncertainty, and an
   answer-free worked reasoning check. The exemplar supplies reasoning form,
   not a historical answer or probability.
6. Map the model's direction and magnitude judgment to the public answer
   options in a separate boundary call that emits one probability
   distribution.

The Blueprint is therefore neither a prior forecast nor a complete chain that
must be followed. It is a reusable expert structure that helps the model check
and connect current evidence.

### Controls

- Every article must precede the forecast cutoff.
- Historical events must resolve before the current forecast cutoff.
- Retrieval requires an exact event family and target metric match.
- Historical outcomes and probabilities must not enter HGF.
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
probability scorer with HGF. `direct_forecast`, `structured_reasoning`,
`case_memory`, `principle_memory`, and `structure_memory` use E0;
`factor_memory` and HGF use E1. E0 and E1 are frozen evidence databases, not
model predictions. Preserve this contract in comparisons, or report a new
uniform evidence design as a separate experiment.

Structure Memory and Case Memory inputs need the same semantic leakage audit
before their numbers are reportable. The launcher does not silently rewrite
user-provided DAG or case artifacts.

HGF is not one of the six. It is a separate implementation under
`hgf/method/hgf_e2e_topology`, always run through `scripts/run_hgf.py`.

## Tests

```bash
python3 -m pytest
```

## License

MIT. See [LICENSE](LICENSE).
