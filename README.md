# Procedural Topology HGF

This branch contains the single canonical implementation used for the final
five-model experiment. It is intentionally separated from the exploratory HGF
variants in the research repository.

Procedural Topology HGF first organizes current cutoff-safe evidence. It then
retrieves outcome-redacted Blueprints from resolved events in the same event
family, instantiates their useful subgraphs with current evidence, and uses the
verified partial topology as a reasoning scaffold. The forecaster may add
current factors that are absent from history and must consider counterevidence
before mapping the resulting direction and magnitude assessment to a single
probability forecast.

Historical outcomes, option mappings, and probabilities are never passed to
the forecaster. The implementation does not pool probabilities, adjust a
posterior, anchor to a baseline prediction, or choose a result by its score.

## What is included

- `hgf/method_src` contains the final HGF method.
- `hgf/hgf_historical_base_src` contains frozen shared utilities and the
  six baseline implementations used by the controlled experiment.
- `hgf/input_adapter_src` optionally freezes model-specific evidence and
  historical retrieval.
- `hgf/execution_src` records raw calls and pins an OpenRouter provider.
- `scripts/run_hgf.py` runs HGF on a compatible benchmark.
- `scripts/run_baselines.py` runs the six baselines.
- `scripts/validate_inputs.py` checks the required benchmark artifacts without
  making an API call.
- `results` contains compact summaries from the registered five-model run.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENROUTER_API_KEY=YOUR_KEY
```

## Validate a benchmark

```bash
python3 scripts/validate_inputs.py --dataset-root /path/to/benchmark
```

## Run HGF

```bash
python3 scripts/run_hgf.py \
  --dataset-root /path/to/benchmark \
  --model google/gemini-2.5-flash-lite \
  --provider google-ai-studio \
  --output-dir /path/to/runs/gemini_seed0 \
  --workers 20 \
  --limit 100
```

Add both `--evidence-selection-manifest` and `--retrieval-manifest` to replay
frozen model-specific inputs. If neither is supplied, the same canonical method
performs deterministic evidence reranking and exact-family Blueprint retrieval
from the provided benchmark artifacts.

See [docs/METHOD.md](docs/METHOD.md),
[docs/GENERAL_BENCHMARK.md](docs/GENERAL_BENCHMARK.md), and
[docs/DEPENDENCY_AUDIT.md](docs/DEPENDENCY_AUDIT.md) before a paper run.

## Tests

```bash
python3 -m pytest
python3 scripts/audit_dependencies.py
```
