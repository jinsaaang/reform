# Hindsight-Guided Forecasting

This repository contains the public HGF forecasting pipeline, its six
baselines, and the complete frozen input bundle for the 100-question
evaluation.

The HGF path uses:

- one fixed retrieved memory question per test question;
- one fixed worked exemplar derived from its audited DAG;
- a 200-entry DAG memory bank;
- cutoff-safe E1 evidence;
- cached semantic lessons;
- current-question reasoning followed by boundary-aware probability mapping.

Fixed exemplars are loaded from disk and are never regenerated during a
forecasting run.

## Repository structure

```text
study_kakao/
|-- data/
|   |-- questions/
|   |   |-- memory_questions.jsonl
|   |   |-- test_questions.jsonl
|   |   `-- selection.json
|   |-- memory_bank/
|   |   `-- manifest.json
|   |-- dags/
|   |   `-- <memory_question_id>/
|   `-- evidence/
|       |-- e0/
|       `-- e1/
|-- artifacts/
|   |-- exemplars/
|   `-- semantic_lessons/
|-- configs/
|-- experiments/
|-- src/hgf/
`-- tests/
```

E0 is the question-only evidence bank used by the appropriate baselines. E1 is
the factor-guided evidence bank used by Factor Memory and HGF.

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Run commands from the repository root. To run them elsewhere, set `HGF_ROOT`
to the absolute repository path.

## Validate the bundle

The following commands do not call an external model:

```bash
hgf-manifest
hgf-verify
hgf-preflight
python -m pytest
```

`hgf-preflight` validates all seven registered methods, all 100 E0 databases,
all 100 E1 databases, the 200-entry memory bank, the 100 fixed exemplars, and
the semantic cache.

## Run HGF

Set `OPENROUTER_API_KEY` in the environment or a local `.env` file:

```bash
hgf-replay
```

The default run evaluates 100 questions with four workers. Common overrides:

```bash
hgf-replay --model google/gemini-2.5-flash-lite --limit 10 --workers 4
```

Results are written to `runs/hgf/`.

## Run the main table

```bash
hgf-main-table
```

The main-table runner supports:

1. Search-only
2. Factor Memory
3. Case Memory
4. Text Memory
5. Direct DAG
6. Prospective DAG
7. HGF

Use `--methods` to run a subset:

```bash
hgf-main-table --methods search_only direct_dag hgf
```

Results are written to `runs/main_table/`.

For repeated live-model experiments, use a separate output directory and
run seed for every repetition. The optional generation controls are recorded
in both `protocol.json` and `results.json`:

```bash
hgf-main-table \
  --model openai/gpt-5-mini \
  --workers 20 \
  --reasoning-effort medium \
  --max-output-tokens 8000 \
  --run-seed 1 \
  --output-dir runs/gpt_5_mini/repeat_1
```

Omitting these options preserves the canonical stage-specific token limits
and seed behavior.

## Evaluate a completed run

```bash
hgf-evaluate runs/hgf/results.json
```

The evaluator reports accuracy, multiclass Brier score, and natural-log NLL for
the supplied result file.

## Reproducibility boundary

The public package preserves deterministic inputs, evidence selection,
exemplar selection, prompt construction, schemas, validation, repair behavior,
seed values, boundary mapping, and scoring logic.

New model responses can differ between provider calls when a model alias does
not identify an immutable snapshot. Bundle validation therefore checks the
frozen inputs and deterministic pipeline stages independently of live API
outputs.

## Publication note

No license has been assigned in this folder. Add the project owners' chosen
license and citation metadata before publishing it as a standalone repository.
