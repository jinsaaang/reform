# Financial WorldReasoner: Team Setup and Reproduction Guide

This guide is the portable handoff for the finance extension. It covers local
setup, the immutable WorldReasoner seed database, offline smoke tests, paid
experiments, and the maintained quality gate. Commands are run from the
repository root and do not depend on a personal filesystem layout.

## What is implemented

The finance extension provides:

- a read-only loader for resolved WorldReasoner finance DAG episodes;
- a two-pass Searcher with temporal evidence admission;
- a non-graph-building Forecaster that consumes admitted evidence and optional
  historical DAG memory;
- Direct, Search-only, and Search+DAG experiment arms;
- order-balanced pairwise LLM judging with complete reasoning artifacts;
- post-resolution Brier score and accuracy analysis;
- atomic `suite.json`, `report.md`, and SHA-256 receipts.

The Forecaster never creates a DAG. It only references resolved historical DAGs
selected from the offline memory.

## Prerequisites

- Python 3.11 or newer; Python 3.13 is recommended.
- [uv](https://docs.astral.sh/uv/).
- GitHub CLI (`gh`) for the seed database download, or an equivalent manual
  download.
- OpenRouter credit and `OPENROUTER_API_KEY` only for paid forecasting/judging.
- `SEARXNG_BASE_URL` only for the current live-search path. The public-DB
  metadata backtest does not require it.

## Install

```bash
git clone <team-repository-url>
cd <repository-directory>
uv sync --locked
```

Browser-backed live fetching additionally requires:

```bash
uv run playwright install chromium
```

Copy the environment template for paid or live runs:

```bash
cp .env.example .env
```

Set secrets in `.env` or the shell. Never add `.env` or a provider key to Git.

## Download the immutable seed database

The expected asset is the WorldReasoner `v1.0.0` public database:

```bash
mkdir -p data/releases/worldreasoner/v1.0.0
gh release download v1.0.0 \
  --repo cyzus/worldreasoner \
  --pattern worldreasoner_public.db \
  --dir data/releases/worldreasoner/v1.0.0
chmod 0444 data/releases/worldreasoner/v1.0.0/worldreasoner_public.db
```

The canonical asset contract is:

| Field | Value |
|---|---|
| Size | `55,554,048` bytes |
| SHA-256 | `94ffd8cca51906edec0b05f7e94e78de80d26f268f082521bded80d0aed06fab` |
| Finance graph-built questions | 37 |
| Question types | 23 binary, 1 MCQ, 8 quantity, 5 timeframe |

The database is ignored by Git and opened using an immutable read-only SQLite
URI. Do not migrate it or build an index beside it.

## Verify the seed and CLI

```bash
uv run wr finance seed-audit \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --manifest docs/research/finance_seed_v1_manifest.json \
  --json
```

Expected status is `ok`, with 37 unique finance questions and no SQLite
sidecars.

Run both deterministic offline modes before using a provider:

```bash
uv run wr finance pipeline-smoke \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --question 'Will NVIDIA revenue exceed analyst expectations?' \
  --cutoff 2026-06-01T00:00:00+00:00 \
  --context 'GPU demand and semiconductor quarterly revenue' \
  --mode current --json

uv run wr finance pipeline-smoke \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --question 'Will NVIDIA revenue exceed analyst expectations?' \
  --cutoff 2026-06-01T00:00:00+00:00 \
  --context 'GPU demand and semiconductor quarterly revenue' \
  --mode historical --json
```

These commands use fixture providers and incur no model or search cost.

## Run a current forecast

After setting `OPENROUTER_API_KEY` and the live search configuration:

```bash
uv run wr finance pipeline-run \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --manifest docs/research/finance_seed_v1_manifest.json \
  --question 'Will NVIDIA revenue exceed analyst expectations?' \
  --context 'GPU demand and semiconductor quarterly revenue' \
  --cutoff 2026-06-01T00:00:00+00:00 \
  --mode current \
  --top-k 3 \
  --search-result-limit 5 \
  --json
```

`pipeline-run` intentionally rejects historical mode because live web pages are
not verified frozen pre-cutoff snapshots.

## Reproduce an experiment

The tracked manifests under `configs/experiments/` contain no provider keys or
ground-truth fields. A paid three-arm run is:

```bash
uv run wr finance experiment-run \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --seed-manifest docs/research/finance_seed_v1_manifest.json \
  --experiment-manifest configs/experiments/finance_resolved_remaining_12.json \
  --output-dir artifacts/finance/experiments/team-run \
  --json
```

Join outcomes only after the ex-ante suite is complete:

```bash
uv run wr finance backtest-analyze \
  --suite artifacts/finance/experiments/team-run \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --seed-manifest docs/research/finance_seed_v1_manifest.json \
  --output-dir artifacts/finance/experiments/team-run-analysis \
  --json
```

The full protocol and manifest preparation commands are documented in
[finance_experiment_framework.md](finance_experiment_framework.md).

## Maintained quality gate

Run the team check before opening a pull request:

```bash
bash scripts/finance_check.sh
```

It runs Ruff lint and format checks, strict basedpyright, and the offline
finance test suite. No provider key is required. The repository's older live
integration tests outside the finance scope require additional services and are
not part of this gate.

## Artifact policy

- `configs/experiments/finance_*.json` and `finance_*.txt` are tracked and must
  not contain outcomes or credentials.
- Raw model suites under `artifacts/` are intentionally ignored because they
  are large generated records. Share them through a controlled release or team
  storage when full reasoning audit is required.
- Compact aggregate metrics and reports belong under `docs/research/` and are
  tracked.
- Every shared raw suite must retain its `SHA256SUMS` receipt.

## Important validity limitations

The public database contains article metadata but not article bodies. It also
lacks exact `resolution_available_at` and several same-event relation fields.
The bootstrap policy therefore uses explicit audit markers and must not be
described as a fully frozen historical web corpus.

For the 13-question resolved pilot, temporal eligibility left only one to three
usable historical DAGs per question. The result is evidence about the current
sparse-memory treatment, not a general verdict on well-covered finance DAG
retrieval. See
[finance_resolved_remaining_13_results_2026-07-18.md](finance_resolved_remaining_13_results_2026-07-18.md).

## Research scope

The complete research questions, non-claims, and planned memory expansion are
in [FINANCIAL_WORLDREASONER_RESEARCH_DIRECTION.md](../../FINANCIAL_WORLDREASONER_RESEARCH_DIRECTION.md).
