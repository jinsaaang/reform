# Finance experiment framework

## Protocol

The checked-in experiment manifest fixes the binary targets, aware cutoff,
`finance-evidence-snapshot/v1` evidence pack, search policy, retrieval bound,
forecast and judge settings, arm order, retries, and trial seeds before a run.
The live path uses the `live-publication-filtered` temporal policy: evidence is
admitted only when its publication timestamp is before the manifest cutoff.
This is not leakage-free frozen retrospective data; current live publication
availability can change and therefore does not constitute a historical
backtest.

## Commands

Run the three-arm suite with explicit paths:

```bash
wr finance experiment-run \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --seed-manifest docs/research/finance_seed_v1_manifest.json \
  --experiment-manifest configs/experiments/finance_live_10_2026-07-18.json \
  --output-dir artifacts/finance/experiments/run-001 \
  --json
```

After outcomes are available, analyze a verified ex-ante suite into a new
directory. The analyzer never rewrites the source suite or its judge results:

```bash
wr finance experiment-analyze \
  --suite artifacts/finance/experiments/run-001 \
  --resolution-manifest outcomes/finance-run-001.json \
  --output-dir artifacts/finance/experiments/run-001-resolution \
  --json
```

Both commands require explicit destinations. A malformed input, existing
destination, or immutable DB identity mismatch exits nonzero before live
provider construction.

### Resolved-question pilot backtest

Prepare a manifest from graph-built binary finance questions that have both a
realized outcome and an `estimated_start_time`. The current v1 seed yields 22
eligible questions. The generated manifest contains no ground-truth field and
assigns each question its own simulated forecast cutoff:

```bash
wr finance backtest-prepare \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --seed-manifest docs/research/finance_seed_v1_manifest.json \
  --template-manifest configs/experiments/finance_live_10_2026-07-18.json \
  --output-file configs/experiments/finance_resolved_backtest_v1.json \
  --manifest-id finance-resolved-backtest-v1 \
  --model openrouter/openai/o4-mini \
  --limit 10
```

Run that manifest through the same A/B/C runner, then join outcomes only after
the suite has been persisted:

```bash
wr finance experiment-run \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --seed-manifest docs/research/finance_seed_v1_manifest.json \
  --experiment-manifest configs/experiments/finance_resolved_backtest_v1.json \
  --output-dir artifacts/finance/experiments/resolved-v1

wr finance backtest-analyze \
  --suite artifacts/finance/experiments/resolved-v1 \
  --db data/releases/worldreasoner/v1.0.0/worldreasoner_public.db \
  --seed-manifest docs/research/finance_seed_v1_manifest.json \
  --output-dir artifacts/finance/experiments/resolved-v1-analysis
```

`backtest-analyze` reports per-arm correctness at a 0.5 threshold, binary
Brier scores, macro accuracy, macro Brier, and whether adding the second arm
improved or worsened Brier. The ex-ante suite remains ground-truth-free.
The model flag applies to both the reasoning forecaster and blind judges. Use
a model whose declared knowledge cutoff predates every simulated forecast
cutoff; the example uses the contamination-control model from the earlier
direction pilot rather than the current-question GPT-5.6 configuration.

To run only the 13 eligible questions not used in the earlier 10-question
DAG-direction pilot, add:

```bash
  --exclude-question-file \
  configs/experiments/finance_dag_direction_previous_10_ids.txt
```

The earlier set contains one GM question without `estimated_start_time`, so
the arithmetic is 22 currently eligible targets minus 9 overlapping targets,
leaving 13 new targets.

## Artifacts

`experiment-run` atomically writes a new directory containing `suite.json`,
the deterministic ex-ante `report.md`, and `SHA256SUMS`. The suite records
Direct, Search-only, and Search+DAG terminal arms, pairwise A-B/B-C/A-C panel
diagnostics, sanitized structured reasoning, and stable provider-input
digests. The JSON output reports the destination, suite digest, report digest,
status, and scheduled trial count without credentials, headers, raw bodies, or
provider payloads.

`experiment-analyze` and `backtest-analyze` write a separate `analysis.json`,
`report.md`, and `SHA256SUMS` bundle. They bind outcomes to the verified suite
digest and emit binary Brier, threshold accuracy, and pair direction metrics
only for that derived bundle.

## Limitations

The public seed DB is immutable and contains only the checked-in bootstrap
episodes. Historical relation metadata is therefore admitted under the
declared public-DB policy, and lexical top-k retrieval can select noisy or
weakly related episodes. Live RSS publication and provider availability are
not reproducible frozen snapshots. Resolution analysis is intentionally a
separate command: no outcome, Brier score, or realized target value enters
the ex-ante suite.

The resolved pilot does not claim a frozen historical web corpus. Its Search
arm ranks only title/source/tag/URL metadata already present in the pinned
WorldReasoner DB and filters `published_date < forecast_cutoff`. The policy is
explicitly named `public_db_metadata_proxy/v1`: article inclusion, titles, and
the model's parametric knowledge can still encode hindsight. It is suitable
for an initial pipeline and directionality diagnostic, not the paper's main
leakage-minimized result. A main retrospective claim still requires verified
pre-cutoff body versions, while prospective unresolved evaluation remains the
cleanest validation.

The provider preserves the DB row's real `created_at` as snapshot provenance;
it does not relabel publication time as capture time. Because that metadata
record can have been created after the simulated cutoff, this source receives
an explicit proxy-only admission exception. Only `published_date` is cutoff
filtered. Consequently these Search results must never be described as frozen
historical bodies or leakage-free evidence.
