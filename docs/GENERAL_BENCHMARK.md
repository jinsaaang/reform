# General Benchmark Contract

The portable launcher assumes that historical DAGs have already been converted
to the canonical topology-preserving Blueprint format. It does not build DAGs.

## Directory layout

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

HGF requires E1. The complete baseline suite additionally requires E0, the raw
DAG manifest, and factor-memory artifacts.

## Questions

Both question files are JSON Lines files accepted by the frozen Pydantic
`Question` model. Each target and historical event must include a stable ID,
question text, answer options, resolution time, and metadata. The metadata must
contain at least the following fields either directly or under `finance`,
`finfactorbench`, or `benchmark`.

```json
{
  "family_id": "recurring_event_family",
  "target_metric": "metric_name",
  "category": "category_name",
  "forecast_date_options": ["2026-01-15"]
}
```

Historical events used as memory must resolve before the target forecast
cutoff. `selection.json` has one field.

```json
{"question_ids": ["target_001", "target_002"]}
```

## Evidence databases

Each SQLite file must contain an `articles` table with these columns.

```text
id, title, source, published_date, content, collected_for_question_id
```

The runtime independently rejects articles at or after the target cutoff.

## Blueprints

The Blueprint manifest schema is `hgf_blueprint_manifest_v1`. Every case uses
`hgf_blueprint_topology_v2`. The manifest must cover every row in
`memory_questions.jsonl`, and its optional canonical hashes must match. A
Blueprint preserves checkpoint IDs, directed edges, conditional paths, lag,
support, confidence, target bridge, and source provenance while excluding the
historical answer from the forecast payload.

## Answer-free worked exemplars

The exemplar directory must contain one worked exemplar for every historical
memory question. The loader accepts JSON files containing a historical ID in
`retrieved_memory_question_id`, `source_question_id`, or
`memory_question_id`, and a `worked_exemplar` object. The method removes the
historical estimate, option mapping, evidence article IDs, and forecast-time
evidence before it constructs the reasoning check.

## Dynamic and frozen execution

The normal portable command reranks current evidence and retrieves compatible
Blueprints at runtime. This is appropriate when transferring the method to a
new benchmark.

For a controlled replay, provide both model-specific manifests.

```bash
python3 scripts/run_hgf.py \
  --dataset-root BENCHMARK_ROOT \
  --model MODEL \
  --provider PROVIDER \
  --output-dir RUN_DIR \
  --evidence-selection-manifest INPUTS/evidence.json \
  --retrieval-manifest INPUTS/retrieval.json
```

The evidence manifest fixes selected current evidence IDs. The retrieval
manifest fixes historical memory question IDs. Supplying only one manifest is
rejected because it would create a partially frozen, ambiguous experiment.

