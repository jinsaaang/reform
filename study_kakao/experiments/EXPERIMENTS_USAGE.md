# Paper experiment extensions

These entry points implement `../experiments.md` without modifying the frozen
forecasting code under `src/hgf`.

## Live-model runners

- `run_paper_main_table.py`: three models, seven methods, three sequential
  repetitions, four workers.
- `run_component_ablation.py`: Raw DAG, Full HGF, and three component removals.
- `run_topk_sensitivity.py`: fixed-rule `k = 1, 3, 5, 7` sensitivity.
- `run_reasoning_judge.py`: blinded Raw-DAG versus Full-HGF reasoning judge.
  It accepts either component-ablation outputs (`raw_dag`, `full_hgf`) or
  completed main-table outputs (`direct_dag`, `hgf`).
- `run_all_paper_experiments.py`: sequential master orchestrator.

Every live runner requires `OPENROUTER_API_KEY`. The reasoning judge defaults
to Gemini 3.1 Flash Lite (`google/gemini-3.1-flash-lite`) and reports the
paper-defined evidence coverage, invalid reasoning, and invalid-among-correct
rates.

Use `--dry-run` with the main-table or master runner to inspect commands without
calling a model.

To judge three completed main-table repetitions with the paper metrics:

```bash
python experiments/run_reasoning_judge.py \
  --forecast-results runs/repeated_main_table/gemini_2_5_flash_lite/repeat_1/results.json \
  --forecast-results runs/repeated_main_table/gemini_2_5_flash_lite/repeat_2/results.json \
  --forecast-results runs/repeated_main_table/gemini_2_5_flash_lite/repeat_3/results.json \
  --workers 30 \
  --reasoning-effort medium \
  --max-tokens 8000 \
  --dry-run \
  --output-dir runs/reasoning_quality/gemini_2_5_flash_lite
```

Remove `--dry-run` after the input summary confirms 300 paired questions.
The judge receives each forecast's own cutoff-safe evidence, while method,
forecaster model, and ground truth remain blinded. Ground truth is joined only
after the invalid-reasoning decision to calculate invalid among correct.

## Offline analysis

- `analyze_main_table.py`
- `analyze_ablation.py`
- `analyze_topk.py`
- `analyze_reasoning_judge.py`
- `build_case_studies.py`
- `verify_experiment_extensions.py`

The analysis scripts do not call an external model.

`verify_experiment_extensions.py` checks every file listed by the original
frozen artifact manifest, permits the additive experiment files, and reports a
separate digest for the extension layer.

## Frozen-artifact boundary

The canonical HGF artifact root contains 200 memory exemplars and 100 fixed
test-case mappings. Top-k sensitivity uses those exemplars together with the
same fixed semantic lessons as canonical HGF. Replace the complete matched
artifact bank with `--hgf-artifact-root`; partial Blueprint or Exemplar
overrides are intentionally unsupported. The top-k runner writes
`preflight.json` and stops before any model call if coverage is incomplete or
if the computed rank-1 item differs from the fixed k=1 mapping.

All runners preserve the canonical stage-specific token limits unless an
explicit override is supplied.
