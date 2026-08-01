# Procedural Topology HGF full-100 reproduction bundle

This directory freezes the implementation and inputs used for the 2 August 2026 registered run. It is independent of the experimental HGF variants elsewhere in the repository.

## What is frozen

- `hgf_method_src` is the exact Procedural Topology HGF implementation archived at commit `a3b07a06e51772bb25d2fb99b3d36a61fccc2898`.
- `hgf_historical_base_src` contains the exact shared `hgf` dependencies loaded from commit `27ff13cf8b2e1f20e88822e895a7b02055d9be30`.
- `hgf_input_adapter_src` is the non-forecasting adapter used to supply model-specific frozen evidence and retrieval manifests.
- `baseline_src` contains the exact Outcome-Redacted Case Retrieval and Outcome-Neutral Direct DAG Retrieval implementation.
- `inputs` contains the five model-specific evidence and retrieval manifests and the frozen outcome-neutral topology cache.
- `manifests` contains the canonical HGF and baseline provenance plus the completed full-100 comparison.

The canonical HGF performs no probability pooling, posterior adjustment, baseline-answer reuse, or result-conditioned retry. Historical worked exemplars are excluded in this exact implementation.

## Replay

Set `OPENROUTER_API_KEY` in the environment, then run from the repository root.

```bash
python reproducibility/procedural_topology_hgf_full100_20260802/replay_full100.py \
  --suite both \
  --workers-per-model 8 \
  --max-parallel-models 5 \
  --output-root runs/replay_procedural_topology_full100
```

Use `--dry-run` to inspect every command without making API calls. The output root must be new. Provider or parsing failures must be recovered only for the exact failed question and method in a fresh directory. Successful predictions must not be resampled or selected by score.

## Registered scope

The main comparison uses five models, seven methods, one fixed set of 100 questions per model, and seed 0. Four earlier baseline methods remain from the registered 1 August suite. HGF and the two leakage-sanitized history baselines were run afresh on 2 August. The resulting 3,500-row table is in `experiments/final_results_20260802`.

All final performance claims must use the complete 100 questions for each model. The fixed 40-question subset is diagnostic only.
