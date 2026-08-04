# Procedural Topology HGF v1.7.0 strict

This is a separated copy of the frozen v1.6.0-strict bundle. The parent
directory is not imported or modified at runtime. v1.7.0 fixes one boundary
contract conflict discovered on three-way recent-range questions whose public
`within recent range` interval does not contain zero.

The arithmetic option for the model's central estimate is always preserved.
Weak magnitude evidence now limits confidence instead of prohibiting a below-
or above-range modal option:

- `direction_only`: every option probability must be at most `0.50`.
- `insufficient`: every option probability must be at most `0.45`.

These caps apply only to three-option recent-range contracts. Binary and other
contracts use ordinary normalized probabilities summing to `1.0`.

No prediction, central estimate, mapped option, or probability is changed by
code. The model receives the joint mapping and confidence constraints as
validation feedback and must return a conforming forecast. Probability
postprocessing and score-conditioned selection remain disabled.

When the arithmetic option and probability argmax conflict, repair feedback
explicitly tells the model to preserve the arithmetic mapping and adjust the
probabilities. This avoids alternating between two individually incomplete
repairs.

## One-command Qwen seed-0 reproduction

From the repository root, with `OPENROUTER_API_KEY` available:

```bash
.venv/bin/python study_kakao/reproducibility/procedural_topology_hgf_canonical_v1_7_0_strict_20260804/run_qwen_seed0.py
```

Defaults are fixed in the script:

- model: `qwen/qwen-plus-2025-07-28`
- provider: Alibaba only, no fallback
- native provider reasoning: disabled
- seed: `0`
- reasoning effort: `medium`
- maximum output tokens: `16,000`
- workers: `20`
- frozen selection: `100` questions
- maximum full trials: `2`
- per-trial wall timeout: `720` seconds; completed reportable cases are harvested

Each trial uses a fresh persistent directory under `results/`. Only the first
independently reportable case is merged into `full100`; rejected trials, raw
requests/responses, prediction audits, source hashes, status, and metrics are
retained. The script exits nonzero if 100 audit-clean cases are not obtained by
the end of trial 2.

Use `--dry-run` to inspect all resolved defaults without creating files or
calling an API.

## Contract tests

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=study_kakao/reproducibility/procedural_topology_hgf_canonical_v1_7_0_strict_20260804/method_src:study_kakao/reproducibility/procedural_topology_hgf_canonical_v1_7_0_strict_20260804/hgf_historical_base_src:study_kakao/reproducibility/procedural_topology_hgf_canonical_v1_7_0_strict_20260804/input_adapter_src:study_kakao/reproducibility/procedural_topology_hgf_canonical_v1_7_0_strict_20260804/execution_src \
.venv/bin/python -m pytest -q -s -p no:cacheprovider -c /dev/null \
study_kakao/reproducibility/procedural_topology_hgf_canonical_v1_7_0_strict_20260804/test_contract.py
```

The copied historical campaign utilities remain for parent provenance. The
supported v1.7.0 Qwen entrypoint is `run_qwen_seed0.py`.
