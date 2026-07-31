# Hindsight-Guided Forecasting

`study_kakao` contains the canonical HGF implementation, six comparison
baselines, frozen questions and evidence, and reproducible memory artifacts.
The public method name is `hgf`; tables display it as `HGF (Ours)`.

## Canonical layout

```text
src/hgf/                         canonical HGF and six baselines
data/                            questions, evidence, and refined DAGs
artifacts/hgf/blueprints/        200 topology-preserving Blueprints
artifacts/hgf/exemplars/memory/  200 cutoff-safe memory Exemplars
artifacts/hgf/exemplars/cases/   fixed mapping for 100 evaluation cases
artifacts/baselines/factor_memory/
                                 frozen 200-card Factor-Memory input
runs/hgf/                        preserved canonical HGF results
runs/baselines/                  preserved baseline-only results
legacy/original_hgf/             executable archive of the former HGF
```

The HGF and Factor-Memory loaders are intentionally separate. HGF accepts only
the complete `artifacts/hgf` root; a Blueprint-only override is not supported,
because it could silently mismatch the Blueprint and Exemplar.

## Setup and verification

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"

hgf-build-memory --check
hgf-build-exemplars --check
hgf-preflight
python -m pytest
python -m compileall -q src
hgf-verify
```

To rebuild the deterministic Blueprint bank:

```powershell
hgf-build-memory
```

To resume Exemplar generation, provide one or more validated seed banks. Valid
existing memory files are retained and only missing entries invoke the model.

```powershell
hgf-build-exemplars `
  --seed-dir artifacts/hgf/exemplars `
  --workers 10
```

`OPENROUTER_API_KEY` is required only when an experiment or missing Exemplar
needs a model call.

## Experiments

Run canonical HGF alone:

```powershell
hgf-replay --workers 10 --output-dir runs/hgf_replay
```

Run HGF and all six baselines under the same protocol:

```powershell
hgf-main-table --workers 10 --output-dir runs/main_table
```

Both commands use the same HGF execution function. The HGF route always checks
exact `family_id + target_metric` compatibility, removes incompatible memory,
uses a sanitized demonstration, enforces the current target operator and
boundary contract, and records the model's boundary probability without
temperature scaling, boosting, or other probability postprocessing.

The complete HGF artifact can be replaced only as one unit:

```powershell
hgf-main-table --hgf-artifact-root D:\frozen_hgf_artifact
```

See [HGF.md](HGF.md) for the method contract. The former implementation and
its results are preserved only under
[legacy/original_hgf](legacy/original_hgf/README.md).
