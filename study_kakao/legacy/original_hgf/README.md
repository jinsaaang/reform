# Original HGF legacy archive

This directory preserves the pre-canonical HGF implementation, its 200
materialized V1 factor cards, 105 source guidance files, former Exemplars,
semantic lessons, migration inputs, configurations, and historical runs.
Canonical code never imports this package.

Run it from the `study_kakao` root in an isolated environment so its `hgf`
package does not shadow the canonical package:

```powershell
python -m venv .legacy-venv
.\.legacy-venv\Scripts\Activate.ps1
pip install -e legacy/original_hgf

$env:HGF_ROOT = (Resolve-Path ".").Path
hgf-legacy-replay `
  --memory-bank-manifest legacy/original_hgf/data/memory_bank/manifest.json `
  --exemplar-dir legacy/original_hgf/artifacts/exemplars `
  --semantic-cache-dir legacy/original_hgf/artifacts/semantic_lessons `
  --output-dir legacy/original_hgf/runs/new_replay
```

The environment variable points legacy graph and evidence references to the
shared immutable dataset. The legacy manifest redirects its former
`guidance_path` entries into this archive. Historical run directories are
ignored by Git but retained locally.
