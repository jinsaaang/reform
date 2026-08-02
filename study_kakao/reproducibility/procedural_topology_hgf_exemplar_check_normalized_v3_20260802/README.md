# Procedural Topology HGF with Worked Reasoning Check, normalized v3

This v3 directory preserves the same forecasting method and adds only
provider-robust reasoning trace normalization. Common aliases such as
`CURRENT_CASE`, combined path identifiers, and target-bridge node identifiers
are mapped to the registered audit identifiers. If a model returns the required
information in `target_semantics`, `causal_balance`, and `magnitude_readiness`
but omits the corresponding baseline or target-bridge step label, that existing
content is projected into the missing step. Every action is recorded under
`trace_normalization`. No answer, probability, or external prediction is used
or modified.

Any unregistered audit label is handled conservatively. Target-definition
steps map to `TARGET_CONTRACT`, while evidence-grounded driver, mechanism, and
counterevidence steps map to `CURRENT_NEW`. Unknown labels never create a claim
that a DAG path was used.

This isolated variant preserves the registered Procedural Topology HGF pipeline
and tests one focused correction. A historical DAG should improve the structure
of current reasoning without becoming a complete answer template or a hard
constraint on what the forecaster may consider.

The pipeline first reads current cutoff-safe evidence and builds an evidence
ledger. It retrieves exact-family historical DAGs, routes their relevant intact
subgraphs, and fills their nodes and edges with current evidence. The forecaster
then uses helpful nodes, edges, and partial paths as an incomplete structural
scaffold. It may add current-only factors and intermediate reasoning. An
explicitly contradicted relation cannot support the forecast, while an
unverified relation remains an uncertain hypothesis rather than an automatic
failure.

Each selected historical DAG also supplies a fixed worked reasoning check. The
check retains the historical forecast-time reasoning order, counterevidence,
uncertainty, and structural lesson. It excludes the historical target estimate,
option mapping, evidence payload, answer, and probability. The forecaster uses
the check only to review whether its current argument omitted a baseline,
mechanism, competing explanation, or uncertainty. The check is built once and
shared across models. It creates no additional live LLM stage.

The response schema requests at most ten reasoning steps, matching the
registered baselines. Provider-side schema enforcement is not assumed. The
observed trace length is therefore reported exactly as returned rather than
being truncated after generation. The target boundary mapper, single
probability call, model-specific frozen evidence, frozen retrieval, and
absence of probability postprocessing remain unchanged. The registered
canonical method is not modified.

Run contract tests.

```bash
PYTHONPATH=method_src:../procedural_topology_hgf_full100_20260802/hgf_historical_base_src:../../src \
python3 -m pytest -q -s -p no:cacheprovider -c /dev/null test_contract.py
```

Run a fresh Gemini Flash Lite full-100 experiment.

```bash
python3 run.py \
  --models google/gemini-2.5-flash-lite \
  --selection-file ../../data/questions/selection.json \
  --limit 100 \
  --workers-per-model 20 \
  --max-parallel-models 1 \
  --output-root ../../runs/procedural_topology_hgf_exemplar_check_full100_20260802
```

The exact final run roots and aggregate results are recorded in
`FINAL_RESULTS.md` and `SYNC_MANIFEST.md`. Qwen uses a separate provider
compatibility adapter because the Alibaba endpoint did not reliably preserve
the requested structured response when native hidden reasoning was enabled.
The adapter normalizes field shapes only. It never changes the predicted
option or probability. This Qwen run is an exploratory transfer result and is
not pooled with the five-model controlled comparison.
