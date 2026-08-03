# Canonical Procedural Topology HGF v1.6.3

This bundle freezes the paper method and the exact execution contract used for
the five-model financial forecasting experiment.

The method first reads the current cutoff-safe evidence and establishes the
target operation, available baseline, current drivers, and data gaps. It then
retrieves outcome-redacted subgraphs from resolved events in the same event
family. Their original node and edge relations are instantiated with current
evidence. The forecaster uses helpful partial paths as an expert reasoning
scaffold, adds current factors missing from history, tests counterevidence,
and connects the resulting direction and magnitude support to the public
forecast boundary.

The historical graph is neither a prior answer nor a mandatory template.
Current evidence overrides it. An unverified relation may identify a question
to check but is not a present fact. A contradicted relation can only be used as
counterevidence. No fixed path count or step label is required. The output
contract instead requires substantive current-evidence-grounded reasoning,
explicit counterevidence and uncertainty, and a target direction and magnitude
assessment.

Historical answers, estimates, option mappings, and probabilities are
excluded. There is no probability pooling, posterior adjustment, baseline
anchoring, answer reuse, semantic forecast fallback, or score-conditioned
retry. The boundary call produces probabilities once. Validators never change
an estimate or probability.

Version 1.6.3 fixes an execution-validity bug without changing this forecast
procedure. A provider response that exhausted its output budget could return
hidden reasoning but no JSON content. The previous parser converted that empty
content to an empty object, and the reasoning validator filled it with generic
placeholders. The parser now treats missing or length-terminated JSON as a
failed execution, and the reasoning validator rejects semantically empty
objects. Recovery is allowed only for such contract failures and selects the
first contract-valid execution in declared run order, never the best score.

The bundle is self-contained with respect to executable Python code, model
configuration, frozen model-specific evidence selection, and frozen historical
retrieval manifests. Repository-tracked questions, evidence databases,
blueprints, and answer-free exemplars remain explicit data inputs.

## Layout

- `method_src/hgf_e2e_topology` contains the complete forecasting method.
- `hgf_historical_base_src/hgf` contains its frozen shared implementation.
- `input_adapter_src/hgf_original_input_adapter` freezes evidence and retrieval.
- `execution_src` contains the raw-call recorder and provider-only wrapper.
- `inputs/model_evidence` contains model-specific evidence and retrieval manifests.
- `config.json` pins models, providers, token budgets, reasoning effort, and seed.
- `run.py` launches a fresh suite and records the complete command and source hashes.
- `recover_failures.py` retries only execution failures.
- `finalize_results.py` applies prediction and reasoning validity gates.
- `audit_reasoning.py` audits reasoning completeness and provenance without an LLM judge.
- `build_paper_table.py` joins validated HGF results with the registered baselines.

## Contract tests

```bash
PYTHONPATH=reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/method_src:reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/hgf_historical_base_src:reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/input_adapter_src:reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/execution_src \
python3 -m pytest -q -s -p no:cacheprovider -c /dev/null \
reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/test_contract.py
```

## Full suite

```bash
python3 reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/run.py \
  --models google/gemini-2.5-flash-lite openai/gpt-5-mini deepseek/deepseek-v3.2 meta-llama/llama-4-maverick minimax/minimax-m2.5 \
  --selection-file data/questions/selection.json \
  --limit 100 \
  --workers-per-model 20 \
  --max-parallel-models 5 \
  --output-root runs/procedural_topology_hgf_canonical_full100
```

Every output root must be new. The launcher records transport-only provider
overrides, if any, in the suite manifest. A successful prediction is never
resampled or replaced by score.

## Final validity and reasoning audit

```bash
python3 reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/finalize_results.py \
  --main-root runs/MAIN_RUN \
  --recovery-roots runs/RECOVERY_1 runs/RECOVERY_2 \
  --selection-file data/questions/selection.json \
  --output-dir reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/final_results_v1_6_3

python3 reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/audit_reasoning.py \
  --final-results reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/final_results_v1_6_3/FINAL_RESULTS.json \
  --output-dir reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803/final_results_v1_6_3
```

The deterministic reasoning audit checks completeness and provenance. It is
not a substitute for the paper's separate LLM-judge evaluation.
