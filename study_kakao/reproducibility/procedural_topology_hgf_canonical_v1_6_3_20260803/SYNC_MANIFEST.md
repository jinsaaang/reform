# Paper and experiment sync manifest

## Canonical version

- Branch: `experiment/live-topology-hgf-v1`
- Repository basis commit: `9353138350bb9ceb7230bb22f5fec6a8287a6124`
- Canonical archive commit: `8ae09cc223d5b3c93a17815c1cba8311fc24bb23`
- Sync manifest commit: the follow-up commit containing this line
- Method: Procedural Topology HGF
- Bundle: `reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803`
- Paper source to update later: `/home/kong/code/study_kakao/paper/main1.tex`
- Bibliography: `/home/kong/code/study_kakao/paper/reference.bib`

This bundle is a separate canonical archive. It was not merged into or copied
over the original `/home/kong/code/study_kakao` implementation tree.

## Fixed evaluation contract

- Question selection: `data/questions/selection.json`
- Selection SHA256: `13e18cfd89300e819c1dd0a35450caa1a3cd7fa8dededa1b6c83cfffbff7eba9`
- Questions per model: 100
- Seed: 0
- Models: Gemini 2.5 Flash Lite, GPT-5 mini, DeepSeek V3.2, Llama 4 Maverick, and MiniMax M2.5
- Metrics: Accuracy, Brier score, and negative log likelihood
- Methods in the main comparison: Structured Direct Forecasting, DAG Forecasting, Outcome-Neutral Direct DAG Retrieval, Factor Memory, Outcome-Redacted Case Retrieval, Forecasting Principles, and Procedural Topology HGF
- Each model uses its own frozen cutoff-safe evidence selection and frozen exact-family historical retrieval.
- Provider-native reasoning effort is medium for Gemini, GPT, DeepSeek, and MiniMax. It is disabled for Llama because that endpoint did not reliably support the same native control. Explicit HGF reasoning is required for every model.
- No probability pooling, posterior adjustment, baseline anchoring, answer reuse, semantic fallback, or score-conditioned retry is used.
- Validity recovery selects the first contract-valid execution in declared run order. It never examines forecast score.

## Method actually executed

The current evidence is read first to lock the target operation, establish the
available baseline, identify present drivers, and expose missing information.
HGF then retrieves outcome-redacted subgraphs from resolved events in the same
event family. It routes the relevant partial paths, instantiates their nodes and
relations using current evidence, and lets the forecaster use supported paths as
an expert reasoning scaffold. The model may reject a historical relation, add a
current factor absent from history, and must consider counterevidence and
uncertainty. A separate boundary call maps the resulting direction and magnitude
assessment to the exact public answer space once.

The historical graph is not a prior answer and is not enforced as a complete
template. The answer-free worked reasoning check preserves useful procedure but
contains no historical answer, option mapping, estimate, or probability.

Version 1.6.3 changes execution validity only. Empty or length-terminated model
content can no longer be converted into generic reasoning placeholders. A valid
row must contain substantive current-evidence-grounded reasoning, at least three
material reasoning steps, explicit counterevidence and uncertainty, and valid
evidence and path provenance. Fixed step labels are not required.

## Canonical code and inputs

- Launcher: `run.py`
- Failure-only launcher: `recover_failures.py`
- Forecast method: `method_src/hgf_e2e_topology`
- Frozen shared implementation: `hgf_historical_base_src/hgf`
- Frozen input adapter: `input_adapter_src/hgf_original_input_adapter`
- Provider-only routing and raw-call recorder: `execution_src`
- Model-specific evidence and retrieval: `inputs/model_evidence`
- Exact MiniMax validity-recovery selection: `inputs/reasoning_validity_recovery_selection.json`
- Registered baseline table: `inputs/registered_baselines/main_method_results_20260802.csv`
- Contract tests: `test_contract.py`
- Final validity gate: `finalize_results.py`
- Reasoning provenance audit: `audit_reasoning.py`
- Paper and resource table builders: `build_paper_table.py` and `build_resource_table.py`

The executable-code bundle passed 12 contract tests and a fresh one-question
smoke run while importing the method, provider wrapper, recorder, and provider
serialization code exclusively from this bundle.

## Final results

| Model | N | Acc | Brier | NLL | DAG path use |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash Lite | 100 | 0.550 | 0.2131 | 0.9319 | 85% |
| GPT-5 mini | 100 | 0.550 | 0.2158 | 0.9475 | 83% |
| DeepSeek V3.2 | 100 | 0.520 | 0.2202 | 0.9129 | 77% |
| Llama 4 Maverick | 100 | 0.500 | 0.2212 | 0.9302 | 100% |
| MiniMax M2.5 | 100 | 0.530 | 0.2180 | 0.9203 | 97% |
| Pooled | 500 | 0.530 | 0.2177 | 0.9286 | 88.4% |

The strongest pooled baseline by Brier is Outcome-Neutral Direct DAG Retrieval
with Accuracy 0.494 and Brier 0.2324. HGF lowers Brier by 6.3% and raises
accuracy by 3.6 percentage points. HGF has the lowest Brier score for all five
models.

The deterministic reasoning contract is valid for 500 of 500 selected rows.
This verifies completeness and provenance, not semantic quality. The paper's
LLM-judge study remains a separate evaluation and must not be replaced by this
audit.

Selected results contain 24,449,026 tokens and cost $11.1158. The complete
campaign, including failed transport and output-contract attempts, contains
31,686,903 tokens across 2,576 recorded calls and costs $15.6311.

## Selected providers

| Model | Provider cases in the final 100 |
|---|---|
| Gemini 2.5 Flash Lite | Google AI Studio 100 |
| GPT-5 mini | OpenAI 100 |
| DeepSeek V3.2 | Baidu 82, AtlasCloud 18 |
| Llama 4 Maverick | DeepInfra 100 |
| MiniMax M2.5 | Friendli 67, Inceptron 33 |

Provider changes are transport recovery only. The model slug, method, prompt,
frozen evidence, frozen retrieval, seed, reasoning policy, and output contract
remain unchanged.

## Result roots and selection order

The main root is
`runs/procedural_topology_hgf_canonical_v1_6_full100_all_models_20260803`.
The declared recovery order is recorded verbatim in
`final_results_v1_6_3/FINAL_RESULTS.json`. It includes the registered Gemini,
DeepSeek, Llama, GPT, and MiniMax transport recoveries followed by the MiniMax
reasoning-validity recoveries under
`runs/procedural_topology_hgf_canonical_v1_6_3_reasoning_recovery*_20260803`.
Rejected provider probes are retained in campaign accounting and cannot replace
a valid row.

The sanitized baseline source is
`runs/baseline_sanitation_full100_v1_2_20260802/canonical`. Its Direct DAG and
Resolved Case reruns contain 100 successful rows for all five models and match
the registered baseline table.

## Artifact hashes

- Config: `fe7b8e43a85a9decfd96a3bcf63e941d89005e31fce329b964843977230ea479`
- Registered baselines: `351b6fe74c76e1fc225acafb4afdfd5c7ecc32b7337ee53344d0d744c1763bb0`
- Final result JSON: `531be8793f773b2cbcc727504e256b001d04b40102f48e3fb0daa3fc2d4e2a9d`
- Main comparison CSV: `37c0d9c2c10609495c5ab231571f5c1141166dbfec390feec3c9204b6179de12`
- Reasoning audit JSON: `03570917c413a696863b0c12e54d7334f793ca182ed96b29806a70373dcab2e0`
- Resource summary CSV: `cb9f2d1a0ce09b858d29076074ce69ef9020ea03a752488f0aa5016e8693ac9d`

## Paper changes required

- Keep the current motivation and benchmark story, but replace the single-graph v27 method description with the executed Procedural Topology HGF pipeline above.
- Replace the single selected memory notation with a small set of eligible same-family subgraphs and distinguish routed partial paths from their current instantiation.
- Describe current-evidence reading before historical transfer, flexible use or rejection of historical relations, current-only factors, counterevidence, uncertainty, and the separate one-time boundary mapping.
- Describe the worked reasoning check as answer-free procedural guidance, not as a second memory or a historical answer.
- Expand the financial table from three to five models and seven methods using the exact 100-question rows in `final_results_v1_6_3/MAIN_COMPARISON.csv`.
- Remove the current claim of three independent runs and mean plus or minus standard deviation. The registered table is one fixed seed-0 run over 100 questions per model.
- Report model-specific provider routes and the Llama native-reasoning exception accurately.
- Report cost, token, and latency from `RESOURCE_SUMMARY.csv` separately from performance.
- Use the 500 of 500 deterministic reasoning audit only as an implementation-integrity statement. Report LLM-judge reasoning quality as a separate experiment when available.
- Do not use internal names such as v27 or v1.6.3 in the paper method description.
