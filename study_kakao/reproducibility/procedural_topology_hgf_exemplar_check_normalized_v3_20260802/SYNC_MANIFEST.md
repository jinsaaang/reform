# Paper and experiment sync manifest

## Version

- Branch: `experiment/live-topology-hgf-v1`
- Basis commit: `e8d6dc1`
- Method: Procedural Topology HGF with an answer-free worked reasoning check
- Question set: `data/questions/selection.json`
- Question set SHA256: `13e18cfd89300e819c1dd0a35450caa1a3cd7fa8dededa1b6c83cfffbff7eba9`
- Questions per model: 100
- Run seed: 0
- Reasoning policy: explicit HGF reasoning for every model. Provider-native
  medium reasoning for Gemini, GPT, DeepSeek, and MiniMax. Provider-native
  reasoning disabled for Llama and exploratory Qwen.
- Worker limit: 20 per active model
- Metrics: Accuracy, Brier score, and negative log likelihood

## Code and launcher

- Launcher: `run.py`
- Input adapter: `input_adapter_src/hgf_original_input_adapter`
- Final method source: `method_src/hgf_e2e_topology`
- Contract tests: `test_contract.py`
- Qwen evidence builder: `qwen_build_evidence.py`
- Qwen provider adapter: `qwen_provider_adapter_src/hgf_qwen_provider_adapter`

The final method retrieves outcome-redacted exact-family historical subgraphs,
grounds their nodes and edges using current cutoff-safe evidence, and uses the
result as an incomplete check on the current forecast reasoning. Current-only
factors remain allowed. The answer-free worked trace reviews reasoning order,
counterevidence, and uncertainty. It does not provide a historical answer,
probability, or target estimate.

## Models and observed providers

| Model slug | Provider observed in raw calls |
|---|---|
| `google/gemini-2.5-flash-lite` | Google AI Studio |
| `openai/gpt-5-mini` | OpenAI |
| `deepseek/deepseek-v3.2` | Baidu |
| `meta-llama/llama-4-maverick` | DeepInfra |
| `minimax/minimax-m2.5` | Friendli |
| `qwen/qwen3-235b-a22b` | Alibaba |

## Final result roots

- Primary controlled run: `runs/procedural_topology_hgf_exemplar_check_normalized_v3_full100_20260802`
- DeepSeek recovery: `runs/procedural_topology_hgf_exemplar_check_normalized_v3_recovery_w20_deepseek_20260802`
- DeepSeek final recovery: `runs/procedural_topology_hgf_exemplar_check_normalized_v3_recovery2_w20_deepseek_20260802`
- MiniMax recovery: `runs/procedural_topology_hgf_exemplar_check_normalized_v3_recovery_w20_minimax_20260802`
- MiniMax final recovery: `runs/procedural_topology_hgf_exemplar_check_normalized_v3_recovery2_w20_minimax_20260802`
- Qwen final run: `runs/procedural_topology_hgf_exemplar_check_normalized_v3_qwen_full100_final_20260802`

DeepSeek combines 40 primary successes, 58 first-recovery successes, and 2
final-recovery successes. MiniMax combines 23 primary successes, 70
first-recovery successes, and 7 final-recovery successes. Deduplication is by
question ID in that fixed order. No successful case is replaced.

## Qwen frozen inputs

- Evidence manifest SHA256: `cf70faaeadf654a327a5d42e695a7bcbe12ae94868693320855a9ad3cef716dd`
- Retrieval manifest SHA256: `b74c5a821e5889387319151b308e1a8524956c7684f62c2a91ff5a027955937b`
- Input root: `runs/procedural_topology_hgf_exemplar_check_normalized_v3_qwen_inputs_assembled_20260802/model_evidence/qwen_qwen3-235b-a22b`

## Paper changes required

- Replace the v27-style method description with Procedural Topology HGF.
- Describe the answer-free worked trace as a reasoning completeness check, not
  as a source of the current answer.
- Report the controlled five-model results from `FINAL_RESULTS.md`.
- Do not claim that all providers enforced a hard ten-step output cap. Report
  observed reasoning length instead.
- Keep Qwen outside the pooled main comparison because its provider required
  native hidden reasoning to be disabled for reliable structured output.
- Disclose that Llama also ran without provider-native hidden reasoning. Do not
  describe all five endpoints as using an identical native reasoning setting.
- State that each model selected its own evidence from the frozen cutoff-safe
  candidate pool and then used frozen exact-family historical retrieval.
- Retain Accuracy, Brier score, and NLL as the registered performance metrics.
