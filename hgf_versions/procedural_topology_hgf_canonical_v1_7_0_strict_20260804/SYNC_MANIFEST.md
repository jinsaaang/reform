# Paper and experiment sync manifest

## Repository state

- Branch: `experiment/live-topology-hgf-v1`
- Base commit before this archive: `dbf76bce37955e7ef5f287e70b7dc56556446971`
- Canonical archive commit: `TO_BE_RECORDED_AFTER_ARCHIVE_COMMIT`
- No files were merged or copied into `/home/kong/code/study_kakao`.

## Canonical experiment contract

- Main method: Procedural Topology HGF
- Models: `google/gemini-2.5-flash-lite`, `openai/gpt-5-mini`,
  `deepseek/deepseek-v3.2`, `meta-llama/llama-4-maverick`, and
  `minimax/minimax-m2.5`
- Baselines: Direct Forecasting, DAG Forecasting, Direct DAG Retrieval, Factor
  Memory, Resolved Case, and Forecasting Principles
- Questions: the same frozen 100-question selection for every cell
- Seeds: 0, 1, and 2
- Seed 0: preserved registered v1.6.3 outputs
- Seeds 1 and 2: v1.6.0-strict execution with rejected invalid generations
  retried on the same model, prompt, input, and seed
- Workers: 20 per model
- Metrics: Accuracy, Brier score, and NLL
- Saved auxiliary measurements: reasoning, cited and prediction-used evidence,
  prompt and completion tokens, API calls, billed cost, elapsed time, provider,
  returned model, and raw-call provenance
- Selection rule for technical continuations: first truth-free
  forecast-contract-valid execution in declared root order. Scores and labels
  are not read during selection.

## Provider record

- Gemini baseline and HGF runs use Google AI Studio.
- GPT-5 mini baseline and HGF runs use OpenAI.
- DeepSeek baseline and HGF runs use Atlas Cloud.
- Llama baseline uses DeepInfra Base and HGF uses DeepInfra. Native reasoning
  is disabled for this model.
- MiniMax baselines use Friendli. The first HGF partial execution uses
  Inceptron. Unfinished technical cases continue under OpenRouter automatic
  latency routing with compatible-provider fallback.
- MiniMax successful rows are preserved across continuations. Early successful
  rows use the 32,768 output cap and later missing-case continuations use the
  16,000 cap. No successful row is rerun or selected by score.

## Completion audit

- MiniMax HGF seeds 1 and 2: 200 of 200 rows passed.
- All-model HGF seeds 1 and 2: 1,000 of 1,000 rows passed.
- Six baselines plus HGF for seeds 1 and 2: 7,000 of 7,000 rows passed.
- Every audited row contains prediction probabilities, substantive reasoning,
  prediction-used evidence, metrics, token usage, elapsed time, raw-call
  provenance, and worker count 20.
- Full three-seed paper summary: 10,500 forecasts, comprising 5 models, 7
  methods, 100 questions, and 3 seeds.

## Paper updates required

- Replace the v27 method description with Procedural Topology HGF.
- Report five models and the six baselines listed above.
- State that each model independently uses its frozen model-specific evidence
  selection and historical retrieval manifest.
- State 100 questions and seeds 0, 1, and 2. Report mean and standard deviation
  across seeds.
- Report Accuracy, Brier score, and NLL in that order in prose, while retaining
  the table direction indicators.
- Report provider routing and reasoning settings in the reproducibility
  appendix rather than treating provider as a method variable.
- Do not claim that HGF wins on every model. It has the best pooled results and
  the lowest Brier on four of five models. MiniMax's best Brier is obtained by
  Resolved Case.

## Frozen files and hashes

- `data/questions/selection.json`: `13e18cfd89300e819c1dd0a35450caa1a3cd7fa8dededa1b6c83cfffbff7eba9`
- `config.json`: `3a66c818b263fac5d243f8e56af322d89ef468cad6ab7846e04a0a9a69290178`
- `minimax_multiseed_campaign_manifest.json`: `6f6c0d35110a8a2f0314156ecd2467005d8f603ce0b6bc739879df39e24db257`
- Final run manifest: `f277ebd7bff6bc98fe7876e78050c6bd9325eb394eaa791f5b9e88c686056a0e`
- Three-seed summary: `f80dbde0c684c1aeeff1b2f359420533e5e5c3535c382550d68b0317a168a8e4`
- Per-seed metric and resource summary: `2d9cf871d6e366ea3a5ef70f140e17822a23ea31c24a9e0807595693bab2cbe8`
- Full 7,000-row audit: `65d483aeba77ceade6648780b5f384615c4ed5869ca2802f21b33386774d0478`
- HGF 1,000-row audit: `c2c05ca3bf104807b925082df1e4201f93c803cd895a1a0d8d4c89938b21752e`

## Included archive scope

- Frozen HGF method source and historical base source
- Execution adapter and provider wrapper
- Config, launcher, recovery, validation, finalization, and contract tests
- Five model-specific evidence-selection and retrieval manifests
- MiniMax continuation manifest
- Final run manifest, per-seed metrics and resource summary, three-seed summary,
  and completeness audits
- Human-readable final result summary

The full raw results are retained at
`runs/procedural_topology_hgf_all_methods_multiseed_final_20260803` and the
MiniMax finalized HGF subset is retained at
`runs/procedural_topology_hgf_v1_6_0_strict_minimax_final_20260803` in this
worktree. The archive commit intentionally contains summaries and audit records
rather than the 351 MB full result payload.
