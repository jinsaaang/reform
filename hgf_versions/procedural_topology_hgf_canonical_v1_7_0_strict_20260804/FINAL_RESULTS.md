# Final multi-seed results

The final comparison uses five models, seven methods, 100 questions, and seeds
0, 1, and 2. Each model-method cell therefore contains 300 forecasts. The table
below compares Procedural Topology HGF with the lowest-Brier baseline for each
model.

| Model | Best baseline | Baseline Acc | Baseline Brier | HGF Acc | HGF Brier | HGF NLL | Relative Brier change |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash Lite | Direct DAG Retrieval | 0.520 | 0.2332 | 0.550 | 0.2121 | 0.9268 | -9.1% |
| GPT-5 mini | Forecasting Principles | 0.513 | 0.2264 | 0.547 | 0.2158 | 0.9457 | -4.7% |
| DeepSeek V3.2 | Forecasting Principles | 0.520 | 0.2254 | 0.523 | 0.2202 | 0.9326 | -2.3% |
| Llama 4 Maverick | Direct Forecasting | 0.433 | 0.2382 | 0.517 | 0.2239 | 0.9544 | -6.0% |
| MiniMax M2.5 | Resolved Case | 0.520 | 0.2217 | 0.510 | 0.2229 | 0.9390 | +0.6% |

Across all five models and three seeds, HGF obtains 0.5293 Accuracy, 0.2190
Brier, and 0.9397 NLL. The strongest pooled baseline by Brier is Direct DAG
Retrieval at 0.4973 Accuracy, 0.2312 Brier, and 0.9795 NLL. HGF reduces pooled
Brier by 5.3 percent and improves Accuracy by 3.2 percentage points. MiniMax is
the only model where HGF does not obtain the lowest mean Brier.

The machine-readable per-seed metrics, three-seed summary, run manifest, and
completeness audits are archived with this bundle's sync commit. The full raw
row and call-provenance artifacts remain in the immutable run roots identified
in `SYNC_MANIFEST.md`.
