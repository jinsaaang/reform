# Procedural Topology HGF full-100 results

All performance numbers below use the same fixed 100 questions per model. The table contains 3,500 predictions from five models and seven methods.

| Model | HGF Acc | HGF Brier | Best method | Best Brier |
|---|---:|---:|---|---:|
| Gemini 2.5 Flash Lite | 0.550 | 0.2122 | Procedural Topology HGF | 0.2122 |
| GPT 5 mini | 0.550 | 0.2112 | Procedural Topology HGF | 0.2112 |
| DeepSeek V3.2 | 0.500 | 0.2257 | Forecasting Principles | 0.2218 |
| Llama 4 Maverick | 0.530 | 0.2207 | Procedural Topology HGF | 0.2207 |
| MiniMax M2.5 | 0.510 | 0.2227 | Procedural Topology HGF | 0.2227 |

Pooled HGF over 500 predictions is 0.528 Accuracy, 0.2185 Brier, and 0.9337 NLL.

HGF is the lowest-Brier method on four of five models. DeepSeek is the exception. This is a single registered seed and is not a multi-seed robustness claim.
