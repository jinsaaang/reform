# Canonical Procedural Topology HGF Results

| Model | N | Acc | Brier | NLL | DAG path use | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek V3.2 | 100 | 0.520 | 0.2202 | 0.9129 | 77.0% | 27 |
| Gemini 2.5 Flash Lite | 100 | 0.550 | 0.2131 | 0.9319 | 85.0% | 7 |
| Llama 4 Maverick | 100 | 0.500 | 0.2212 | 0.9302 | 100.0% | 4 |
| MiniMax M2.5 | 100 | 0.530 | 0.2180 | 0.9203 | 97.0% | 43 |
| GPT-5 mini | 100 | 0.550 | 0.2158 | 0.9475 | 83.0% | 1 |

Pooled N = 500, Acc = 0.530, Brier = 0.2177, NLL = 0.9286.

Recorded campaign usage was 31,686,903 tokens across 2,576 raw calls, with observed cost $15.6311.

Recovery denotes cases repeated solely because the original execution failed the provider or output contract. No result was selected by its forecast score.
