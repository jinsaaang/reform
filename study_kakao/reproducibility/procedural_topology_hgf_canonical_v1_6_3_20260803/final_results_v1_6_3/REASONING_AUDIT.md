# Reasoning contract audit

| Model | Valid | Avg steps | Avg evidence | Avg claims | DAG use | Multi-DAG | Counter step |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek/deepseek-v3.2 | 100/100 | 7.48 | 5.96 | 6.05 | 77 | 23 | 98 |
| google/gemini-2.5-flash-lite | 100/100 | 8.65 | 7.40 | 5.86 | 85 | 54 | 73 |
| meta-llama/llama-4-maverick | 100/100 | 5.72 | 4.17 | 3.12 | 100 | 55 | 100 |
| minimax/minimax-m2.5 | 100/100 | 7.38 | 6.34 | 3.83 | 97 | 44 | 88 |
| openai/gpt-5-mini | 100/100 | 8.00 | 8.27 | 6.70 | 83 | 35 | 94 |

Overall contract validity was 500/500.

This is a deterministic completeness and provenance audit. It is not an LLM-judge quality score.
