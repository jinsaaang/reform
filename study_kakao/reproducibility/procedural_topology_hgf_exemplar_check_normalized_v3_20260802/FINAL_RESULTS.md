# Final v3 HGF results

All five registered models use the same fixed set of 100 financial forecasting
questions, run seed 0, model-specific frozen evidence, and frozen exact-family
historical retrieval. The explicit HGF reasoning stage is common to every
model. Provider-native reasoning was requested at medium where the endpoint
supported it and was disabled for Llama. Failed transport attempts were
recovered only for missing question IDs. A successful prediction was never
resampled or selected by score.

| Model | Acc | Brier | NLL | Mean reasoning steps | DAG-used cases | Tokens | Observed cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash Lite | 0.560 | 0.2056 | 0.8760 | 8.65 | 85 | 7,088,829 | $1.6887 |
| GPT-5 mini | 0.540 | 0.2135 | 0.9284 | 8.00 | 84 | 4,493,705 | $3.5437 |
| DeepSeek V3.2 | 0.560 | 0.2087 | 0.8875 | 7.28 | 75 | 5,249,095 | $1.2543 |
| Llama 4 Maverick | 0.480 | 0.2320 | 0.9845 | 5.87 | 100 | 2,937,409 | $0.8040 |
| MiniMax M2.5 | 0.510 | 0.2181 | 0.9231 | 5.68 | 63 | 5,642,841 | $4.4164 |

The registered five-model pooled result is 0.530 Accuracy, 0.2156 Brier, and
0.9199 NLL over 500 predictions. The registered canonical Procedural Topology
HGF result was 0.528 Accuracy, 0.2185 Brier, and 0.9337 NLL. The normalized v3
variant improves the pooled result, but the model-level effect is heterogeneous.
It improves Brier for Gemini, DeepSeek, and MiniMax, is nearly neutral for GPT,
and degrades Llama.

## Exploratory Qwen transfer

| Model | Acc | Brier | NLL | Mean reasoning steps | DAG-used cases | Tokens | Observed cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 235B A22B | 0.430 | 0.2370 | 0.9895 | 3.90 | 7 | 4,716,413 | $3.0234 |

Qwen completed all 100 questions with no terminal failure. Its Alibaba endpoint
required response-shape recovery on 97 questions and produced 622 billed calls.
Native hidden reasoning was disabled because it did not reliably compose with
the structured response contract. The HGF reasoning stage remained explicit,
and no prediction or probability was altered by the compatibility adapter.
Because this transport policy differs from the controlled five-model setup,
Qwen should be reported as an exploratory portability result rather than pooled
into the main table.

## Audit facts

- Every final case used exactly one probability-producing boundary call.
- No final case received another method's prediction or probability.
- Probability pooling and posterior adjustment were absent in all final cases.
- Gemini returned 11 to 14 steps in ten cases despite the schema requesting a
  maximum of ten. These traces were preserved rather than truncated after the
  prediction.
- Qwen used a schema compatibility layer that moved model-produced content into
  canonical node, edge, path, reasoning, and boundary fields. Its audit records
  mark `probability_modified` as false.
- Llama and Qwen used the same explicit HGF reasoning procedure but no
  provider-native hidden reasoning. This execution difference must be disclosed
  when comparing model families.
