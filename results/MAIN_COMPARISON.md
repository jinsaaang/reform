# Main comparison

All cells use the same fixed 100 questions per model and seed 0.

| Model | Method | Acc | Brier | NLL |
|---|---|---:|---:|---:|
| Gemini 2.5 Flash Lite | Direct Forecast | 0.500 | 0.2324 | 0.9913 |
| Gemini 2.5 Flash Lite | Structured Reasoning | 0.410 | 0.2589 | 1.0694 |
| Gemini 2.5 Flash Lite | Factor Memory | 0.440 | 0.2575 | 1.0652 |
| Gemini 2.5 Flash Lite | Principle Memory | 0.510 | 0.2501 | 1.0905 |
| Gemini 2.5 Flash Lite | Case Memory | 0.470 | 0.2532 | 1.0556 |
| Gemini 2.5 Flash Lite | Structure Memory | 0.520 | 0.2332 | 1.0011 |
| Gemini 2.5 Flash Lite | Procedural Topology HGF | 0.550 | 0.2131 | 0.9319 |
| GPT-5 mini | Direct Forecast | 0.480 | 0.2375 | 1.0126 |
| GPT-5 mini | Structured Reasoning | 0.420 | 0.2517 | 1.0520 |
| GPT-5 mini | Factor Memory | 0.480 | 0.2374 | 1.0058 |
| GPT-5 mini | Principle Memory | 0.520 | 0.2300 | 0.9700 |
| GPT-5 mini | Case Memory | 0.520 | 0.2311 | 0.9790 |
| GPT-5 mini | Structure Memory | 0.520 | 0.2215 | 0.9523 |
| GPT-5 mini | Procedural Topology HGF | 0.550 | 0.2158 | 0.9475 |
| DeepSeek V3.2 | Direct Forecast | 0.490 | 0.2356 | 1.0141 |
| DeepSeek V3.2 | Structured Reasoning | 0.500 | 0.2300 | 0.9745 |
| DeepSeek V3.2 | Factor Memory | 0.500 | 0.2252 | 0.9993 |
| DeepSeek V3.2 | Principle Memory | 0.510 | 0.2218 | 0.9711 |
| DeepSeek V3.2 | Case Memory | 0.520 | 0.2250 | 0.9724 |
| DeepSeek V3.2 | Structure Memory | 0.500 | 0.2237 | 0.9413 |
| DeepSeek V3.2 | Procedural Topology HGF | 0.520 | 0.2202 | 0.9129 |
| Llama 4 Maverick | Direct Forecast | 0.410 | 0.2454 | 1.0073 |
| Llama 4 Maverick | Structured Reasoning | 0.420 | 0.2513 | 1.0074 |
| Llama 4 Maverick | Factor Memory | 0.480 | 0.2240 | 0.9535 |
| Llama 4 Maverick | Principle Memory | 0.420 | 0.2415 | 0.9926 |
| Llama 4 Maverick | Case Memory | 0.420 | 0.2504 | 1.0152 |
| Llama 4 Maverick | Structure Memory | 0.420 | 0.2532 | 1.0237 |
| Llama 4 Maverick | Procedural Topology HGF | 0.500 | 0.2212 | 0.9302 |
| MiniMax M2.5 | Direct Forecast | 0.480 | 0.2382 | 0.9785 |
| MiniMax M2.5 | Structured Reasoning | 0.490 | 0.2278 | 0.9600 |
| MiniMax M2.5 | Factor Memory | 0.520 | 0.2284 | 0.9697 |
| MiniMax M2.5 | Principle Memory | 0.500 | 0.2268 | 0.9496 |
| MiniMax M2.5 | Case Memory | 0.530 | 0.2275 | 0.9948 |
| MiniMax M2.5 | Structure Memory | 0.510 | 0.2305 | 0.9664 |
| MiniMax M2.5 | Procedural Topology HGF | 0.530 | 0.2180 | 0.9203 |

## Pooled

| Method | N | Acc | Brier | NLL |
|---|---:|---:|---:|---:|
| Direct Forecast | 500 | 0.472 | 0.2378 | 1.0008 |
| Structured Reasoning | 500 | 0.448 | 0.2439 | 1.0126 |
| Factor Memory | 500 | 0.484 | 0.2345 | 0.9987 |
| Principle Memory | 500 | 0.492 | 0.2341 | 0.9948 |
| Case Memory | 500 | 0.492 | 0.2374 | 1.0034 |
| Structure Memory | 500 | 0.494 | 0.2324 | 0.9770 |
| Procedural Topology HGF | 500 | 0.530 | 0.2177 | 0.9286 |

HGF has the lowest Brier score for 5 of 5 models. Against the strongest pooled baseline, Structure Memory, its Brier is lower by 6.3% and its accuracy is higher by 3.6% in absolute terms.
