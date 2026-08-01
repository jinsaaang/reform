# Procedural Topology HGF

이 문서는 현재 논문과 실험에서 사용하는 HGF 방법론의 유일한 기준 문서다. 이전 개발 버전과 진단 기록은 `legacy/experimental_variants`에 보존하며 현재 방법으로 간주하지 않는다.

## 연구 목적

반복되는 금융 이벤트는 예측 대상과 작동 구조가 비슷하지만 매 시점의 증거와 시장 상태는 달라진다. HGF는 과거 사건의 정답이나 확률을 현재 답의 prior로 사용하지 않는다. 해결된 과거 사건에서 얻은 인과 구조를 가져와 현재 증거로 각 관계를 다시 확인함으로써 LLM의 예측 추론을 개선한다.

핵심 가설은 과거 사건의 가치가 정답 자체보다 증거를 어떤 요인과 관계로 해석해야 하는지를 보여 주는 구조에 있다는 것이다.

## Offline graph bank

각 과거 금융 이벤트에는 WorldReasoner로 구축한 hindsight DAG가 있다. DAG는 예측 요인인 node와 이들 사이의 방향을 가진 edge, lag, support, confidence, root to target path를 포함한다. 현재 예측에는 forecast cutoff 전에 해결된 같은 event family와 target metric의 DAG만 사용할 수 있다.

이 graph bank는 데이터와 같은 고정 연구 산출물이다. Forecaster model마다 다시 만들지 않는다. 과거 정답, 과거 확률, 실현값과 사후 결론은 live forecaster 입력에서 제외한다. Graph의 node와 edge topology는 유지하며 과거 상태를 현재 사실처럼 전달하지 않는다.

## End to end forecasting pipeline

Procedural Topology HGF는 하나의 예측 흐름으로 작동한다.

1. 현재 질문의 metric, period, unit, comparison rule과 answer boundary를 target contract로 고정한다.
2. 해당 모델이 cutoff safe evidence 후보에서 현재 질문에 필요한 증거를 독립적으로 선택한다.
3. 같은 event family와 target metric에 속하고 시간상 사용 가능한 hindsight DAG를 검색한다.
4. 검색된 여러 DAG에서 현재 증거와 관련된 complete subgraph를 선택한다. Node, edge, 방향, 관계, lag와 path 순서는 보존한다.
5. 선택한 subgraph의 node와 edge를 현재 evidence로 다시 확인한다. 각 node에는 current state와 evidence를, 각 edge에는 현재 관계가 유지되는지, 반전되는지, 반박되는지 또는 아직 확인되지 않는지를 기록한다.
6. 현재 상태가 채워진 graph를 따라 baseline, driver, mechanism, counterevidence와 target bridge를 작성한다. 현재 증거가 지지하지 않는 path는 기각하거나 uncertainty로 남긴다.
7. 별도 boundary mapper가 완성된 reasoning을 target unit과 answer boundary에 맞춰 하나의 probability distribution으로 변환한다.

이 exact canonical 구현은 historical worked exemplar를 사용하지 않는다. 성능을 위해 baseline 답을 참조하거나 확률을 사후 조정하지도 않는다.

## 정보 흐름과 실행 계약

- 현재 evidence와 과거 DAG는 모두 forecast cutoff를 지킨다.
- 다른 방법의 prediction이나 probability는 HGF 입력에 들어가지 않는다.
- 과거 topology는 현재 사실의 근거로 인용할 수 없다.
- 현재 factual claim은 현재 evidence ID와 연결되어야 한다.
- Reasoning 단계는 최종 probability를 출력하지 않는다.
- Probability는 마지막 boundary 단계에서 한 번만 생성한다.
- Probability pooling, posterior adjustment와 result conditioned retry를 금지한다.
- API 또는 parsing 실패만 같은 model, provider, seed와 입력으로 다시 호출한다. 이미 성공한 출력을 점수에 따라 다시 생성하거나 선택하지 않는다.

## Baselines

한 모델 안에서는 모든 방법이 같은 100개 질문, 같은 model specific current evidence와 같은 historical retrieval manifest에서 시작한다.

| Method | 입력과 역할 |
|---|---|
| Structured Direct Forecasting | 과거 정보 없이 현재 evidence로 target semantics, baseline, driver, mechanism과 counterevidence를 작성하는 강한 통제군 |
| DAG Forecasting | 현재 evidence만으로 prospective DAG를 구축하고 예측 |
| Outcome Neutral Direct DAG Retrieval | 검색된 과거 DAG에서 과거 값과 실현 방향 및 episode specific conclusion을 제거한 뒤 graph를 직접 전달 |
| Factor Memory | 이전 registered memory bank의 주요 예측 요인을 전달하는 강한 expert factor baseline |
| Outcome Redacted Case Retrieval | 과거 질문과 cutoff time evidence를 제공하되 resolved option, realized value와 post resolution rationale는 제거 |
| Forecasting Principles | 해결된 과거 사건에서 추출한 일반 예측 원칙을 전달 |
| Procedural Topology HGF | 과거 subgraph의 구조를 현재 evidence로 다시 채우고 활성 경로와 경쟁 설명을 따라 예측 |

Factor Memory는 현재 HGF graph에서 edge만 제거한 topology matched ablation이 아니다. 정보 출처와 정제 방식이 다른 독립적인 강한 baseline으로 해석한다.

## Canonical full 100 experiment

최종 성능 판단에는 모델별로 고정된 100문항만 사용한다. 40문항 subset과 중간 결과는 진단용이며 논문의 성능 주장에 사용하지 않는다. Seed는 0 한 번이며 다중 seed 안정성을 주장하지 않는다.

| Model | HGF Accuracy | HGF Brier | HGF NLL | Lowest Brier method |
|---|---:|---:|---:|---|
| Gemini 2.5 Flash Lite | 0.550 | 0.2122 | 0.9078 | HGF 0.2122 |
| GPT 5 mini | 0.550 | 0.2112 | 0.9068 | HGF 0.2112 |
| DeepSeek V3.2 | 0.500 | 0.2257 | 0.9532 | Forecasting Principles 0.2218 |
| Llama 4 Maverick | 0.530 | 0.2207 | 0.9302 | HGF 0.2207 |
| MiniMax M2.5 | 0.510 | 0.2227 | 0.9708 | HGF 0.2227 |

HGF의 500개 pooled 결과는 Accuracy 0.528, Brier 0.2185, NLL 0.9337이다. 일곱 방법 중 네 모델에서 HGF의 Brier가 가장 낮고 DeepSeek에서는 Forecasting Principles가 더 낮다.

Leakage를 제거한 두 history baseline의 pooled 결과는 다음과 같다.

| Method | Accuracy | Brier | NLL |
|---|---:|---:|---:|
| Outcome Redacted Case Retrieval | 0.492 | 0.2374 | 1.0034 |
| Outcome Neutral Direct DAG Retrieval | 0.494 | 0.2324 | 0.9770 |
| Procedural Topology HGF | 0.528 | 0.2185 | 0.9337 |

문항 ID를 cluster로 사용한 10,000회 paired bootstrap에서 HGF minus Case Retrieval의 Brier 차이는 -0.0190이고 95 percent interval은 [-0.0314, -0.0070]이다. HGF minus Direct DAG Retrieval은 -0.0139이고 interval은 [-0.0273, -0.0002]다.

## Reasoning and execution quality

500개 HGF 출력 모두 written reasoning step과 실제 prediction pipeline이 사용한 evidence 기록을 가진다. Strict structured contract 통과 수는 Gemini 86, GPT 95, DeepSeek 95, Llama 65, MiniMax 59다. Strict 실패는 reasoning 부재와 같지 않으며 boundary fallback, incomplete path contract와 graph default 사용을 별도 항목으로 보고해야 한다. 특히 Llama와 MiniMax는 성능과 함께 낮은 strict compliance를 limitation으로 명시한다.

모든 raw request와 response, provider, prediction, probability, evidence ID, reasoning, token, cost, elapsed time, repair와 transport retry가 canonical run directory에 보존되어 있다.

## Reproduction and paper synchronization

재현 번들은 `reproducibility/procedural_topology_hgf_full100_20260802`에 있다. 이 번들은 exact HGF method source, historical shared dependencies, input adapter, 두 sanitized baseline, five model specific evidence와 retrieval manifest, neutral topology cache와 canonical provenance를 포함한다.

결과 표와 논문 동기화 정보는 `experiments/final_results_20260802`에 있다. 논문의 방법론은 이전 v27 설명이 아니라 이 문서의 Procedural Topology HGF 흐름으로 수정해야 한다. 모델, provider, 100문항, seed 0, metric, baseline 이름과 제한도 sync manifest를 따른다.
