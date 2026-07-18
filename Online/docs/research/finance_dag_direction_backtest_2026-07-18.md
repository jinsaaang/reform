# Historical DAG direction backtest

## Objective

질문만 제공한 direct reasoning forecast와, 동일 질문에 cutoff 이전에 해결된 다른 금융 질문의 historical DAG top-3를 추가한 forecast를 비교한다. 단순히 확률이 변했는지가 아니라 실제 ground truth 방향으로 이동했는지와 Brier score가 개선됐는지를 측정한다.

## Design

- Target: WorldReasoner public DB에서 ground truth가 확정된 금융 이진 질문 10개
- Target cutoff: 각 질문 resolution timestamp의 7일 전
- Direct arm: question/context만 제공
- DAG arm: direct input + cutoff 이전에 해결된 historical DAG top-3
- Web evidence: 두 arm 모두 없음
- Forecast model: `openrouter/openai/o4-mini`
- Reasoning effort: `high`
- Model knowledge cutoff: 2024-06
- Initial seed: `20260718`
- Output: 동일한 strict scenario JSON schema와 weighted-probability validation

현재 GPT-5.6 모델을 사용한 첫 진단 실행은 2025년 target outcome을 model memory로 알고 있을 가능성이 있어 최종 판단에서 제외했다. 최종 표는 [OpenRouter가 2024-06 knowledge cutoff로 명시한 o4-mini](https://openrouter.ai/openai/o4-mini)를 사용했다.

## Results

`Improvement = Direct Brier − DAG Brier`이며 양수면 DAG arm이 정답 확률에 더 가깝다.

| # | Question | Truth | Direct Yes | DAG Yes | Direct Brier | DAG Brier | Direction |
|---:|---|:---:|---:|---:|---:|---:|:---:|
| 1 | U.S. recession during 2025 | No | 20.0% | 27.5% | 0.0400 | 0.0756 | Worse |
| 2 | GM Q4 EV charges exceed $5B | Yes | 25.0% | 31.0% | 0.5625 | 0.4761 | Better |
| 3 | Fed continues cutting in Jan 2026 | No | 60.0% | 38.0% | 0.3600 | 0.1444 | Better |
| 4 | Non-USD oil settlement share exceeds 25% | No | 34.0% | 38.0% | 0.1156 | 0.1444 | Worse |
| 5 | Domino's beats quarterly earnings | No | 59.0% | 60.0% | 0.3481 | 0.3600 | Worse |
| 6 | NYSE trading-floor opening delayed | No | 7.5% | 4.5% | 0.0056 | 0.0020 | Better |
| 7 | AMC beats quarterly earnings | Yes | 38.0% | 60.0% | 0.3844 | 0.1600 | Better |
| 8 | USD reaches 1.7M Iranian rials | Yes | 31.0% | 40.0% | 0.4761 | 0.3600 | Better |
| 9 | S&P 500 closes above 5,500 | Yes | 50.5% | 38.0% | 0.2450 | 0.3844 | Worse |
| 10 | Fed cuts at any point in 2025 | Yes | 34.0% | 36.0% | 0.4356 | 0.4096 | Better |

## Aggregate

| Metric | Direct | DAG | Change |
|---|---:|---:|---:|
| Mean Brier | 0.2973 | 0.2517 | −0.0456 |
| Relative mean-Brier reduction | — | — | 15.4% |
| Majority-label accuracy | 40% | 50% | +10%p |
| Questions moved toward ground truth | — | 6/10 | 60% |
| Questions moved away from ground truth | — | 4/10 | 40% |

Paired mean Brier improvement의 100,000-sample bootstrap 95% interval은 약 `[-0.0198, 0.1147]`이다. 0을 포함하므로 이 10-question pilot만으로 통계적으로 안정적인 개선이라고 주장할 수 없다.

## Majority flips

- Fed January 2026: direct는 잘못된 Yes 우세였으나 DAG arm은 올바른 No 우세로 전환했다.
- AMC earnings: direct는 잘못된 No 우세였으나 DAG arm은 올바른 Yes 우세로 전환했다.
- S&P 500: direct는 올바른 Yes 우세였으나 DAG arm은 잘못된 No 우세로 전환했다.

순효과는 majority-label 기준 `+1/10`이다.

## Retrieval diagnosis

방향성은 DAG 존재 여부보다 retrieval relevance에 크게 좌우됐다.

- Fed January target은 과거 Fed rate-cut DAG를 1순위로 가져왔고 Yes 60%를 38%로 낮춰 실제 No에 가까워졌다.
- AMC earnings target은 NVIDIA earnings DAG를 1순위로 가져왔고 Yes 38%를 60%로 올려 실제 Yes에 가까워졌다.
- S&P 500 target은 NVIDIA earnings, Instacart settlement, Dogecoin ETF DAG를 가져왔고 올바른 50.5%를 38%로 악화시켰다.
- Recession target도 earnings/settlement/prediction-market DAG를 가져오면서 No 정답에서 멀어졌다.

즉, 이번 pilot은 “DAG가 항상 개선한다”가 아니라 “관련 DAG는 유의미한 correction을 만들 수 있지만 현재 lexical retriever의 noise가 그 효과를 상쇄한다”는 결과에 가깝다.

## Validity limitations

- 표본이 10개뿐이며 단일 forecast draw다.
- public DB에는 candidate-to-target relation metadata가 없어 near-duplicate 및 shared-event 여부가 완전 검증되지 않는다.
- historical DAG는 episode resolution date로 eligibility를 판단했지만, DAG 내부 모든 문장의 생성 시점까지 frozen snapshot으로 증명되지는 않는다.
- 두 DAG arm은 첫 출력이 strict reference/schema validation을 통과하지 못해 다음 seed로 한 번 재시도했다. 따라서 완전히 동일 seed인 paired experiment는 아니다.
- 이번 실험은 DAG-only contribution 검사다. 현재 전체 시스템의 live-search contribution은 포함하지 않았다.

## Decision

현재 결과는 연구 방향을 중단할 신호는 아니다. 평균 Brier와 majority accuracy는 개선됐고, 관련 DAG가 선택된 Fed/AMC 사례에서는 정답 방향의 큰 correction이 나타났다. 그러나 6/10 correct shift와 0을 포함하는 confidence interval 때문에 성능 향상을 결론으로 쓰기에는 이르다.

다음 구현 우선순위는 forecasting prompt가 아니라 historical DAG retrieval이다.

1. entity, asset class, event type, horizon, resolution type을 사용하는 hybrid retriever
2. lexical-only retriever와 hybrid retriever의 paired ablation
3. 최소 50개 이상의 resolved binary targets와 반복 seed 평가
4. Brier와 별도로 blind LLM Judge를 통한 DAG relevance 및 scenario quality 평가
