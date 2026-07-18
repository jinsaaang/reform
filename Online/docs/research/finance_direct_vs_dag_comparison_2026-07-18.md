# Direct LLM vs. evidence + historical DAG pilot

## Question

금융 질문과 해결 조건만 reasoning LLM에 직접 전달한 forecast와, 날짜 필터링된 live evidence 및 유사 과거 질문의 WorldReasoner DAG를 함께 전달한 forecast가 실제로 다른가?

## Compared conditions

### A. Direct LLM baseline

- Input: question, context, cutoff, binary outcome space
- Live evidence: none
- Historical DAG reference: none
- Model: `openrouter/openai/gpt-5.6-sol`
- Reasoning effort: `high`
- Output: proposed condition과 동일한 strict scenario/forecast schema

두 조건은 같은 실험일에 순차 실행했지만 동일 inference seed를 고정하지 않았고, direct run은 proposed run 직후에 수행됐다. 따라서 아래 수치는 차이의 존재를 확인하는 pilot이지, 통제된 평균 treatment effect 추정치는 아니다.

### C. Proposed pipeline

- Input: 동일 question, context, cutoff, binary outcome space
- Search: cutoff 이전 live evidence에 대한 initial search + DAG-guided search
- Historical memory: immutable WorldReasoner finance DB에서 top-3 DAG
- Forecaster는 검색 또는 DAG 생성 권한 없이 filtered evidence와 DAG reference만 사용
- Model과 reasoning effort 및 output schema는 direct baseline과 동일

## Probability comparison

`Δ`는 `Proposed Yes − Direct Yes`다.

| # | Question | Direct Yes | Proposed Yes | Δ | Majority flip |
|---:|---|---:|---:|---:|:---:|
| 1 | Fed cuts by 2026-12-31 | 59.75% | 33.20% | −26.55%p | Yes → No |
| 2 | S&P 500 closes at or above 7,000 | 46.70% | 62.00% | +15.30%p | No → Yes |
| 3 | Bitcoin closes at or above $150K | 32.25% | 11.71% | −20.55%p | — |
| 4 | Gold closes at or above $4,000 | 39.00% | 54.84% | +15.84%p | No → Yes |
| 5 | Brent closes at or above $100 | 23.25% | 20.50% | −2.75%p | — |
| 6 | NVIDIA ends 2026 above $5T market cap | 46.00% | 36.50% | −9.50%p | — |
| 7 | December 2026 U.S. CPI is below 3% | 58.05% | 52.00% | −6.05%p | — |
| 8 | Bank of England cuts by 2026-12-31 | 61.50% | 57.35% | −4.15%p | — |
| 9 | ECB cuts by 2026-12-31 | 49.00% | 42.00% | −7.00%p | — |
| 10 | Ethereum closes above $10K | 27.00% | 6.28% | −20.73%p | — |

Aggregate observations:

- Mean absolute Yes-probability change: **12.84 percentage points**
- Majority outcome changed: **3/10 questions**
- Proposed probability was lower than direct baseline: **8/10 questions**
- Largest changes: Fed −26.55%p, Ethereum −20.73%p, Bitcoin −20.55%p

## Reasoning difference

Direct baseline의 explanation은 반복적으로 다음 한계를 스스로 명시했다.

- evidence pack과 historical memory가 비어 있어 prior-driven forecast임
- 현재 가격, 추세, valuation, macro data를 알 수 없어 broad conditional scenario에 의존함
- 구체적인 current fact보다 일반적인 bullish/base/bearish path를 사용함

Proposed pipeline은 반대로 다음과 같은 cutoff 이전 관측치를 reasoning에 사용했다.

- Bitcoin: 약 $60K–$61.5K에서 $150K까지 필요한 상승 배수
- Gold: 2025-11 이후 처음 $4,000 아래로 내려왔다는 관측
- Brent: 수요 및 OECD inventory draw 전망 하향
- NVIDIA: 최근 약 $1T valuation drawdown과 부분 회복
- CPI: gasoline-driven disinflation과 sticky household costs
- ECB: 최근 tightening pivot과 inflation shock
- Ethereum: 약 $1,735에서 $10K까지 필요한 상승 배수

따라서 direct forecast와 proposed forecast는 단순히 표현만 다른 것이 아니다. Proposed forecaster는 cutoff 이전 evidence로 current state를 anchoring하고, 그 상태에서 scenario probability를 다시 배분했다. Direct baseline은 이 anchoring이 없어 모델의 일반 prior에 더 크게 의존했다.

Grounding 차이도 구조적으로 확인된다.

| Measure | Direct | Proposed |
|---|---:|---:|
| Admitted evidence items | 0 | 25 |
| Unique evidence URLs | 0 | 19 |
| Historical DAG references | 0 | 30 (3 per question) |
| Valid scenario forecasts | 10/10 | 10/10 |

## What can and cannot be claimed

이번 pilot으로 주장할 수 있는 것은 다음과 같다.

1. 검색 근거와 historical DAG를 추가하면 동일 reasoning model의 확률과 scenario reasoning이 실질적으로 달라진다.
2. Proposed condition은 current fact에 근거한 scenario를 생성하지만, direct baseline은 evidence가 없음을 밝히고 generic prior에 의존한다.
3. 차이는 평균 12.84%p이며 단순 rounding 수준이 아니다.

아직 주장할 수 없는 것은 다음과 같다.

1. Proposed forecast가 더 정확하다는 주장: 질문들이 미해결 상태여서 ground truth와 Brier score가 없다.
2. 차이가 DAG 때문에 발생했다는 단독 인과 주장: 현재 A-vs-C 비교는 live search와 DAG 효과가 합쳐져 있다.
3. 안정적인 평균 효과라는 주장: 각 조건을 한 번씩 실행했으며 sampling variance를 추정하지 않았다.

## Required three-arm ablation

DAG의 독립적 기여를 확인하려면 최종 실험을 다음 세 조건으로 고정해야 한다.

| Arm | Question/context | Filtered live evidence | Historical DAG |
|---|:---:|:---:|:---:|
| A. Direct | ✓ | — | — |
| B. Search-only | ✓ | ✓ | — |
| C. Proposed | ✓ | ✓ | ✓ |

- `B − A`: 검색 및 current-state anchoring의 효과
- `C − B`: historical DAG가 scenario reasoning에 추가하는 효과
- `C − A`: 전체 proposed pipeline의 효과

모든 arm에서 동일 question manifest, cutoff, evidence snapshot, model, output schema를 사용해야 한다. 각 조건을 반복 실행하고 paired blind LLM Judge로 evidence faithfulness, scenario coverage, mutual exclusivity, causal coherence, domain plausibility, DAG relevance를 평가한다. 질문이 해결된 뒤에는 accuracy와 Brier score를 추가한다.

## Pilot conclusion

Direct LLM 대비 차이는 분명히 확인됐다. 현재 결과에서 가장 강한 차이는 **prior-driven generic forecast가 cutoff-aware evidence-grounded forecast로 바뀐 것**이다. 다만 연구의 핵심인 DAG의 추가 가치를 입증하려면 다음 단계에서 반드시 search-only arm을 추가해야 한다.
