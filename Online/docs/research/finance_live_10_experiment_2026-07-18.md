# Financial WorldReasoner live forecast pilot — 10 questions

## Summary

2026-07-18 KST에 `offline finance DAG memory → two-pass live search → reasoning forecaster` 전체 경로로 금융 이진 질문 10개를 실행했다. 10개 모두 검색 근거와 과거 DAG 참조를 입력받아 시나리오 기반 확률을 생성했으며, 최종 출력은 스키마 검증과 확률 일관성 검증을 통과했다.

이 결과는 파이프라인 연결을 확인하기 위한 current-mode pilot이며, 아직 예측 성능이나 투자 유효성을 입증하는 backtest가 아니다.

## Execution configuration

- Run mode: `current_unresolved`
- Evidence cutoff: 각 실행 시작 시점, `2026-07-17T17:15:39Z`–`17:41:28Z`
- Offline memory: immutable WorldReasoner public DB의 finance DAG episode 37개
- Historical retrieval: 질문과 context를 반영한 top-3 DAG retrieval
- Search: live Bing News RSS, initial pass와 DAG-guided pass, pass당 최대 2개 후보
- Temporal rule: `available_at < cutoff`인 근거만 admission
- Forecaster: `openrouter/openai/gpt-5.6-sol`, reasoning effort `high`
- Forecaster boundary: 검색 도구와 DAG 생성 권한 없이, 필터링된 evidence와 검색 단계가 전달한 historical DAG reference만 사용
- Output constraints: strict JSON schema; scenario weights와 각 conditional outcome은 합이 1; 최종 확률은 scenario-weighted sum과 정확히 일치
- OpenRouter references: [reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), [model metadata API](https://openrouter.ai/api/v1/models)

## Results

`Evidence`는 admission된 item 수와 서로 다른 citation URL 수를 함께 표시한다. 동일 기사가 initial/guided pass에서 중복 수집된 경우 두 값이 다르다.

| # | Exact question | Yes | No | Evidence (items/unique URLs) | DAGs | Highest-weight scenario |
|---:|---|---:|---:|---:|---:|---|
| 1 | Will the U.S. Federal Reserve lower its target federal funds range at least once between July 18 and December 31, 2026? | 33.20% | 66.80% | 3/3 | 3 | Inflation remains elevated; the Fed holds or raises rates (60%) |
| 2 | Will the S&P 500 index close at or above 7,000 on December 31, 2026? | 62.00% | 38.00% | 3/2 | 3 | Earnings-led advance finishes above 7,000 (45%) |
| 3 | Will Bitcoin close at or above 150,000 U.S. dollars on December 31, 2026? | 11.71% | 88.29% | 4/2 | 3 | Substantial rebound that remains below $150,000 (49%) |
| 4 | Will spot gold close at or above 4,000 U.S. dollars per troy ounce on December 31, 2026? | 54.84% | 45.16% | 2/2 | 3 | Gold recovers and finishes above the threshold (40%) |
| 5 | Will Brent crude oil close at or above 100 U.S. dollars per barrel on December 31, 2026? | 20.50% | 79.50% | 2/1 | 3 | Weak demand and limited inventory draws keep Brent below $100 (60%) |
| 6 | Will NVIDIA market cap exceed 5.0 trillion dollars at the end of 2026? | 36.50% | 63.50% | 3/2 | 3 | Partial recovery but year-end capitalization remains below $5 trillion (45%) |
| 7 | Will U.S. CPI be below 3.0 percent in December 2026? | 52.00% | 48.00% | 3/2 | 3 | Energy-led disinflation continues (50%) |
| 8 | Will the Bank of England cut Bank Rate before the end of 2026? | 57.35% | 42.65% | 1/1 | 3 | Gradual economic softening permits one or more cuts (47%) |
| 9 | Will the European Central Bank lower its deposit facility rate at least once between July 18 and December 31, 2026? | 42.00% | 58.00% | 1/1 | 3 | War-related inflation persists and the ECB maintains or extends tightening (50%) |
| 10 | Will Ethereum close above 10,000 U.S. dollars on December 31, 2026? | 6.28% | 93.72% | 3/3 | 3 | Weak market persists and ETH remains far below the threshold (65%) |

## Historical DAG references

각 행은 forecaster에 실제 전달된 과거 question ID다. Forecaster가 DAG를 새로 만들지는 않는다.

| # | Selected historical question IDs |
|---:|---|
| 1 | `polymarket_0xfa48...`, `q_finance_20251231_007_ea7430e9`, `q_finance_20260108_001_18a819d3` |
| 2 | `q_finance_20260108_001_18a819d3`, `q_finance_20251231_006_02162c5a`, `q_finance_20251231_015_8dcf1b5a` |
| 3 | `polymarket_0xc3c8...`, `polymarket_0x13c8...`, `polymarket_0x50ee...` |
| 4 | `polymarket_0xc3c8...`, `polymarket_0x13c8...`, `polymarket_0x50ee...` |
| 5 | `polymarket_0xc3c8...`, `polymarket_0x13c8...`, `polymarket_0xfa48...` |
| 6 | `polymarket_0x13c8...`, `polymarket_0x50ee...`, `q_finance_20260108_001_18a819d3` |
| 7 | `q_finance_20251231_007_ea7430e9`, `polymarket_0xc3c8...`, `q_finance_20260131_040_4a5b566f` |
| 8 | `polymarket_0xc3c8...`, `q_finance_20251231_007_ea7430e9`, `polymarket_0xfa48...` |
| 9 | `polymarket_0xc3c8...`, `polymarket_0xfa48...`, `polymarket_0x9fb9...` |
| 10 | `polymarket_0xc3c8...`, `q_finance_20251231_007_ea7430e9`, `polymarket_0x13c8...` |

## What this pilot establishes

1. Searcher와 forecaster의 역할 분리가 실제 실행에서 유지된다. Searcher가 cutoff 이전 live evidence를 수집·필터링하고, forecaster는 그 결과와 과거 DAG 참조만 소비한다.
2. Offline DAG는 새 예측의 답을 복사하는 용도가 아니라 guided search의 mechanism hint와 시나리오 구조를 제공하는 memory로 전달된다.
3. Reasoning forecaster는 세 개의 상호 구분되는 시나리오, scenario weight, scenario별 conditional outcome, 최종 확률, 자연어 설명을 생성했다.
4. 10개 모두 `initial_search → immutable_episode_load → historical_retrieval → guided_search → evidence_admission → forecaster` 순서를 완료했다.

## Main limitations found

- 현재 DAG retriever는 작은 37-question DB 위에서 lexical overlap의 영향이 크다. 예를 들어 gold·Brent 질문에도 NVIDIA/AMC earnings DAG가 선택됐다. 다음 실험의 가장 큰 개선 지점은 entity, asset class, horizon, resolution type을 함께 쓰는 hybrid retrieval이다.
- RSS 결과는 일부 질문에서 동일 기사를 여러 pass로 중복 수집했고, MSN 같은 재배포 기사 비중이 높다. canonical URL 기준 deduplication과 official/primary-source 우선순위가 필요하다.
- Bank of England와 ECB 질문은 unique evidence가 한 건뿐이므로 확률의 근거 폭이 좁다.
- 이번 실행은 현재 시점 질문만 다뤘다. Frozen retrospective DB가 없으므로 historical backtest 및 leakage-free accuracy/Brier 평가는 포함하지 않았다.
- LLM Judge 평가는 이번 빠른 연결 pilot에는 적용하지 않았다. 후속 평가에서는 evidence entailment, scenario coverage, scenario mutual exclusivity, DAG relevance, probability coherence를 Judge rubric으로 측정해야 한다.

## Immediate next experiment

동일 10-question manifest를 고정한 뒤 다음 두 설정을 비교하는 것이 가장 빠르다.

- Baseline: admitted live evidence만 제공
- Proposed: admitted live evidence + top-k historical DAG references 제공

두 설정 모두 동일한 reasoning forecaster와 검색 cutoff를 사용하고, accuracy/Brier 외에 LLM Judge 기반 reasoning quality와 DAG relevance를 함께 기록한다. 현재 pilot에서 드러난 retrieval noise를 분리해 보기 위해 `lexical top-k`와 `entity/asset/horizon-aware hybrid top-k`도 별도 ablation으로 두는 것이 좋다.
