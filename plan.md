# Reasoning Harness Optimization for Financial Forecasting

## 1. 핵심 메시지

LLM forecaster의 한계는 좋은 정보를 가져오지 못하는 것만이 아니라, 가져온 정보와 과거 경험을 바탕으로 금융적으로 타당한 사고 절차를 밟지 못한다는 데 있다.

본 연구는 과거 resolved financial episode를 정답 예시로 직접 제공하는 대신, 새로운 질문을 풀 때 무엇을 확인하고, 어떤 시나리오를 고려하며, 어떤 reasoning failure를 피해야 하는지 알려주는 reasoning harness로 변환한다.

핵심 claim은 다음과 같다.

좋은 금융 예측기는 단순히 정답을 맞히는 모델이 아니라, 예측 시점에서 관측 가능한 정보와 과거 유사 사례를 바탕으로 타당한 사고 과정을 거친 probability forecast를 생성해야 한다.

---

## 2. 방법론 개요

### 2.1 전체 파이프라인

```text
Forecasting Question
        ↓
Recent Information View
        ↓
Short-Term Forecast
        ↓
Calibrated Aggregation
        ↓
Final Probability Forecast
```

```text
Forecasting Question
        ↓
Historical Signal Harness
        ↓
Signal Checklist + Scenario Prompts
        ↓
Signal-Guided Forecast
        ↓
Calibrated Aggregation
        ↓
Final Probability Forecast
```

두 branch를 분리하는 이유는 최신 뉴스와 과거 유사 사례가 서로 다른 역할을 하기 때문이다.

Recent Information View는 forecast date 이전의 뉴스, guidance, consensus, market reaction 등 현재 직접 관측 가능한 정보를 바탕으로 단기 예측을 수행한다.

Historical Signal Harness는 과거 resolved case에서 반복적으로 중요했던 signal, failure mode, scenario pattern을 가져와 이번 질문에서 반드시 점검해야 할 사고 절차를 만든다.

마지막으로 Calibrated Aggregation은 두 예측을 단순 평균내지 않고, 최신 정보의 직접성, signal checklist 충족 여부, scenario 불확실성, overconfidence 가능성을 고려해 최종 probability를 조정한다.

---

## 3. Reasoning Harness의 역할

Reasoning harness는 단순 prompt나 memory retrieval이 아니다. 과거 resolved episode를 바탕으로 현재 forecaster의 사고 과정을 제어하는 장치다.

### 3.1 Harness가 제공하는 정보

```text
1. 이번 질문에서 확인해야 할 핵심 signal checklist
2. 각 signal이 outcome에 미치는 방향성
3. 과거 유사 사례에서 자주 발생한 reasoning failure
4. base / upside / downside / surprise scenario
5. 최종 probability를 조정할 때 고려할 uncertainty cue
```

예를 들어 FOMC 예측에서는 다음과 같은 checklist가 생성될 수 있다.

```text
- Core inflation이 충분히 둔화되었는가
- Labor market cooling이 실제로 확인되는가
- Fed communication이 dovish/hawkish하게 바뀌었는가
- Market-implied probability가 최근 크게 repricing 되었는가
- Financial stability shock이 macro signal보다 우선될 수 있는가
```

이 구조에서 memory는 답을 주는 역할이 아니라, 현재 질문을 더 금융적으로 타당하게 생각하도록 만드는 역할을 한다.

---

## 4. 선행연구 대비 차별점

### 4.1 ForecastCompass와의 차이

ForecastCompass 계열은 과거 factor나 reasoning memory를 저장하고, 새 질문에서 관련 memory를 retrieve해 context로 제공하는 방향에 가깝다.

본 연구는 memory를 단순 context로 제공하지 않고, forecaster의 사고 절차를 제어하는 harness로 사용한다.

```text
ForecastCompass:
과거 factor와 reasoning memory를 저장하고 retrieve한다.

Ours:
과거 resolved case로부터 이번 질문에서 무엇을 확인하고,
어떤 시나리오를 고려하며,
어떤 reasoning failure를 피해야 하는지 결정한다.
```

### 4.2 APODEx-Forecasting과의 차이

APODEx-style 접근은 좋은 evidence를 수집하고 claim이 evidence에 의해 검증되는지를 확인하는 verification 중심 구조에 가깝다.

반면 금융 예측에서는 아직 정답이 없는 상황에서 여러 evidence가 동시에 다른 방향을 지지할 수 있다. 따라서 핵심은 evidence verification만이 아니라, uncertainty 아래에서 어떤 signal을 얼마나 반영하고 어떤 가능성을 열어둘지 결정하는 것이다.

```text
APODEx-style:
evidence chain을 검증한다.

Ours:
불확실한 forecast setting에서 signal coverage, scenario consideration,
confidence correction을 유도한다.
```

---

## 5. Research Questions

### RQ1. Harness가 forecast quality를 개선하는가

동일한 model, 동일한 forecast-time evidence, 동일한 cost budget에서 reasoning harness가 기존 memory/retrieval 방식보다 더 나은 probability forecast를 만드는지 확인한다.

### RQ2. Harness의 어떤 요소가 성능 개선에 기여하는가

Checklist, scenario expansion, critique, calibrated aggregation 중 어떤 요소가 실제 성능에 중요한지 ablation으로 확인한다.

### RQ3. Evolving update가 필요한가

질문이 resolve된 이후 새 episode를 memory와 harness에 반영하는 것이 frozen memory나 단순 retrieval보다 장기적으로 유리한지 확인한다.

---

## 6. Fair한 실험 세팅

모든 baseline과 ours는 다음 조건을 동일하게 맞춘다.

```text
- 동일한 base model
- 동일한 forecast date
- 동일한 forecast-time evidence corpus
- 동일한 retrieved evidence budget
- 동일한 past case budget
- 동일한 output format
- 동일한 temperature / reasoning effort
- 동일한 context 또는 cost budget
- 동일한 chronological test split
```

특히 memory leakage를 막기 위해 test question의 resolution 이후 정보는 forecast 시점에 제공하지 않는다.

가장 적절한 실험 방식은 chronological rolling setting이다.

```text
Past resolved episodes → memory 구축
Current snapshot → forecast
Official resolution 이후 → score 계산 및 memory update
Next snapshot → updated memory로 forecast
```

---

## 7. Baselines

### 7.1 Naive Recent Evidence Forecaster

과거 memory 없이 forecast date 이전의 최신 evidence만 보고 예측한다.

이 baseline은 현재 정보만 사용하는 일반 forecaster 대비 harness가 추가 가치를 주는지 확인하기 위한 기준이다.

### 7.2 Raw Similar Case Retrieval

과거 유사 resolved case의 question, outcome, rationale을 top-k로 retrieve해 그대로 context에 넣는다.

이 baseline은 단순 RAG와 reasoning harness의 차이를 보여주기 위해 필요하다.

### 7.3 Factor Memory Baseline

과거 case에서 factor만 추출해 저장하고, 새 질문에서 관련 factor를 retrieve한다.

ForecastCompass와 가장 가까운 비교군으로 사용할 수 있다.

### 7.4 Reflection Memory Baseline

과거 예측의 성공과 실패를 자연어 reflection으로 저장하고, 새 질문에서 유사 reflection을 제공한다.

일반적인 agent memory 방식과 forecast-specific harness의 차이를 보기 위한 baseline이다.

### 7.5 Evidence Verification Baseline

현재 evidence를 검색하고 source support와 temporal validity를 점검한 뒤 예측한다.

APODEx-style verification 접근과의 차이를 확인하기 위한 baseline이다.

### 7.6 Ours: Reasoning Harness Forecaster

과거 resolved case를 직접 답변 context로 넣지 않고, 다음 요소로 변환해 forecaster의 사고 절차를 제어한다.

```text
signal checklist
scenario prompts
failure mode warnings
missing-signal critique
calibrated aggregation rule
post-resolution harness update
```

---

## 8. Evaluation Metrics

Reasoning quality 평가는 과하게 가져가지 않고, main evaluation은 forecast score 중심으로 구성한다. 다만 우리 claim을 뒷받침하기 위해 최소한의 process-level 분석만 추가한다.

### 8.1 Main Forecast Metrics

```text
Accuracy
Brier Score
Log Score
ECE
Cost
```

Brier Score와 ECE를 핵심 metric으로 둔다. 본 연구는 probability forecast와 calibration을 개선하는 것이 목표이기 때문이다.

Accuracy는 보조적으로 사용하고, Cost는 harness가 복잡해질수록 발생하는 trade-off를 보여주기 위해 포함한다.

### 8.2 Lightweight Process Metrics

Reasoning quality를 별도 대규모 annotation으로 평가하지 않고, 작은 subset에서 간단히 확인한다.

```text
Signal Checklist Hit Rate
모델이 harness가 제시한 핵심 checklist 중 몇 개를 실제 reasoning에서 다루었는지 확인한다.

Direction Consistency
모델이 signal의 방향성을 명백히 반대로 해석하지 않았는지 확인한다.

Scenario Inclusion
모델이 단일 결론으로 수렴하지 않고 최소한 base / alternative scenario를 고려했는지 확인한다.
```

이 세 metric은 full reasoning evaluation이 아니라, harness가 의도한 사고 절차를 실제로 유도했는지 확인하는 sanity check로 사용한다.

따라서 논문에서는 process-level metric을 main claim의 중심에 두기보다, forecast score 개선을 설명하는 보조 분석으로 배치하는 것이 좋다.

---

## 9. Ablation Study

Ablation은 방법론의 핵심을 보여주는 데 가장 중요하다.

```text
Ours-full
= checklist + scenario expansion + critique + calibrated aggregation + evolving update

w/o checklist
= 과거 signal 점검 없이 scenario만 생성

w/o scenario expansion
= signal checklist만 사용

w/o critique
= missing signal / overconfidence 교정 제거

w/o calibrated aggregation
= recent forecast와 signal-guided forecast를 단순 평균

frozen memory
= resolution 이후 memory update 없음
```

가장 중요한 비교는 다음 네 가지다.

```text
Ours-full vs Raw Similar Case Retrieval
Ours-full vs Factor Memory
Ours-full vs Evidence Verification
Ours-full vs Frozen Memory
```

이 비교에서 성능과 calibration이 개선되면, 단순 retrieval이나 factor memory가 아니라 reasoning harness 자체가 기여했다는 주장을 할 수 있다.

---

## 10. 최종 논문 메시지

본 연구는 LLM forecaster의 성능 향상을 retrieval, memory recall, answer aggregation의 문제가 아니라, 금융 예측에 적합한 사고 절차를 어떻게 유도할 것인가의 문제로 재정의한다.

과거 resolved financial episode는 정답을 알려주는 example이 아니라, 새로운 예측에서 반드시 점검해야 할 signal, 고려해야 할 scenario, 피해야 할 reasoning failure를 구성하는 재료로 사용된다.

이를 통해 본 연구는 forecasting agent가 좋은 정보를 가져오는 것을 넘어, 불완전한 정보 아래에서 더 타당한 금융적 사고를 거쳐 probability forecast를 생성하도록 만드는 reasoning harness optimization framework를 제안한다.
