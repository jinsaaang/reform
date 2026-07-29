# HGF Experiment Design

## 1. 연구 목적

본 실험은 **사전에 생성한 worked exemplar를 고정한 HGF**의 forecasting
성능과 reasoning 품질을 평가한다. 핵심 질문은 다음과 같다.

1. HGF가 동일 조건의 6개 baseline보다 높은 forecasting 성능을 보이는가?
2. HGF의 counterevidence, target bridge, uncertainty 구성요소가 각각 성능에
   기여하는가?
3. 고정된 retrieval rule에서 exemplar 수 `k`가 달라질 때 성능은 어떻게
   변하는가?
4. HGF가 생성한 reasoning은 baseline보다 정량적으로 우수한가?
5. reasoning 품질의 향상이 실제 forecasting 성능 향상과 연관되는가?
6. HGF가 baseline의 오답을 정답으로 바꾸는 과정을 사례 수준에서 설명할 수
   있는가?

## 2. 공통 고정 조건

- Test set: chronological holdout 100문항
- Memory questions: 200문항
- DAG memory bank:
  `hgf_300_v2_final/memory_bank_manifest.json`에 정의된 최종 200개
- Evidence: 기존 cutoff-safe frozen E0/E1 evidence DB
- HGF implementation: 공개 패키지의 canonical forecasting logic
- Exemplar: `artifacts/exemplars`에 저장된 fixed worked exemplar
- Fixed exemplar는 모든 실험에서 **재생성하지 않는다**.
- 동일 question, cutoff, target contract, evidence, probability validator 및
  scorer를 사용한다.
- 각 실행의 worker 수는 4로 고정한다.
- 모델별 실행은 순차적으로 수행하여 전체 동시 호출 수도 4로 유지한다.
- 코드, manifest, exemplar 및 evidence artifact의 hash와 실행 시각을
  기록한다.
- Accuracy, Brier, NLL은 동일한 성공 100문항에서 계산한다. 실패한 case는
  동일 조건으로 재시도하여 100문항을 완성한 뒤 집계한다.

## 3. 실험 1: Main Table

### 3.1 비교 방법

1. Search-only Agent
2. Factor-Memory Agent
3. Case-Memory Agent
4. Text-Memory Agent
5. Direct DAG Agent
6. Prospective DAG Agent
7. HGF

HGF는 canonical runner와 고정된 worked exemplar를 사용한다. Baseline은 각
방법의 원본 구현을 사용하며 HGF의 worked exemplar를 입력으로 받지 않는다.

### 3.2 비교 모델

- `google/gemini-2.5-flash-lite`
- `openai/gpt-5-mini`
- `deepseek/deepseek-v3.2`

각 `model × method` 조건을 독립적으로 3회 반복한다. 모델 alias, provider,
실행 날짜 및 가능한 경우 provider snapshot 정보를 함께 기록한다.

### 3.3 평가 지표

- Accuracy
- Multiclass Brier score
- Negative log-likelihood, NLL
- Mean input/output/total tokens
- Mean latency

각 모델 안에서 3회 반복의 `mean ± standard deviation`을 보고한다. 서로
다른 모델의 결과를 하나의 평균으로 합치지 않는다.

HGF와 각 baseline의 차이는 동일 문항에 대한 paired difference로 계산한다.
HGF와 가장 강한 baseline의 Accuracy, Brier, NLL 차이에 대해서는
question-level paired bootstrap 95% confidence interval을 함께 보고한다.

### 3.4 Main table 형식

| Model | Method | Accuracy ↑ | Brier ↓ | NLL ↓ | Tokens ↓ | Latency ↓ |
|---|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash-Lite | Search-only | mean ± std | mean ± std | mean ± std | mean | mean |
| Gemini 2.5 Flash-Lite | ... | ... | ... | ... | ... | ... |
| GPT-5 mini | ... | ... | ... | ... | ... | ... |
| DeepSeek V3.2 | ... | ... | ... | ... | ... | ... |

## 4. 실험 2: HGF Component Ablation

### 4.1 목적

HGF 내부의 주요 구조가 forecasting 성능과 calibration에 기여하는지
평가한다.

### 4.2 조건

| Condition | 제거 또는 변경 내용 | 검증 대상 |
|---|---|---|
| Raw DAG | 동일하게 retrieval한 DAG를 HGF 구조화 없이 제공 | structured exemplar 전체의 효과 |
| Full HGF | 모든 구성요소 사용 | 기준 조건 |
| − Counterevidence | counterevidence path와 failure condition 제거 | 반대 경로 분석의 효과 |
| − Target Bridge | target bridge와 prospective target estimate 제거 | 현재 질문의 target·기간·threshold 연결 효과 |
| − Uncertainty | uncertainty instruction과 uncertainty field 제거 | calibration 효과 |

각 ablation은 해당 구성요소만 제거하고 다음 조건은 Full HGF와 동일하게
유지한다.

- 현재 evidence와 evidence limit
- retrieved memory question 및 DAG
- 고정된 worked exemplar 원본
- reasoning 및 boundary-mapping backbone
- probability schema, validator와 scorer
- max-token 설정과 worker 수

Raw DAG는 Main Table의 Direct DAG 결과가 위 조건을 모두 만족하면 해당
결과를 재사용한다. Evidence bank, retrieval 또는 forecast backbone이 다르면
ablation용 Raw DAG를 별도로 실행한다.

### 4.3 평가

- Accuracy, Brier, NLL
- Full HGF 대비 각 ablation의 paired difference
- Paired bootstrap 95% confidence interval
- `− Uncertainty`는 Brier와 NLL 변화를 중심으로 해석한다.

Component ablation은 기본적으로 Gemini 2.5 Flash-Lite에서 3회 반복한다.
다른 모델에 대한 ablation은 Main Table 결과상 모델별 경향이 크게 다를 때만
추가한다.

## 5. 실험 3: Number-of-Exemplars Sensitivity

### 5.1 목적

Exemplar selection algorithm 자체가 아니라, 고정된 rule에서 사용하는
exemplar 수 `k`에 대한 민감도를 평가한다.

### 5.2 조건

- `k ∈ {1, 3, 5, 7}`
- 모든 k에서 동일한 retrieval score와 tie-breaking rule을 사용한다.
- Top-k exemplar는 동일한 ranked order로 입력한다.
- 모든 exemplar는 사전에 생성된 artifact를 사용하며 재생성하지 않는다.
- 여러 exemplar를 결합하는 방식은 모든 k에서 동일하게 유지한다.
- 여러 exemplar를 새로 LLM으로 요약하거나 재구성하지 않는다.
- Exemplar별 schema와 token cap을 동일하게 유지한다.

본 실험은 **exemplar-count sensitivity**로 해석한다. Random selection,
mismatched exemplar 또는 exemplar-order sensitivity는 본 실험 범위에
포함하지 않는다.

### 5.3 평가

- Accuracy, Brier, NLL
- Input/total tokens, latency
- k에 따른 평균 성능과 비용 변화
- 각 k는 Gemini 2.5 Flash-Lite에서 3회 반복

결과는 k를 x축으로 하고 Accuracy, Brier, NLL을 y축으로 하는 sensitivity
plot으로 제시한다. 각 점에는 반복 실행의 mean ± standard deviation을
표시한다.

## 6. 실험 4: Reasoning LLM-as-a-Judge

### 6.1 평가 대상

기본 비교는 다음 두 조건으로 한다.

- Raw DAG
- Full HGF

Judge 모델은 모든 평가에서 **Gemini 3.1 Flash Lite**로 고정한다.
OpenRouter model ID는 `google/gemini-3.1-flash-lite`이다. Judge에는 다음을
제공한다.

- Question과 target contract
- Forecast cutoff
- 모델이 사용한 cutoff-safe evidence
- 생성된 written reasoning
- probability argmax로 정한 selected outcome

Judge에는 method 이름, forecaster 모델 이름 및 ground-truth outcome을
노출하지 않는다. 출력 순서는 무작위화한다. Forecast를 생성한 모델과 다른
모델을 judge로 사용하고, prompt를 모든 평가에서 고정한다. Correctness는
judge 판정이 저장된 뒤에만 결합한다.

### 6.2 평가 축

페이퍼의 Table 4 및 Section 5.4에 따라 1–5점 rubric과 composite score를
사용하지 않고 다음 세 비율을 보고한다.

#### Evidence coverage ↑

Written reasoning의 결정적 경로에 필요한 원자적이고 중복되지 않은 factual
evidence requirement를 식별한다. 각 requirement는 제공된 evidence record
ID를 인용해야 하며, 인용한 record가 forecast cutoff 시점에 이용 가능하고
실제로 해당 주장을 지지할 때만 supported로 판정한다.

```text
EvidenceCoverage =
    SupportedRequiredEvidenceItems / AllRequiredEvidenceItems
```

높을수록 좋다.

#### Invalid reasoning ↓

다음 중 하나라도 해당하면 invalid로 판정한다.

- 결정적 주장이 제공된 forecast-time evidence의 지지를 받지 못한다.
- Forecast cutoff 이후의 정보를 사용한다.
- Exact target contract 아래에서 selected outcome을 정당화하지 못한다.

```text
InvalidReasoningRate = InvalidForecasts / AllForecasts
```

낮을수록 좋다. 실제 outcome이 틀렸다는 사실만으로 reasoning을 invalid로
판정하지 않는다.

#### Invalid among correct ↓

Judge가 ground truth를 보지 않은 상태에서 `invalid_reasoning`을 먼저 기록한다.
그 후 저장된 실제 outcome과 결합하여 correct forecast 중 invalid reasoning인
비율을 계산한다.

```text
InvalidAmongCorrect =
    InvalidAndCorrectForecasts / AllCorrectForecasts
```

낮을수록 좋다. 이 값은 정확한 forecast가 잘못된 reasoning을 성공으로
가리는지를 진단하는 핵심 지표다.

### 6.3 보고 결과

- 방법별 Evidence coverage
- 방법별 Invalid reasoning rate
- 방법별 Invalid among correct rate
- 각 비율의 numerator와 denominator
- Judge의 parse failure 및 재시도 수

기존 네 개 1–5점 축의 평균인 composite reasoning score와 이를 사용한
Reasoning–Performance Link는 페이퍼 프로토콜에 없으므로 계산하지 않는다.

## 7. 실험 5: Qualitative Reasoning Case Study

### 7.1 목적

Main Table에서 확인된 HGF의 정량적 개선이 실제 case에서 어떤 reasoning
변화로 발생했는지 설명한다. Case study는 정량적 우월성을 다시 입증하기 위한
것이 아니라 개선 mechanism을 보여주는 illustrative analysis로 사용한다.

### 7.2 사례 선택

Main Table에서 HGF의 성능 우위가 확인된 뒤 다음 후보군을 만든다.

```text
HGF correct AND strongest baseline incorrect
```

후보군에서 Brier improvement가 크고 서로 다른 category와 개선 mechanism을
보여주는 2–3개 문항을 선택한다. 사례 선택 조건과 각 사례의 Brier improvement를
명시한다.

Failure case를 본문 case study의 필수 조건으로 두지 않는다.

### 7.3 사례별 제시 내용

1. Question, cutoff와 ground truth
2. Retrieval된 memory question과 고정된 worked exemplar의 핵심 부분
3. 사용된 current evidence
4. Baseline reasoning과 probability
5. HGF의 main path
6. HGF의 counterevidence 및 failure condition
7. Target bridge와 prospective target estimate
8. Uncertainty 판단과 최종 probability
9. Baseline 대비 Brier/NLL 개선

필요하면 다음 흐름을 하나의 diagram으로 표현한다.

```text
Retrieved DAG
  → Fixed worked exemplar
  → Current-evidence instantiation
  → Main/counter path comparison
  → Target bridge
  → Uncertainty-aware probability
```

## 8. 최종 논문 산출물

1. **Table 1 — Main Forecasting Results**
   - 3 models × 7 methods
   - Accuracy, Brier, NLL의 mean ± std

2. **Table 2 — HGF Component Ablation**
   - Raw DAG, Full HGF, 세 가지 removal condition

3. **Figure 1 — Number-of-Exemplars Sensitivity**
   - k별 Accuracy, Brier, NLL 및 token cost

4. **Table 4 — Reasoning Evaluation**
   - Evidence coverage, Invalid reasoning, Invalid among correct

5. **Figure/Table 5 — Qualitative Case Studies**
   - HGF가 baseline 오답을 교정한 2–3개 illustrative case

## 9. 본 설계에서 제외하는 실험

- Retrieval selection rule 자체의 비교
- Random 또는 mismatched exemplar control
- Exemplar order sensitivity
- Composite reasoning score
- Reasoning–Performance Link
- Reasoning 축별 performance regression
- 다변량 회귀 분석
- 별도의 temporal leakage 성능 실험
- 본문에서의 필수 failure-case study

