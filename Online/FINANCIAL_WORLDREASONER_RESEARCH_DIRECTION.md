# Financial WorldReasoner

## Temporal Causal Memory를 활용한 근거 기반 금융 시나리오 예측

> 공동연구 방향 제안서 · Discussion Draft
> 작성일: 2026-07-17
> 상태: 공동연구진 논의 및 구체화를 위한 초안

---

## 1. Executive Summary

본 연구의 주 construct는 실제 결과를 가린 상태에서, **예측 시점에 허용된 근거로 확률과 시나리오를 사전에 정당화하는 품질(outcome-blind ex-ante justification/reasoning quality)**이다. 정답을 맞혔는지는 이 주장을 정의하지 않는다. 대신 결과가 공개된 뒤의 **실현 경로 예상(realized-path anticipation)**과, 근거·기억을 바꿨을 때 출력이 어떻게 바뀌는지 보는 **행동적 반사실 반응(behavioral counterfactual responsiveness)**을 별도 진단으로 둔다.

운영 흐름은 다음과 같다.

1. **Searcher(Searching Agent)**는 질문·cutoff를 분석하고, 시간상 적격한 resolved-DAG memory를 검색 계획과 gap/counterevidence 탐색에 사용한다. 현재 시점의 근거는 Searcher만 수집하며, 시간 필터를 통과한 Evidence Pack으로 정리한다.
2. **Forecaster(Forecasting Agent)**는 필터된 Evidence Pack과 적격한 과거 DAG를 입력으로 받아 서로 다른 금융 시나리오, 결과 방향, 시나리오별 확률 구성, 설명과 불확실성을 출력한다. Forecaster는 현재 질문을 그래프로 만들지 않는다.

```text
Forecast-time eligible evidence + resolved-DAG memory
                         ↓
                  Searcher Evidence Pack
                         ↓
       Forecaster: scenarios / directions / probabilities / explanations
```

여기서 확률의 각 수치가 특정 원인의 기여도를 정확히 식별한다거나 모델 내부의 경제적 인과를 증명한다고 주장하지 않는다. 관측된 근거가 결과의 방향을 어떻게 지지·반박하는지, 어떤 대안 시나리오를 열어 두는지, 그 시나리오들을 어떤 근거와 불확실성으로 조합했는지를 설명하는 것이 목표다. Accuracy, Brier score, log score 및 calibration은 reasoning 품질을 대체하지 않는 **secondary outcome-safety/non-inferiority check**로 유지한다.

---

## 2. 연구 배경과 문제 정의

### 2.1 금융 예측에서는 정답만으로 충분하지 않다

맞은 예측도 근거 없이 결과를 추측했거나 미래 정보를 암묵적으로 사용했을 수 있고, 합리적인 reasoning도 낮은 확률의 충격 때문에 틀릴 수 있다. 따라서 한 번의 정답 여부로 reasoning 품질을 판정하지 않는다. 세 구성개념은 다음처럼 분리한다.

| 구성개념 | 관찰 시점과 질문 | 해석 범위 |
|---|---|---|
| **Ex-ante justification** (주 평가) | resolution 전, 현재 결과를 가리고 읽을 수 있는 근거·시나리오·확률 구성이 정당한가? | outcome-blind reasoning/evidence quality. 예측이 맞았는지와 독립적으로 평가한다. |
| **Realized-path anticipation** (post-resolution diagnostic; 사후 진단) | resolution 후, 당시 제시한 시나리오와 설명이 실제 경로의 일부 방향·전환을 예상했는가? | hindsight를 이용한 설명 보조 진단이며 ex-ante 점수를 덮어쓰지 않는다. |
| **Behavioral counterfactual responsiveness** (행동 감사) | 핵심 근거·역사 DAG·반대 신호를 제거·교란했을 때 검색·시나리오·확률이 어떻게 변하는가? | 관찰 가능한 반응과 auditability를 측정하며 내부 인과 증명이나 정확한 수치 귀속을 뜻하지 않는다. |

결과 점수는 위 주 평가를 안전성 관점에서 보완한다. 시나리오 확률은 단일 원인에 대한 정밀한 귀속값이 아니라, 근거·메커니즘·대안·불확실성을 함께 반영한 **scenario-based probability construction**으로 보고한다.

### 2.2 기존 LLM forecasting의 주요 한계

일반적인 search-augmented forecasting agent에는 다음과 같은 문제가 있다.

- 검색 결과를 나열하지만 어떤 정보가 결과에 영향을 주는지 구조화하지 못한다.
- 중요한 중간 변수나 반대 방향의 신호를 누락한다.
- 유사한 과거 사건을 발견하더라도 표면적 텍스트 유사성에 의존한다.
- 근거와 최종 확률 사이의 연결이 불투명하다.
- 결과를 먼저 생성한 뒤 그럴듯한 설명을 붙이는 post-hoc rationalization이 가능하다.
- generic web search의 날짜 필터 또는 최신 모델의 parametric knowledge를 통해 미래 정보가 유입될 수 있다.

특히 금융에서는 금리, 인플레이션, 기업 실적, 규제, 유동성, 신용위험, 시장 기대가 여러 사건·메커니즘 단계를 거쳐 결과 방향을 바꿀 수 있다. 따라서 단순 문서 나열보다 **시간 순서, 영향 방향, 반대 신호, 대안 시나리오를 근거와 함께 연결하는 접근**이 필요하다. 이는 경제적 인과 식별을 약속하는 것이 아니라, 예측 시점의 reasoning을 검토 가능한 형태로 만드는 것이다.

---

## 3. WorldReasoner와 공동연구의 출발점

WorldReasoner는 resolved forecasting question, simulated forecast date, forecast-time evidence를 기반으로 확률 예측·인용 근거·선택적 causal event graph를 평가하는 프레임워크다. **temporal search, forecasting, causal graph construction 및 graph evaluation은 이미 제공되는 WorldReasoner의 일반 기능이며 본 연구의 신규성으로 주장하지 않는다.** 본 연구는 그 금융 전용 데이터·기억·역할 경계를 확장한다.

- Repository: <https://github.com/cyzus/worldreasoner>
- v1.0.0 release: <https://github.com/cyzus/worldreasoner/releases/tag/v1.0.0>
- Paper: <https://arxiv.org/abs/2606.11816>
- Pinned finance pipeline config: <https://github.com/cyzus/worldreasoner/blob/6fd8ed1208f45290b017160edc8f928f45336785/src/config/pipeline.py#L10-L24>
- Forecasting architecture: <https://github.com/cyzus/worldreasoner/blob/main/docs/04_forecasting.md>
- Evaluation protocol: <https://github.com/cyzus/worldreasoner/blob/main/docs/05_evaluation.md>

v1.0.0에는 built graph가 있는 금융 질문 37개가 포함된다: **23 binary, 1 MCQ, 8 quantity, 5 timeframe**. 이 37개는 본 연구의 주 benchmark가 아니라 파이프라인과 평가 계약을 점검하는 **pilot/seed**로 사용한다. 이후 주 데이터는 WorldReasoner의 native finance-only 흐름인 **question → evidence → GraphBuilder**로 확장한다. 질문 유형은 섞어 단일 점수로 합치지 않고 각각 보고한다.

기존에 따로 모아 둔 legacy local 300/9,678 데이터셋은 개발·외부 검증의 참고 자료일 뿐, 주 benchmark로 삼지 않는다. 본 연구의 주 memory는 cutoff 이전에 resolution이 확인된 금융 episode의 resolved DAG와 historical outcome metadata이며, 현재 질문의 결과는 ex-ante 평가에서 가린다.

최근의 primary preprint [Analogical Deep Research](https://arxiv.org/abs/2607.13602)는 메커니즘에 맞는 역사적 유추(mechanism-aligned historical analogies)를 다루는 인접 연구다. 본 연구는 여기에 **확률적 금융 forecasting, temporal access control, resolved-DAG memory의 검색·사용, Searcher/Forecaster 분리, reasoning-centered 금융 평가**를 결합한다.

---

## 4. 핵심 연구 목표

### Goal 1. Finance-only resolved-DAG memory 확장

37개 v1.0 graph-built 금융 질문을 pilot/seed로 삼아, native **question → evidence → GraphBuilder** pipeline으로 질문·근거·resolution을 확장한다. 각 episode에는 cutoff, provenance, 시간 순서, 영향 방향, resolution 및 적격성 메타데이터를 함께 보존한다.

### Goal 2. Temporal resolved-DAG memory를 쓰는 Searcher 개발

Searcher가 먼저 cutoff와 현재 근거를 독립적으로 확인한 뒤, 적격 historical DAG를 검색 계획·coverage checklist·gap/counterevidence 탐색에 사용하도록 한다. Searcher는 최종 outcome probability를 만들지 않는다.

### Goal 3. Scenario/probability reasoning을 수행하는 Forecaster 개발

Forecaster는 필터된 Evidence Pack과 historical DAG를 받아 대안 시나리오, 결과 방향, scenario-based probability, 인용·불확실성·analogy limitation을 출력한다. 현재 질문을 그래프로 만들지 않으며, 모든 방향 설명은 근거와 역사적 node/path 대응으로 추적 가능하게 한다.

### Goal 4. Reasoning-centered 평가 체계 확립

Outcome metric을 주 endpoint로 삼지 않고, outcome-blind ex-ante justification·evidence grounding·temporal validity·scenario quality·설명 일관성을 중심으로 평가한다. Accuracy, Brier, log score, calibration은 유형별 secondary outcome-safety/non-inferiority check로 둔다.

### Goal 5. Memory retrieval과 reasoning transfer의 기여를 분해

검색에 historical DAG를 제공한 효과와, 동일한 Evidence Pack에서 Forecaster가 historical DAG를 사용한 효과를 별도 비교한다. 이 비교는 구성요소의 관찰 가능한 기여를 분해하기 위한 것이며, 경제적 인과나 모델 내부 인과를 식별한다고 해석하지 않는다.

따라서 신규성은 generic WorldReasoner 기능 자체가 아니라, (i) finance-specific temporally eligible resolved-DAG memory retrieval/use, (ii) Searcher/Forecaster decomposition, (iii) historical mechanism과 시나리오·확률 reasoning의 transfer, (iv) reasoning-centered finance evaluation에 있다.

---

## 5. 연구 질문과 가설

### RQ1. 적격 historical DAG memory가 ex-ante justification을 개선하는가?

현재 결과를 가린 동일한 질문·근거 조건에서, 적격 resolved-DAG memory를 사용한 조건이 search-only 또는 raw similar-case RAG보다 시간상 유효한 근거와 사건·메커니즘 연결을 더 잘 정당화하는지 확인한다.

**H1:** Historical-DAG 조건은 search-only 및 raw-case 조건보다 outcome-blind ex-ante justification 품질과 근거 coverage가 높고 unsupported leap가 적을 것이다.

### RQ2. Historical DAG가 검색 품질을 개선하는가?

과거 DAG의 node·mechanism 구조를 Searcher의 query planning과 gap/counterevidence 탐색에 사용했을 때, 단순 question-based search보다 cutoff 이전에 허용된 target-relevant 신호를 더 폭넓고 균형 있게 수집하는지 확인한다.

**H2:** DAG-guided search는 target-relevant source precision을 유지하면서 critical-signal recall과 counterevidence coverage를 높일 것이다.

### RQ3. Historical DAG에서 어떤 정보가 scenario/probability reasoning으로 전이되는가?

다음 조건을 비교해 node/path 대응, historical outcome 포함 여부, 구조적 추상화의 역할을 분리한다.

- raw historical question and rationale
- full historical DAG
- outcome-masked structural DAG
- domain-level causal template
- no historical memory

**H3:** 어떤 memory 표현이 근거에 맞는 결과 방향, 대안 시나리오, 불확실성 및 scenario-based probability 설명을 더 안정적으로 만드는지 검증한다. 이 실험은 확률 수치의 유일한 원인별 기여도를 산출하지 않는다.

### RQ4. Ex-ante reasoning 개선이 outcome safety와 양립하는가?

주 endpoint인 reasoning 품질을 높이면서도 question type별 accuracy, Brier, log score 및 calibration에서 사전 등록한 non-inferiority 기준을 지키는지 확인한다.

**H4:** Historical-DAG 조건은 ex-ante justification을 개선하되, 유형별 outcome safeguard에서 baseline 대비 허용된 범위를 벗어난 악화를 보이지 않을 것이다.

### RQ5. 금융 도메인 제약이 개연성 없는 시나리오를 줄이는가?

금융 ontology, 시간 순서, 영향 방향, 단위·범위 제약을 사용하는 조건과 일반적인 유추만 사용하는 조건을 비교한다.

**H5:** Domain-constrained 조건은 시간 순서 오류, 방향 불일치, target mismatch 및 핵심 메커니즘을 건너뛴 시나리오를 줄이고, 반대 신호를 설명에 포함하는 비율을 높일 것이다.

### RQ6. Historical memory와 근거 교란에 행동적으로 반응하는가?

핵심 근거 제거, 반대 근거 추가, historical DAG node/path 교란 및 counterfactual 질문으로 Searcher의 검색·coverage와 Forecaster의 시나리오·확률 변화를 측정한다. 이는 행동적 반응과 auditability를 보는 것이며 내부 causal proof가 아니다.

**H6:** 신호가 충분히 강한 경우 관련 교란은 결과 방향과 시나리오 확률을 일관되게 조정하고, 신호가 약한 경우에는 불확실성을 보존할 것이다.

---

## 6. 전체 시스템 아키텍처

본 연구의 online path는 현재 질문과 forecast cutoff에서 시작해, Searcher만 live web에 접근하는 두 단계 검색을 거쳐 Forecaster로 이어진다. 먼저 DAG와 독립적인 직접·open-world 검색으로 현재 맥락을 모으고, 시간·body-version gate를 통과한 항목만 flat profile로 만든다. 그 profile과 질문·Target Profile을 이용해 적격 historical DAG를 검색한 뒤, DAG-guided gap/counterevidence/open-world 검색을 한 번 더 수행한다. 최종 입력은 Evidence Pack과 원형 그대로의 적격 historical DAG이며, 질문 시점의 새 DAG 산출물은 온라인 경로에 없다.

~~~mermaid
flowchart LR
    subgraph OFFLINE["Offline: Finance-only Resolved DAG Memory"]
        SEED["WorldReasoner v1.0 finance seed"]
        FQ["Finance-only questions"]
        FE["Timestamped question evidence"]
        GB["GraphBuilder"]
        RD["Raw full resolved historical DAGs<br/>+ historical outcome + episode metadata"]
        SEED --> FE
        FQ --> FE
        FE --> GB --> RD
    end

    subgraph ONLINE["Online: Searcher-only live-web path"]
        Q["Current question + forecast cutoff"]
        TP["Target Profile"]
        FP["First-pass direct/open live-web search<br/>DAG-independent"]
        GATE["Temporal + body-version admission gate"]
        CCP["Flat Current Context Profile"]
        HF["Hard eligibility before ranking"]
        RET["Question/target + context/mechanism-aligned<br/>diverse historical-DAG retrieval"]
        SP["DAG-guided second-pass<br/>gap/counterevidence/open-world search"]
        EP["Final Evidence Pack<br/>+ raw eligible historical DAGs"]
        FC["Forecaster<br/>filtered inputs; no web access"]
        OUT["Scenarios / probabilities / explanations"]

        Q --> TP --> FP --> GATE --> CCP --> HF --> RET --> SP --> EP --> FC --> OUT
        RD --> HF
        CCP --> RET
    end

    subgraph POST["Post-resolution diagnostics"]
        RES["Resolved target outcome + hindsight reference"]
        EVAL["Outcome safety + realized-path + behavioral checks"]
        OUT --> EVAL
        RES --> EVAL
    end
~~~

### 6.1 역할 분리 원칙

Searcher는 현재 자료를 찾고 정리하는 유일한 web-access component다. Forecaster는 Searcher가 넘긴 자료와 historical DAG만 읽으며 외부 web, 검색 API, 검색 결과 순위를 볼 수 없다. 두 component 사이의 계약은 다음과 같다.

| 구성요소 | 책임 | 하면 안 되는 일 |
|---|---|---|
| **Offline Resolved-DAG Memory** | WorldReasoner finance seed와 finance-only native pipeline으로 만든 full resolved historical DAG, historical outcome, episode metadata 보존 | 온라인 질문에 맞춰 DAG 내용을 다시 쓰거나 target outcome으로 필터링 |
| **Searcher** | Target Profile 작성, first-pass direct/open 검색, temporal/body-version admission, flat Current Context Profile, eligible historical DAG retrieval, second-pass gap/counterevidence/open-world 검색, Evidence Pack 작성 | probability 또는 preferred outcome 출력, resolution 추측을 근거 선택에 사용, 질문용 DAG 산출 |
| **Temporal/body-version gate** | published/updated/available/retrieval timestamp와 body snapshot/hash를 확인하고 cutoff 이전에 허용되는 사실만 admission | 검색 snippet이나 현재 ranking을 evidence로 인정 |
| **Forecaster** | Target Profile, filtered Evidence Pack, raw eligible historical DAG, outcome space, domain constraints로 scenario/probability/explanation 작성 | web 접근, 새로운 자료 검색, historical DAG 구조 변경, 질문용 그래프 산출 |
| **Validator** | evidence ID·시간·scenario·probability·historical DAG reference·analogy mapping 계약 검사 | 정답을 보고 forecast를 수정하거나 경제적 인과를 증명한다고 해석 |

Searcher는 preferred outcome이나 probability를 내보내지 않으므로 evidence selection이 예상 정답에 맞춰 닫히지 않는다. Forecaster 출력은 prose/table 기반 scenario contract이며, 현재 질문을 표현한 별도 DAG 파일을 포함하지 않는다.

---

## 7. Offline Finance-only Resolved DAG Memory

Offline memory는 WorldReasoner v1.0의 graph-built finance questions를 pilot/seed로 삼고, native finance-only question → evidence → GraphBuilder pipeline으로 확장한다. Memory의 단위는 해결된 금융 episode이며, online에서 재구성하는 임시 구조가 아니라 사전에 provenance와 resolution을 고정한 역사 기록이다.

### 7.1 Main memory asset과 ablation 경계

Main condition은 **raw/full eligible resolved historical DAG + historical outcome + episode metadata**다. 선택된 DAG는 online 입력에서 structural rewriting, outcome 삭제, node 병합을 거치지 않고 그대로 Forecaster에 전달한다. Historical outcome은 과거 episode의 일부이며 현재 target의 realized outcome과 혼동하지 않는다.

다음 표현은 main input이 아니라 ablation 또는 diagnostic condition으로만 둔다.

- outcome-masked full DAG
- structural abstraction 또는 transferable mechanism template
- domain-level template
- raw similar-case text 또는 question/rationale

이 분리는 full resolved record가 제공하는 정보와 구조적 추상화·텍스트 유사성의 기여를 별도로 관찰하기 위한 것이다. 어느 표현도 정확한 경제적 인과 식별을 뜻하지 않는다.

### 7.2 Native finance-only 확장과 episode schema

확장 pipeline은 다음 계약을 따른다.

~~~text
finance-only question
        → timestamped evidence
        → GraphBuilder
        → resolved historical episode
~~~

각 episode는 최소한 다음 필드를 보존한다.

| 필드 | 의미 |
|---|---|
| episode_id, question_id, dag_id | 질문과 resolved DAG의 안정적 식별자 |
| target profile, outcome space, forecast cutoff | 질문이 요구하는 target·기간·판정 규칙 |
| historical outcome, resolution_available_at | 과거 resolution 결과와 그 결과가 사용 가능해진 시점 |
| node_id, path_id, event time, available time | historical DAG 안의 사건·경로와 관찰 가능 시점 |
| source_id, canonical URL, body version, content hash | 각 사실의 provenance와 재현 가능한 문서 본문 |
| mechanism tags, domain, actor/entity IDs | 검색 정렬과 mechanism diversity에 사용할 제한된 표지 |
| episode relation flags | same-event, shared-resolution, derived-question, near-duplicate 여부 |
| construction and quality audit | GraphBuilder 기록, source coverage, temporal consistency, offline DAG quality |

occurred_at과 available_at은 분리한다. 사건이 일어난 시점과 외부 관찰자가 그 사실을 확인할 수 있게 된 시점이 다를 수 있기 때문이다. Historical outcome과 resolution evidence도 별도 필드로 남겨 사후 정보의 출처를 추적한다.

### 7.3 Historical DAG 의미와 품질

Historical DAG의 node·edge에는 observed, supported, hypothesis, disputed 같은 epistemic status를 붙인다. 문서가 직접 지지하는 관계와 분석적 가설을 같은 사실처럼 취급하지 않으며, 영향 방향도 positive, negative, mixed, unknown으로 기록한다. 동일 변수의 feedback은 time-expanded historical node로 표현할 수 있지만, 이 표현은 경제적 인과를 증명하지 않는다.

Offline quality gate는 historical asset 자체에만 적용한다.

- historical DAG가 acyclic인지와 event time 순서가 가능한지
- factual node와 path가 accepted source에 연결되는지
- resolution evidence와 historical outcome이 명확히 분리되는지
- 중복 episode와 지나치게 세분화된 event가 없는지
- causal relation과 단순 선후관계의 epistemic status가 구분되는지
- domain mechanism, target relation, unit/range가 일관되는지

이 검사는 offline memory의 provenance를 확인하는 절차이며, online에서 새 질문을 그래프로 판정하는 단계가 아니다.

---

## 8. Historical DAG Retrieval

### 8.1 Hard eligibility: ranking보다 먼저

Forecast cutoff를 t라고 할 때 candidate episode i는 다음 조건을 모두 만족해야 한다.

~~~text
resolution_available_at(i) < forecast_cutoff
same_underlying_event(i, target) = false
shared_resolution(i, target) = false
derived_question(i, target) = false
near_duplicate(i, target) = false
~~~

해당 조건 중 하나라도 확인되지 않으면 candidate는 ranking pool에서 제거한다. Same-event, shared-resolution, derived-question, near-duplicate를 similarity penalty로 낮추어 남겨 두지 않는다. Historical outcome이 이미 사용 가능하다는 사실은 첫 조건을 통과한 뒤 raw/full DAG를 Forecaster에 제공할 수 있게 하는 episode property이며, 현재 target outcome을 미리 노출하는 것이 아니다.

### 8.2 Candidate representation과 alignment

Candidate generation은 현재 question과 Target Profile을 기반으로 한다. Hard gate를 통과한 뒤에만 accepted first-pass Current Context Profile과 mechanism alignment를 추가한다.

사용 가능한 feature는 다음과 같다.

- question semantics, domain/subdomain, target entity와 actor type
- target variable, outcome space, horizon, resolution rule
- Current Context Profile의 accepted slot, source timing, direction, contradiction/missing-signal tag
- financial mechanism, event family, institutional/regulatory setting
- historical DAG의 node/path type과 mechanism tags

Target outcome, resolution 이후의 target-specific future information, 또는 현재 ranking index의 hindsight signal은 query representation에 넣지 않는다.

### 8.3 Retrieval order와 diversity

Retrieval은 다음 순서를 고정한다.

1. Question/Target Profile로 broad candidate를 생성한다.
2. resolution_available_at < forecast cutoff와 네 가지 event-relation exclusion을 hard filter로 적용한다.
3. Accepted Current Context Profile과 mechanism alignment를 사용해 relevance를 rerank한다.
4. 서로 다른 mechanism과 episode family가 남도록 diversity selection을 적용한다.
5. 선택된 item을 structural rewriting 없이 raw/full resolved historical DAG와 historical outcome metadata로 반환한다.

Context와 mechanism이 모두 빈약하면 profile의 missing/uncertain slot을 그대로 기록하고, 근거가 없는 candidate를 preferred outcome으로 채우지 않는다.

### 8.4 Retrieval output

Searcher와 Forecaster 사이에서 historical memory는 다음 필드를 함께 전달한다.

- dag_id, episode_id, question_id
- historical outcome 및 resolution_available_at
- raw node/path IDs와 source IDs
- eligibility audit와 exclusion checks
- matched Target Profile field와 Current Context Profile slot
- mechanism alignment, diversity cluster, analogy-fit limitation placeholder

Outcome-masked, structural, domain-template, raw-text 표현은 별도 ablation branch로만 전달하며 main branch의 raw record를 대체하지 않는다.

---

## 9. Online Searching Agent

### 9.1 입력과 출력 경계

Searcher input은 current question, forecast cutoff, outcome space, source access policy, target entity/metric/horizon/resolution rule이다. Offline memory catalog는 retrieval 단계에서 조회하지만, 첫 pass가 끝나기 전에는 historical DAG의 node/path를 query planning에 노출하지 않는다.

Searcher output은 Target Profile, flat Current Context Profile, accepted evidence items, raw eligible historical DAG references, query/audit log, coverage gaps, counterevidence, open-world findings와 limitations다. Searcher는 probability, preferred outcome, 또는 resolution guess를 출력하지 않는다.

### 9.2 검색 절차

#### Step 1. Current question → Target Profile

Searcher는 entity, metric, threshold/range, target period, forecast cutoff, resolution condition, outcome space, 혼동 가능한 인접 target을 명시한다. 질문이 불완전하거나 context를 추출할 수 없으면 missing slot과 uncertainty를 기록하고 sparse Target Profile을 그대로 다음 단계에 전달한다.

#### Step 2. First-pass direct/open live-web search (DAG-independent)

첫 pass는 historical DAG나 mechanism template를 읽지 않고 현재 question과 Target Profile만으로 수행한다. 최소 branch는 다음과 같다.

1. direct target search
2. official/source-of-record search
3. independent open-world search for new factors and disconfirmers
4. resolution-rule search

Searcher는 이 pass에서 preferred outcome이나 probability를 정하지 않는다. 검색 snippet은 lead일 뿐 evidence item이 아니다.

#### Step 3. Temporal and body-version admission gate

각 결과에 대해 canonical URL/document ID, title/publisher, published/updated timestamp, availability timestamp, retrieval timestamp, exact body version, content hash 또는 archived snapshot ID를 기록한다. Forecast cutoff 이전에 실제로 읽을 수 있었던 body version을 확인할 수 없는 항목은 main Evidence Pack에 admission하지 않는다. 인정된 item만 Current Context Profile slot이 된다.

#### Step 4. Flat Current Context Profile

Current Context Profile은 DAG가 아닌 slot-based retrieval profile이다. 각 slot은 독립적으로 다음을 기록한다.

| Slot | 내용 |
|---|---|
| target/cutoff | Target Profile과 cutoff에 대한 확인 |
| observation | accepted claim, source ID, body version, available time |
| direction | target에 대한 supports, weakens, mixed, unknown |
| mechanism tag | 어떤 금융 mechanism 또는 event family와 관련되는지 |
| contradiction | 서로 다른 source가 충돌하는지 |
| missing/uncertainty | 아직 확인되지 않은 signal, sparse context, confidence limitation |

Profile에는 node/edge 구조, acyclicity, 또는 미래의 intermediate event를 넣지 않는다. Accepted item이 없을 때는 빈 observation과 missing/uncertainty slot을 반환하며 사실을 임의로 보충하지 않는다.

#### Step 5. Eligible historical DAG retrieval

Searcher는 Section 8의 순서를 호출한다. Question/Target Profile과 accepted Current Context Profile을 함께 사용하고, hard eligibility와 same-event/shared-resolution/derived/near-duplicate exclusion을 ranking 전에 적용한다. Context 및 mechanism alignment로 rerank한 뒤 diversity를 적용하고 raw/full eligible historical DAG를 보존한다.

#### Step 6. Second-pass DAG-guided gap/counterevidence/open-world search

선택된 historical DAG의 node/path IDs와 mechanism tags는 검색 checklist로만 사용한다. Searcher는 다음 branch를 모두 유지한다.

- historical mechanism에서 비어 있는 gap
- 주 시나리오와 반대되는 counterevidence
- historical DAG에 없던 novelty/open-world factor
- source-of-record와 resolution-rule 재확인

이 pass도 live web 접근의 범위 안에서 수행되며, 결과를 선호 outcome으로 요약하지 않는다. Historical analogy가 맞지 않는 신호와 search limitation을 명시적으로 남긴다.

#### Step 7. Final Evidence Pack

최종 Evidence Pack은 accepted evidence item의 claim, citation, body-version audit, timing, direction, contradiction, assumption boundary와 selected raw historical DAG references를 묶는다. Flat Current Context Profile은 별도 field로 보존하고, target question용 graph artifact는 추가하지 않는다.

### 9.3 Searcher output contract

| 필드 | 계약 |
|---|---|
| target_profile | question, cutoff, outcome space, resolution rule |
| current_context_profile | flat slots, accepted source/body versions, directions, contradictions, missing/uncertainty |
| evidence_items | evidence_id, claim, citation, available/retrieval time, body hash, relation to target/slot |
| eligible_historical_dags | raw dag_id/episode_id, historical outcome, node/path IDs, eligibility audit |
| query_log | first-pass/second-pass branch, query, source policy, result limitation |
| counterevidence/open_world | contrary signals, novel factors, unresolved gaps |
| search_limitations | sparse context, inaccessible body version, analogy mismatch, residual ranking bias |

---

## 10. Online Forecasting Agent

### 10.1 입력과 access boundary

Forecaster는 다음만 입력으로 받는다.

- Target Profile
- temporal/body-version gate를 통과한 filtered Evidence Pack
- raw/full eligible resolved historical DAGs with historical outcome and episode metadata
- allowed outcome space와 question-specific resolution rule
- finance domain constraints, units/ranges, and scenario policy
- Searcher가 기록한 uncertainty, counterevidence, open-world limitations

Forecaster에는 web client, search tool, live ranking, 또는 ground-truth resolution이 제공되지 않는다. Historical DAG는 main condition에서 원형 그대로 읽으며, outcome-masked/structural/domain-template/raw-text는 실험 분기에서만 사용한다.

### 10.2 Scenario contract

각 scenario는 다음 필드를 갖는다.

| 필드 | 요구사항 |
|---|---|
| scenario_id/name | 서로 구분되는 outcome-relevant branch |
| ordered reasoning path | prose로 된 시간 순서와 방향 설명; 각 단계에 evidence_id와 historical dag_id/node_id/path_id를 연결 |
| current-context mapping | 어떤 Current Context Profile slot이 어떤 historical node/path와 대응하는지 |
| outcome direction | target outcome에 대한 supports, weakens, mixed, unknown |
| scenario probability | scenario-based construction과 uncertainty를 함께 설명 |
| conditional outcome distribution | allowed outcome space 안의 조건부 확률 또는 범위 |
| citations | Evidence Pack의 accepted evidence ID만 사용 |
| assumptions | observation과 구분되는 가정·외삽·missing signal |
| triggers/disconfirmers | scenario를 강화·약화할 관찰 가능한 신호 |
| uncertainty | calibration limitation, sparse context, conflicting evidence |
| analogy limitations | historical mechanism이 현재 target에 맞지 않을 수 있는 조건 |

Forecaster는 evidence 방향과 historical node/path mapping을 설명하되, 확률의 각 소수점이 특정 원인의 정밀한 기여도라고 주장하지 않는다. 결과 방향과 scenario-based probability construction은 추적 가능하게 쓰되, 내부 경제적 인과 증명으로 표현하지 않는다. Scenario set은 주요 plausible branches를 포괄하고 중복을 최소화하며, 정보 부족을 residual/uncertain scenario로 보존한다.

### 10.3 Probability aggregation

최종 outcome probability는 scenario reasoning과 별개로 다시 추측하지 않고, scenario probability와 조건부 outcome distribution에서 집계한다.

~~~text
P(outcome = y) = sum over s of P(scenario = s) × P(y | s)
~~~

각 항은 scenario_id, evidence IDs, historical DAG node/path IDs, assumptions와 연결된다. Probability normalization과 aggregation trace는 outcome space별로 보존하며, 이 식은 설명 가능한 조합 규칙이지 원인별 수치 귀속의 증명이 아니다.

### 10.4 Validator와 abstention

Validator는 정답을 보지 않고 다음 계약을 검사한다.

- Evidence Pack의 evidence ID와 citation/body-version admission
- temporal admission과 forecast cutoff
- scenario coverage, overlap, residual branch
- observation, assumption, trigger/disconfirmers의 구분
- Target Profile과 scenario/terminal outcome의 일치
- probability normalization 및 scenario aggregation trace
- unsupported causal leap, direction/polarity inconsistency, domain constraint 위반
- referenced historical dag_id, node_id, path_id의 존재와 eligibility
- Current Context Profile slot과 historical node/path analogy mapping
- analogy limitation과 uncertainty의 명시

온라인 validator는 scenario contract와 historical DAG reference만 검사하며 cycle/edge/acyclicity 검사는 수행하지 않는다. Evidence가 sparse하거나 body version이 admission되지 않으면 Forecaster는 confidence를 낮추거나 uncertainty/residual scenario를 출력하고, 빈 context를 사실로 채우지 않는다.

---
## 11. Temporal Leakage 및 Contamination 방지

### 11.1 하나의 Searcher-only live-web architecture와 모드별 evidence handling

현재 미해결 질문과 historical backtest는 서로 다른 검색 시스템을 만들지 않고, 하나의 Searcher-only live-web architecture를 공유한다. Searcher가 유일한 web-access boundary이며 질문, cutoff, Target Profile, source policy를 받아 live web을 조회하고 temporal/body-version gate를 적용한다. Forecaster는 web client, 검색 도구, live ranking, 검색 snippet, 또는 ground truth에 접근하지 못하며, gate를 통과한 Evidence Pack과 eligible historical DAG reference만 입력으로 받는다. Searcher는 preferred outcome이나 probability를 출력하지 않는다.

WorldReasoner의 pinned forecasting 문서는 generic Temporal Gateway를 cutoff-aware evidence access capability로 설명한다. 이 연구는 그 generic gateway를 재사용 가능한 경계로 존중하되, 금융 연구에 더 엄격한 body-version 검증과 exclusion 정책을 추가한다: [WorldReasoner Temporal Gateway documentation, pinned commit, lines 89–99](https://github.com/cyzus/worldreasoner/blob/6fd8ed1208f45290b017160edc8f928f45336785/docs/04_forecasting.md#L89-L99). Gateway의 일반적인 cutoff view만으로 exact historical body를 입증했다고 간주하지 않으며, 아래 mode별 gate와 provenance log가 연구의 admission 기준이다.

별도의 고정형 웹 데이터베이스, 사전 구축 corpus, 또는 웹 검색 인덱스를 연구 요구사항으로 만들지 않는다. Searcher가 retrieval 시점에 읽은 accepted current body를 snapshot하고 hash하는 것은 재현·감사 logging이며, 별도의 고정형 웹 데이터베이스를 새로 만드는 것이 아니다. Snapshot은 audit artifact로 보존하지만 새로운 web corpus나 검색 backend를 그 artifact에서 운영하지 않는다.

### 11.2 Current unresolved questions: live retrieval과 retrieval-time audit

현재 아직 해결되지 않은 질문에는 Searcher가 live web retrieval을 허용한다. 각 accepted evidence item에 대해 다음을 함께 기록한다.

- retrieval_timestamp, publication_timestamp, update_timestamp (제공되지 않으면 missing으로 명시)
- canonical URL, source/publisher, document identifier, title 및 source metadata
- Searcher가 실제로 수락한 exact body와 retrieval-time snapshot identifier
- exact body의 content hash, available-at 판단, query/branch 및 admission decision
- Current Context Profile slot, target relation, 방향, contradiction, uncertainty 및 limitation

Snapshot/hash는 해당 retrieval 시점의 accepted body를 감사 가능하게 고정하는 기록일 뿐, 별도 데이터베이스를 구축하는 데이터 적재 단계가 아니다. 이 모드에서 질문의 ground truth 또는 resolution은 submission 전 시스템에 제공되지 않는다. Ground truth is unavailable to the system before submission. 현재 live search의 ranking과 result ordering은 나중에 발생한 사건이나 현재의 popularity를 반영할 수 있으므로, retrieval-time snapshot을 남겨도 그 hindsight를 완전히 제거했다고 주장하지 않는다.

Search snippet은 discovery hint일 뿐이며 evidence가 아니다. Snippet text, snippet timestamp, 또는 검색 결과의 publication date만으로 Evidence Pack에 항목을 admission하지 않는다. Evidence citation은 Searcher가 읽고 hash한 body에만 연결한다.

### 11.3 Historical backtests: simulated cutoff와 exact pre-cutoff body gate

Historical backtest는 질문별 simulated cutoff value (cutoff)를 선언한다. 동일한 Searcher-only live-web boundary를 사용하더라도, 현재 response가 과거에 보였을 것이라는 단순 날짜 필터를 가정하지 않는다. Evidence item은 다음 조건을 모두 만족할 때만 main backtest에 admission한다.

1. cutoff 이전에 이용 가능했던 exact pre-cutoff body version이 archive 또는 versioned copy에서 독립적으로 확인된다.
2. 그 archived/versioned copy의 body bytes 또는 canonical normalized body가 accepted body와 일치하고, version timestamp/availability와 content hash가 기록된다.
3. canonical URL/source metadata, publication/update/retrieval timestamps, archive/version URI, verification method 및 provenance가 재검증 가능하다.

페이지가 cutoff 전에 published 되었지만 cutoff 후 수정된 경우에는 exact pre-cutoff body를 recover할 수 있을 때만 admission한다. Pre-cutoff body를 recover할 수 없으면 현재 body로 대체하지 않고 제외한다. Publication date alone, current body, 또는 snippet만으로는 충분하지 않다. Archive/versioned copy가 없거나 body equality를 independently verify할 수 없는 항목도 main experiment에서 제외하고 unverifiable_body_version, missing_archive, hash_mismatch, post_cutoff_only, 또는 해당되는 구체적 reason을 log한다.

Historical backtest의 accepted body는 live retrieval의 convenience 결과가 아니라 cutoff 이전 버전을 확인한 결과다. 따라서 current live body를 snapshot한 것만으로 historical admission을 만들지 않으며, exact pre-cutoff copy가 검증되지 않으면 해당 evidence item은 robustness 분석에도 별도 exclusion record로 남긴다.

### 11.4 Historical DAG eligibility와 hard event exclusions

Historical DAG memory에는 다음 gate를 적용한다.

- Historical DAG eligibility: resolution_available_at < cutoff
- Historical DAG source: every exposed source is available by cutoff
- Question ground truth: hidden until the forecast submission

resolution_available_at < cutoff를 만족하지 않는 episode는 unresolved-future로 간주하여 Forecaster memory에 들어가지 않는다. Ranking 전에 다음 관계를 hard exclusion하며, similarity penalty로 낮춰 남겨 두지 않는다.

- same-event cases
- shared-resolution cases
- derived questions or questions directly dependent on the same resolution
- near-duplicate questions

이 hard filter는 question/Target Profile과 accepted Current Context Profile을 이용한 candidate generation보다 먼저 적용된다. Exclusion은 candidate ID, matched event/resolution key, cutoff, 판정 시각 및 reason을 provenance log에 남긴다. Main memory branch는 eligible full resolved DAG와 historical outcome을 사용하며, outcome-masked 또는 structural abstraction은 별도 ablation으로만 취급한다.

### 11.5 Provenance, admission, exclusion, and interruption logs

각 run은 evidence/DAG item별로 admitted 또는 excluded 결정을 append-only audit log에 기록한다. 최소 필드는 item ID, canonical URL/source metadata, retrieval/publication/update/availability timestamps, simulated cutoff와 mode, exact body hash 또는 archive/version identifier, DAG resolution_available_at, event/resolution relation flags, verifier, exclusion reason, query branch, model/provider metadata다. Malformed or ambiguous timestamps, missing timezone, inconsistent update order, malformed archive metadata, cancellation/resume, retry, and interrupted commands는 성공으로 간주하지 않고 해당 item/run의 validation status와 reason을 기록한다.

### 11.6 Model contamination과 정직한 retrospective claim language

문서 access를 막아도 parametric model knowledge가 미래 사실을 포함할 수 있으므로 requested/returned model, provider, model/version hash, declared knowledge cutoff, forecast date 및 contamination eligibility를 기록한다. model cutoff < forecast date인 clean subset을 별도로 보고할 수 있지만, provider cutoff 선언만으로 모든 parametric contamination을 제거했다고 보장하지 않는다. Parametric model contamination cannot be fully eliminated.

또한 current live search ranking과 snippets가 hindsight를 encode할 수 있고, ranking policy를 과거 시점 그대로 재구성할 수 없다. 그러므로 retrospective 결과는 항상 temporally filtered 또는 leakage-minimized라고 부른다. leakage-free, fully leakage-free, zero leakage, 또는 “temporal violations가 없다”라고 표현하지 않는다. 이 명칭은 exact pre-cutoff body gate와 DAG hard exclusions를 통과했다는 뜻이지, 모든 정보 누출을 제거했다는 뜻이 아니다. Prospective evaluation은 검색 ranking과 body version을 미래에 생성되는 순서 그대로 관찰할 수 있어 가장 강한 clean validation으로 별도 운영한다.

### 11.7 Mode-specific reporting boundary

Current unresolved track은 live retrieval, retrieval-time snapshot/hash, provenance 및 ground-truth unavailability를 보고한다. Historical backtest track은 simulated cutoff, independently verifiable exact pre-cutoff body만 admission, unverifiable-body exclusion reason, resolution_available_at < cutoff, hard event exclusions, residual ranking/snippet hindsight, and parametric-contamination limitations를 보고한다. 두 모드의 Evidence Pack과 결과를 섞어 “완전한 temporal isolation” 또는 zero-leakage 결론을 만들지 않는다.

---

## 12. 실험 설계

본 연구는 서로 다른 질문을 답하는 세 estimand를 분리한다. 검색이 더 좋은 Evidence Pack을 만드는지, 동일한 Evidence Pack에서 historical memory 표현이 Forecaster의 조건부 reasoning을 바꾸는지, 그리고 전체 bundle이 baseline보다 나은지는 서로 바꿔 말할 수 없다. 모든 run은 질문·forecast cutoff·chronological split·모델/프롬프트 버전·decoding·budget·출력 schema를 사전에 고정하고, exact parameter와 제외 사유를 run manifest에 기록한다. 구체적인 `top-k`, 모델/API, rubric weight, 통계 검정과 outcome non-inferiority margin은 preregistered/configurable implementation parameter로 남긴다.

### 12.1 Experiment A — Searcher와 Evidence Pack

**질문:** question-only/direct first-pass retrieval에 비해, accepted Current Context Profile과 eligible historical-DAG retrieval을 사용한 두 번째 gap/counterevidence/open-world pass가 Evidence Pack의 관찰 가능한 속성을 개선하는가?

| 조건 | Searcher treatment | Forecast 단계 | 평가 대상 |
|---|---|---|---|
| **A0. Direct first pass** | 현재 question + Target Profile만으로 DAG-independent direct/open search를 한 번 수행하고 temporal/body-version gate를 통과한 pack을 반환 | 실행하지 않음 | temporal admission/provenance, target relevance, source coverage, citation traceability, contradiction와 missing-signal 기록 |
| **A1. Context + eligible-DAG two pass** | A0의 pack으로 flat Current Context Profile을 만든 뒤, hard-eligible historical DAG를 retrieval하고 mechanism/diversity-aware gap·counterevidence·open-world search를 추가 | 실행하지 않음 | A0와 동일한 pack 속성에 더해 counterevidence coverage, mechanism/episode diversity, novel-factor coverage, exclusion audit |

A0 대 A1의 supported estimand는 **Searcher retrieval treatment의 Evidence Pack 기여**다. A의 판단에는 Forecaster output, outcome score, scenario quality를 넣지 않으며, A 결과로 historical DAG가 forecasting reasoning을 개선한다거나 어떤 component의 total causal effect가 있다고 주장하지 않는다. 검색 예산·source policy·temporal gate·pack schema는 두 조건에서 동일하게 preregister한다. A1의 두 번째 pass가 historical DAG를 검색 checklist로 사용하는 것은 의도된 treatment이며, 이 실험은 그 검색 treatment의 retrieval consequence만 추정한다.

### 12.2 Experiment B — Fixed-evidence Forecaster

**Canonical pack origin (locked):** B의 Evidence Pack `Pack-B0`는 historical DAG lookup 또는 DAG-guided query planning이 시작되기 **전에**, DAG-independent A0 first-pass Searcher가 preregistered retrieval policy로 생성한다. Accepted item의 canonical serialization, body-version hash, ordering rule, pack hash를 기록하고 byte-identical하게 freeze한다. 어떠한 DAG-guided second-pass result, historical outcome, hindsight signal도 `Pack-B0`에 들어갈 수 없다. B의 모든 condition은 동일한 `Pack-B0`, question, cutoff, Forecaster model/version, prompt, decoding, output schema, token/cost budget을 받는다.

| 조건 | Forecaster historical-memory treatment | `Pack-B0` 이후 추가 입력 | 목적 |
|---|---|---|---|
| **B0. No historical memory** | 없음 | 없음 | memory가 없는 fixed-evidence 기준선 |
| **B1. Unstructured raw cases** | eligible historical case의 raw text/question/rationale | 구조를 노출하지 않은 raw case만 | DAG 구조가 아닌 사례 텍스트의 조건부 기여 |
| **B2. Full/raw eligible resolved DAG (main)** | raw/full eligible resolved historical DAG와 historical outcome·episode metadata를 원형 그대로 제공 | `dag_id`, node/path reference와 eligibility audit | locked main historical-memory representation의 조건부 기여 |
| **B3. Outcome-masked/structural DAG** | B2와 같은 episode의 outcome-masked 또는 preregistered structural abstraction | masking/abstraction manifest | outcome 또는 표현 구조에 대한 shortcut/representation 진단 |
| **B4. Irrelevant but temporally eligible DAG** | cutoff 전에 resolution이 available하고 모든 same-event/shared-resolution/derived/near-duplicate exclusion을 통과했지만 현재 mechanism과 관련성이 낮은 DAG | eligibility와 irrelevance rationale | 단순 historical-DAG 존재/길이 효과와 anchoring 진단 |

B0–B4는 **동일한 fixed Evidence Pack에서 historical-memory representation을 바꾼 Forecaster의 conditional effect**를 추정한다. B에서는 Searcher를 다시 호출하지 않고, A1의 two-pass output을 고정 pack으로 사용하지도 않는다. 따라서 B의 차이를 DAG-guided search effect, retrieval quality, 모델 차이, prompt 차이, 또는 outcome hindsight의 효과로 해석하지 않는다. B2가 main condition이며 B3는 main input을 대체하지 않는 ablation이다.

### 12.3 Experiment C — End-to-end bundle

End-to-end에서는 Searcher와 Forecaster를 함께 실행해 실제 사용 흐름의 bundle effect를 보고한다.

| 조건 | 전체 treatment | 해석 경계 |
|---|---|---|
| **C0. Question-only + no memory** | A0 Searcher → Evidence Pack → B0 Forecaster | no-memory end-to-end baseline |
| **C1. Two-pass + no memory** | A1 Searcher → Evidence Pack → B0 Forecaster | 두-pass Searcher를 추가한 bundle contrast |
| **C2. Two-pass + full DAG (main bundle)** | A1 Searcher → Evidence Pack + raw/full eligible DAG → B2 Forecaster | 제안하는 full bundle |
| **C3. Two-pass + representation ablation** | A1 Searcher → Evidence Pack + B1/B3/B4 memory → Forecaster | bundle-level memory representation diagnostic |

C 결과는 C0/C1/C2/C3 사이의 **전체 시스템 bundle effect**만 말해 준다. C2가 C0보다 좋다고 해서 그 차이를 Searcher, historical DAG retrieval, memory representation, Forecaster reasoning 중 하나의 component에 귀속하지 않는다. Search/retrieval attribution은 A에서, fixed-pack memory attribution은 B에서만 보고한다. C의 primary reasoning comparison과 secondary outcome safety check는 동일한 cutoff와 preregistered run policy에서 함께 기록하되, retrospective live-web 결과는 temporally filtered/leakage-minimized로 표현한다.

### 12.4 공정한 비교를 위한 공통 통제

- question, outcome space, forecast cutoff, chronological split, source-access policy와 temporal/body-version gate
- Forecaster와 Judge의 model/version, prompt, decoding, output schema, repair/cancellation policy
- search/API call budget, retrieved-token budget, generation-token/cost budget과 timeout policy
- evidence ID/citation serialization, scenario/probability output contract, unit/range와 resolution rule
- run seed, order randomization, run-order-swapped duplicate policy, exclusion/invalid-output logging

조건이 추가 memory 또는 추가 검색을 사용하면 quality만 비교하지 않고 search cost, token usage, latency, exclusion rate를 별도로 보고한다. 어떤 숫자 threshold나 aggregation weight도 결과를 본 뒤 조정하지 않도록 preregister한다.

### 12.5 Experiment contrast audit table

아래 표의 각 행은 treatment, held-constant inputs, supported estimand와 unsupported inference를 명시한다. `Pack-B0`는 위 12.2의 DAG-independent first-pass origin을 따르며, A1 결과를 고정 pack의 원천으로 사용하지 않는다.

| 행 | Treatment/contrast | Held-constant inputs | Supported estimand | Unsupported inference |
|---|---|---|---|---|
| A0 | question-only/direct first-pass retrieval | question, cutoff, source policy, temporal gate, Searcher model/budget, pack schema | direct retrieval이 만드는 Evidence Pack properties | Forecaster reasoning 또는 forecast accuracy |
| A1 | A0 + context/mechanism-aligned eligible-DAG retrieval + gap/counterevidence/open-world pass | A0와 동일한 question/cutoff/source policy/model/budget/schema | two-pass Searcher treatment의 incremental retrieval effect | DAG memory가 reasoning을 원인으로 개선했다는 결론 |
| B0 | `Pack-B0` + no historical memory | byte-identical `Pack-B0`, question, cutoff, Forecaster model/prompt/decoding/schema/budget | fixed evidence에서 memory 부재 기준선 | Searcher quality 또는 total system effect |
| B1 | `Pack-B0` + unstructured raw historical cases | B0와 동일; memory episode/eligibility and token budget preregistered | raw case representation의 conditional effect | DAG structure 자체의 효과 |
| B2 | `Pack-B0` + full/raw eligible resolved DAG + historical outcome (main) | B0와 동일한 pack/model/prompt/cutoff/decoding/schema/budget | full resolved-DAG memory representation의 conditional effect | DAG-guided search effect 또는 economic causal identification |
| B3 | `Pack-B0` + outcome-masked/structural DAG | B2와 동일한 episodes, pack/model/prompt/cutoff/decoding/schema/budget | masking/structural representation diagnostic | main full-DAG claim의 대체 또는 ex-ante ground truth |
| B4 | `Pack-B0` + irrelevant temporally eligible DAG | B0와 동일한 pack/model/prompt/cutoff/decoding/schema/budget; eligibility만 동일 | irrelevant-memory/anchoring diagnostic | irrelevance가 완전한 negative control이라는 주장 |
| C0 | A0 Searcher + B0 Forecaster | model family/version, cutoff policy, budgets, schema, run logging | question-only/no-memory bundle baseline | 어느 한 component의 isolated effect |
| C1 | A1 Searcher + B0 Forecaster | C0와 동일한 model/cutoff/budget/schema; Searcher treatment만 변경 | two-pass/no-memory bundle contrast | A1 retrieval effect와 동일하다는 주장 |
| C2 | A1 Searcher + B2 Forecaster (main bundle) | C0/C1과 동일한 run policy, model/cutoff/schema/budget | proposed full-bundle contrast | total effect을 Searcher 또는 Forecaster 한 component로 귀속 |
| C3 | A1 Searcher + B1/B3/B4 Forecaster representation branch | C2와 동일한 Searcher/model/cutoff/schema/budget | end-to-end representation diagnostic | B의 fixed-pack conditional effect로의 직접 환산 |

---

## 13. 평가 체계

### 13.1 Primary reasoning endpoint: outcome-blind LLM-Judge pairwise preference

본 연구의 primary reasoning endpoint는 **outcome-blind, anonymized, condition/model-identity-blind pairwise preference**다. 각 pair는 같은 질문, cutoff와 허용된 evidence view 아래에서 후보 답변을 `Answer A`와 `Answer B`로 무작위 배치한다. Judge는 realized outcome, post-resolution/hindsight DAG, condition name, model/provider name, run ID, 또는 treatment를 드러내는 answer identity를 볼 수 없다. 출력 길이·섹션 순서·필수 schema·citation 표기를 가능한 한 표준화해 표현 길이만으로 선택되지 않게 한다.

Primary judge에는 질문과 cutoff, 그리고 citation traceability를 검사하는 데 필요한 **중립화된 evidence/citation view**만 제공한다. Historical-memory condition 이름, Searcher branch, model identity와 outcome label은 제거한다. Judge prompt와 payload에는 “정답”, resolution 후 문서, hindsight path, 또는 사후에만 알 수 있는 target-specific fact를 넣지 않는다. 이 원칙은 [LLM-as-a-judge pairwise jury 연구](https://arxiv.org/abs/2404.18796)를 참고한 protocol이며, 해당 연구를 금융 성능의 증거로 해석하지 않는다.

Pair의 A/B 순서와 전체 run 순서를 각각 randomize하고, 같은 pair를 label-swapped order로 다시 판단한다. Swapped judgment는 원래 answer identity로 되돌려 매핑한다. 두 판단이 같은 winner이면 그 preference를 사용하고, 두 판단이 모두 tie이면 tie로 보존한다. winner와 tie가 섞이거나 두 winner가 불일치하면 winner로 강제하지 않고 `inconsistent`로 기록해 primary preference denominator에서 제외하며, 파싱 실패·schema 위반은 `invalid`로 기록해 별도 보고한다. Panel의 기본 aggregation은 유효한 judge preference의 사전 등록된 majority이고, majority가 없거나 유효 preference가 tie뿐이면 overall tie로 남긴다. 이 aggregation·제외 규칙은 run 전에 고정하고 결과를 본 뒤 변경하지 않는다. [position/order effect 연구](https://arxiv.org/abs/2309.00267)를 고려해 order-consistency와 tie/inconsistency rate를 함께 보고한다.

### 13.2 Pairwise rubric과 judge panel

Judge는 다음 차원을 고려해 전체 pairwise preference를 선택한다. 차원별 absolute score와 judge rationale은 실패 분석과 진단용이며 primary endpoint를 대체하거나 임의 가중합한 main score가 아니다.

| 진단 차원 | Judge가 볼 질문 |
|---|---|
| Ex-ante finance-mechanism plausibility | forecast cutoff 당시 알려진 금융 mechanism과 제시된 방향이 그럴듯한가? |
| Evidence grounding / citation traceability | 핵심 주장과 probability rationale이 accepted evidence와 citation으로 추적되는가? |
| Scenario coherence / completeness | scenarios가 서로 구분되고 주요 plausible branch와 residual uncertainty를 다루는가? |
| Counterevidence | 반대 신호와 불리한 evidence를 회피하지 않고 다루는가? |
| Assumptions, triggers, disconfirmers | 관찰과 가정을 구분하고 변경을 확인할 trigger/disconfirming condition을 명시하는가? |
| Uncertainty calibration / qualification | 모르는 부분, conflicting evidence와 확률의 한계를 과장 없이 qualification하는가? |
| Appropriate analog use / limitations | historical DAG node/path analogy를 과도하게 복사하지 않고 mismatch와 한계를 설명하는가? |

Panel은 heterogeneous three-judge configuration을 **provisional, configurable, low-priority engineering choice**로 둔다. Judge panel 구성은 연구 기여가 아니며, 특정 judge brand를 결과에 맞춰 최적화하지 않는다. Judge 수·provider·prompt version, pair aggregation, tie와 invalid 처리, agreement 계산을 preregister하고, inter-judge agreement, order consistency, tie rate, inconsistency/invalid rate를 보고한다. `agreement`가 높다는 이유로 outcome-aware 진실성을 주장하지 않는다. [causal-coverage judge 한계 논의](https://arxiv.org/abs/2603.05167)는 judge rationale/coverage를 독립적인 causal proof로 읽지 않도록 하는 참고 자료다.

### 13.3 Post-resolution realized-path diagnostic

Resolution 후에는 별도의 post-resolution judge가 realized outcome과 hindsight DAG를 볼 수 있다. 이 judge의 목적은 forecast가 당시 제시한 scenario 중 실제 경로의 일부 방향·전환을 어느 정도 **anticipate**했는지 진단하는 것뿐이다. 이 결과는 ex-ante pairwise score를 덮어쓰지 않으며, 실현되지 않았더라도 당시 evidence에 비추어 합리적이었던 대안 scenario를 감점하지 않는다. Hindsight path overlap은 ex-ante ground truth가 아니고, outcome-blind primary endpoint에 입력되지 않는다.

### 13.4 Evidence/DAG perturbation: behavioral audit

Evidence removal, counterevidence substitution, eligible historical-DAG node/path masking 또는 polarity perturbation, irrelevant-but-eligible DAG injection을 수행한다. 출력 probability와 scenario가 사전 등록된 방향성·uncertainty 규칙에 따라 변하는지를 관찰해 **behavioral counterfactual responsiveness와 auditability**를 측정한다. 이 intervention 결과는 내부 모델의 원래 인과 구조나 특정 feature의 정확한 수치 기여를 증명하지 않는다. Perturbation에 둔감하거나 citation과 output이 분리되는 사례는 failure analysis로 보고한다.

### 13.5 Secondary outcome safety/performance endpoints (question type별, 비통합)

Outcome metric은 reasoning endpoint가 아니라 **secondary safety/performance endpoint**다. Binary, MCQ, quantity와 timeframe은 outcome space와 단위가 다르므로 한 점수로 pooling하거나 임의 평균하지 않는다.

| 질문 유형 | Required output contract | Secondary outcome metrics |
|---|---|---|
| **Binary** | 두 outcome에 대한 확률과 선택 outcome | accuracy, Brier score, log score, calibration/reliability |
| **MCQ** | 모든 class에 대한 normalized multiclass probability | top-1 accuracy, multiclass Brier score, multiclass log score |
| **Quantity** | 원래 단위의 predictive quantiles와 median/central estimate | weighted interval score (WIS), original-unit median absolute error |
| **Timeframe** | forecast cutoff부터 resolution까지의 **days**로 encode한 predictive quantiles | WIS, original-date median absolute date error |

Quantity/timeframe quantile output과 WIS protocol은 **proposed preregistered extension**이다. 현재 official WorldReasoner implementation은 이 두 유형에 대해 equally mature한 proper-scoring implementation을 제공하지 않으므로, 구현·분포 가정·interval weighting·invalid quantile 처리 규칙을 main run 전에 등록한다. Binary/MCQ의 outcome score도 primary reasoning preference를 덮어쓰지 않으며, outcome degradation 또는 safety risk를 감시하는 보조 endpoint로 보고한다.

Quantity/timeframe quantile contract의 invalid 상태도 고정한다. Quantile이 non-numeric이거나 허용된 original-unit domain 밖이거나 monotone하지 않으면 schema validation을 실패시킨다. Quantile을 조용히 sort하거나 임의 repair하지 않으며, 동일한 prompt/schema 제약으로 deterministic retry를 한 번만 허용한다. 두 번째 실패는 `invalid forecast`로 표시하고 preregistered missing-output rule에 따라 제외 또는 별도 처리하며, invalid-quantile failure rate를 보고한다.

### 13.6 Retrieval와 output-contract diagnostics

Experiment A에서는 temporal/body-version admission rate, provenance completeness, target relevance, primary-source coverage, citation traceability, counterevidence/novel-factor coverage, mechanism·episode diversity, same-event exclusion audit, pack size·cost·latency를 Evidence Pack properties로 보고한다. Experiment B와 C에서는 scenario coverage, probability normalization, evidence-to-scenario mapping, assumption/trigger/disconfirmers, uncertainty qualification과 analogy limitation을 schema diagnostics로 보고한다. 어떤 diagnostic도 primary pairwise preference나 type-specific outcome score를 사후에 대체하지 않는다.

---

## 14. 핵심 Ablation 및 진단

### 14.1 Searcher treatment ablation

A0 direct first pass와 A1 context + eligible-DAG two pass를 비교한다. Question, cutoff, source policy, temporal/body-version gate, Searcher model/budget와 Evidence Pack schema는 유지하고, Evidence Pack properties만 평가한다. Forecaster 결과를 포함하거나 A의 차이를 forecasting reasoning effect로 번역하지 않는다.

### 14.2 Fixed-pack historical-memory ablation

동일한 byte-identical `Pack-B0`와 동일한 Forecaster model/prompt/cutoff/decoding에서 B0 no memory, B1 raw cases, B2 full/raw eligible resolved DAG + outcome (main), B3 outcome-masked/structural DAG, B4 irrelevant but temporally eligible DAG를 비교한다. B2 main condition을 B3 structural view로 바꾸지 않으며, B4의 irrelevance는 eligibility와 별도로 기록한다. 이 ablation의 estimand는 memory representation의 conditional effect이고, DAG-guided search의 효과나 경제적 인과 식별이 아니다.

### 14.3 End-to-end branch ablation

C0 question-only + no memory, C1 two-pass + no memory, C2 two-pass + full DAG, C3 two-pass + raw/masked/irrelevant representation을 동일한 run policy에서 실행한다. 이 branch는 실제 bundle의 품질·비용·failure mode를 비교하는 용도이며, C2와 C0의 차이를 Searcher 또는 Forecaster 한 component의 효과라고 보고하지 않는다. Component attribution은 A와 B의 별도 contrast에서만 인용한다.

### 14.4 Counterevidence 및 open-world branch

Two-pass Searcher에서 counterevidence branch 또는 open-world branch를 하나씩 제거하고, Evidence Pack의 반대 신호·novel-factor coverage와 end-to-end scenario collapse를 함께 기록한다. Branch 제거는 search treatment를 바꾸므로 fixed-pack B 결과와 혼합하지 않는다.

### 14.5 Historical-memory behavioral audit

Accepted evidence를 제거하거나 반대 신호로 교체하고, historical DAG의 cited node/path를 mask·polarity perturbation하거나 irrelevant-but-eligible DAG를 주입한다. Scenario probability, triggers/disconfirmers, uncertainty qualification이 방향성에 맞게 바뀌는지를 확인한다. 이 ablation은 behavioral counterfactual responsiveness/auditability를 평가하며, 출력 설명이 원래 내부 계산을 인과적으로 증명한다고 해석하지 않는다.

### 14.6 Judge robustness controls

Pairwise label order와 run order를 swap하고, output length/schema를 normalize한 뒤 preference, tie, inconsistency, order-consistency를 재집계한다. Judge panel brand를 결과에 맞춰 선택하거나 absolute-rubric 진단 점수를 primary endpoint로 승격하지 않는다.

---

## 15. Intended research contributions

이 절의 내용은 실험 전의 **proposed/intended contribution**이다. 아직 실행된 실험, effect size, 성능 결과 또는 완성된 데이터셋을 보고하지 않는다. WorldReasoner의 generic capability와 우리가 금융 연구에서 추가하려는 설계를 분리해 읽어야 한다.

### 15.1 Finance-only resolved-DAG memory

첫 번째 의도는 WorldReasoner native finance pipeline을 금융 질문에 한정해 운영하고, v1.0에 이미 built graph가 있는 37개 finance question을 pilot/seed로 감사하는 것이다. 공식 release는 37개를 23 binary, 1 MCQ, 8 quantity, 5 timeframe으로 열거한다([WorldReasoner v1.0.0 release](https://github.com/cyzus/worldreasoner/releases/tag/v1.0.0)). 이후 질문은 **question → timestamped evidence → GraphBuilder → resolved historical episode** 순서로 확장한다. Main memory item은 cutoff 전에 resolution을 사용할 수 있게 된 episode의 raw/full resolved historical DAG, historical outcome, provenance와 episode metadata이다. 37개 seed나 이후 확장이 연구 가설을 지지한다는 결과는 아직 주장하지 않는다.

WorldReasoner repository와 paper는 temporal search, forecasting, graph construction 및 evaluation을 generic framework capability로 제시한다([repository](https://github.com/cyzus/worldreasoner), [paper](https://arxiv.org/abs/2606.11816)). 금융-specific novelty는 그 일반 기능을 새로 발명했다고 주장하는 데 있지 않고, finance-only question selection, temporal eligibility, resolved-DAG memory retrieval/use, 금융 시나리오 reasoning 및 type-specific evaluation 계약을 하나의 연구로 묶는 데 있다. Pinned finance configuration과 temporal gateway 문서는 generic pipeline의 경계를 확인하는 참고 자료다([pinned config](https://github.com/cyzus/worldreasoner/blob/6fd8ed1208f45290b017160edc8f928f45336785/src/config/pipeline.py#L10-L24), [temporal documentation](https://github.com/cyzus/worldreasoner/blob/6fd8ed1208f45290b017160edc8f928f45336785/docs/04_forecasting.md#L89-L99)).

### 15.2 Two-pass Searcher와 non-graph-building Forecaster

두 번째 의도는 **Searcher만** live web을 사용하도록 하고, 첫 pass와 두 번째 pass의 목적을 분리하는 것이다. Searcher는 question/cutoff에서 Target Profile을 만든다. 이어서 historical DAG와 독립적인 direct/open search, temporal/body-version admission, flat Current Context Profile, hard-eligible historical-DAG retrieval, context/mechanism-aware diversity selection, gap/counterevidence/open-world second pass를 순서대로 수행한다. 최종 산출물은 Evidence Pack과 raw eligible historical DAG references이다.

Forecaster는 Searcher가 공급한 filtered Evidence Pack과 raw/full eligible historical DAG, historical outcome metadata, outcome space와 domain constraints만 읽는다. Forecaster는 web을 검색하거나 현재 target question의 graph artifact를 만들지 않는다. 출력은 mutually intelligible scenarios, scenario/outcome probabilities, evidence citations, uncertainty/disconfirmers, explicit historical node/path mappings와 analogy-fit limitations이다. 이 경계와 A/B/C component estimands는 Sections 6, 12–14의 계약을 연구 protocol로 구체화한다.

### 15.3 Strict temporal admission without a separate web backend

세 번째 의도는 historical backtest evidence를 publication date 하나로 통과시키지 않는 것이다. Main experiment에는 exact pre-cutoff body version이 archive 또는 versioned copy로 독립 검증되고 body equality/hash와 provenance가 남은 item만 admission한다. 검증할 수 없는 body는 exclusion reason과 함께 제외한다. Current unresolved questions는 retrieval-time snapshot/hash를 남기되 ground truth를 submission 전에 주지 않는다. 별도의 고정형 web database나 retrieval backend는 연구 요구사항으로 만들지 않는다. 따라서 retrospective 결과는 `temporally filtered` 또는 `leakage-minimized`라고만 부르고, fully leakage-free라고 부르지 않는다.

### 15.4 Outcome-blind reasoning evaluation with type-specific safeguards

네 번째 의도는 **outcome-blind anonymized pairwise LLM-Judge preference**를 primary reasoning endpoint로 두고, absolute rubric, post-resolution realized-path diagnostic, perturbation audit를 진단으로 분리하는 것이다. Binary, MCQ, quantity, timeframe의 outcome metrics는 서로 다른 outcome space와 단위를 보존한 채 별도로 보고한다. Binary/MCQ의 accuracy, Brier, log score, calibration과 quantity/timeframe의 proposed quantile/WIS protocol은 reasoning endpoint를 대체하지 않는 secondary safety/performance checks이다. Quantity/timeframe extension은 [WorldReasoner evaluation protocol](https://github.com/cyzus/worldreasoner/blob/main/docs/05_evaluation.md)을 기준으로 현재 구현 범위를 확인한 뒤 preregister할 제안이지, 이미 제공되는 완성된 scoring result가 아니다. A는 Searcher retrieval estimand, B는 동일 fixed Evidence Pack에서 memory representation의 conditional estimand, C는 전체 bundle contrast를 식별하도록 설계한다. 이 네 가지 기여는 모두 preregistration과 실험을 거친 뒤에야 지지 여부를 평가할 수 있다.

---

## 16. Non-claims

이 연구가 다루는 것은 검토 가능한 ex-ante reasoning과 outcome safety이지, 다음과 같은 강한 주장이 아니다.

- Historical DAG의 edge가 실제 경제적 인과를 식별하거나, 각 edge가 true causal effect임을 증명한다고 주장하지 않는다.
- 어떤 probability의 각 소수점이 특정 원인에 기여한 정확한 수치라고 주장하지 않으며, 모델 내부 causal mechanism을 입증한다고 주장하지 않는다.
- Retrospective live-web backtest를 leakage-free 또는 zero-leakage라고 부르지 않는다. 현재 검색 ranking, snippet, 모델의 parametric knowledge와 body-version 불확실성은 잔여 한계로 남는다.
- Searcher나 Forecaster가 현재 target question을 표현하는 graph를 만들지 않는다. Historical offline DAG를 읽고 참조하는 것과 새로운 target graph를 산출하는 것은 다른 계약이다.
- Reasoning contract, citation traceability 또는 judge preference가 결과의 정확성을 보장한다고 주장하지 않는다. 어떤 agent도 correct outcome을 보장하지 않는다.
- Accuracy, Brier, log score, calibration 또는 WIS를 primary scientific claim으로 삼지 않으며, incompatible question types를 한 점수로 합치지 않는다.
- 아직 실행하지 않은 37-seed audit, finance-only expansion, effect, sample, threshold 또는 forecast result를 완료된 empirical finding처럼 쓰지 않는다.
- Manual annotation, manual pairwise evaluation 또는 public database release는 이 연구의 필수 조건이 아니다. 향후 공개 가능한 artifact는 선택 사항이며 licensing과 privacy 검토를 통과할 때만 제안한다.

이 non-claims는 source-grounded hypothesis, behavioral audit와 ex-ante judgment를 경제적 causal identification이나 hindsight truth로 과장하지 않게 하는 해석 경계다.

---

## 17. Dependency-ordered research work packages

Work package는 구현 제품의 목록이 아니라, 위 연구 질문을 검증하기 위한 순서와 산출물을 정의한다. 각 단계의 exit criterion을 만족하기 전에는 다음 단계의 비교를 주 결과로 해석하지 않는다.

### WP1. Preregistration and data audit

- locked claim, primary/secondary estimands, question-type split, cutoff policy, exclusion rules, model/API version policy와 artifact schema를 preregister한다.
- 37-seed inventory, source/provenance completeness, resolution availability, question-type labels와 duplicate/event relation을 audit한다.
- **Exit:** audit table, exclusion-reason vocabulary, preregistered contrasts와 temporal admission test가 versioned artifact로 검토 가능하다.

### WP2. Finance-only seed audit and native expansion

- WorldReasoner v1.0 finance seed를 재현 가능한 native question → evidence → GraphBuilder flow로 점검한다.
- seed 품질 gate를 통과한 뒤 finance-only questions를 추가하고 raw/full resolved DAG, historical outcome, body provenance와 episode metadata를 보존한다.
- outcome-masked/structural representation은 별도 ablation artifact로 만들고 main raw memory를 대체하지 않는다.
- **Exit:** question-type별 seed/expansion inventory, resolution and source-availability audit, GraphBuilder quality report와 growth rule이 있다.

### WP3. Temporal Searcher

- first-pass direct/open search를 독립적으로 실행하고, exact body-version gate를 통과한 flat Current Context Profile을 만든다.
- hard eligibility와 same-event/shared-resolution/derived/near-duplicate exclusion 뒤 context/mechanism-aware diverse retrieval을 수행한다.
- gap, counterevidence, open-world second pass를 실행해 Evidence Pack과 audit log를 반환한다. Searcher는 preferred outcome이나 probability를 반환하지 않는다.
- **Exit:** A0/A1 contracts, admission/exclusion logs, body hash/availability checks, cost and failure policy가 deterministic replay에 충분하다.

### WP4. Non-graph-building Forecaster

- fixed Evidence Pack과 raw eligible historical DAG를 입력으로 받는 B0–B4 branch를 구현·검증한다.
- scenario, outcome distribution, evidence citation, uncertainty/disconfirmers, historical node/path mapping과 analogy limitation을 schema로 검사한다.
- web access, target graph construction, unlogged evidence substitution을 금지하고 abstention/invalid output을 기록한다.
- **Exit:** identical Pack-B0 replay, output-contract validation, probability normalization trace와 representation-branch manifest가 완성된다.

### WP5. A/B/C experiments and LLM-Judge evaluation

- A는 question-only/direct retrieval 대 context + eligible-DAG two-pass retrieval을 비교한다.
- B는 byte-identical Pack-B0에서 no memory, raw case, full/raw DAG + historical outcome, outcome-masked/structural, irrelevant eligible DAG를 비교한다.
- C는 Searcher와 Forecaster를 함께 실행해 full bundle을 평가하되 component effect를 C 결과에 귀속하지 않는다.
- primary outcome-blind randomized/order-swapped pairwise LLM-Judge를 실행하고 absolute rubric, agreement, tie/inconsistency 및 post-resolution diagnostic을 별도로 기록한다.
- **Exit:** 각 contrast에 treatment, held-constant input, supported estimand와 invalid/exclusion rule이 있고, 결과를 보기 전에 analysis script와 tests가 고정된다.

### WP6. Prospective validation and paper artifacts

- strongest clean validation으로 prospective unresolved questions를 동일한 Searcher-only boundary에서 평가한다.
- temporally filtered retrospective 결과, type-specific secondary outcomes, cost, failure cases와 residual leakage를 분리해 분석한다.
- paper tables/figures, preregistration, provenance/audit data card, reproducibility manifest와 licensing note를 만든다.
- **Exit:** manuscript claim이 primary reasoning result와 secondary safety evidence를 혼동하지 않고, 공개 artifact 범위가 명시된다.

별도의 public web database, retrieval index 또는 manual annotation campaign은 이 work package의 required deliverable이 아니다. 향후 선택적 release artifact가 필요하면 WP6에서 별도 licensing decision으로 다루며 core study scope를 넓히지 않는다.

---

## 18. Roles and collaboration boundaries

역할은 사람의 직함보다 책임 경계로 정의한다. WorldReasoner 팀은 generic pipeline과 graph/retrieval 기술을 제공하고, 금융 연구진은 질문·도메인·source validity·평가 설계를 소유한다. Preregistration, experiment execution, analysis와 manuscript는 공동 책임이다.

| 책임 영역 | WorldReasoner 팀 | 금융 연구진 | 공동 산출물 |
|---|---|---|---|
| generic WR pipeline, GraphBuilder와 retrieval interfaces | native pipeline, config, retrieval/GraphBuilder expertise | finance-only adaptation 요구사항 | versioned schema와 integration test |
| 37-seed audit와 finance-only question expansion | pipeline reproducibility, graph quality tooling | question selection, outcome space, resolution rule, domain coverage | seed/expansion audit와 exclusion log |
| source/body-version validity와 temporal admission | gateway integration, provenance implementation | source-of-record 판단, pre-cutoff body verification 기준 | admission policy와 audit report |
| historical DAG interpretation | node/path representation, retrieval alignment | financial mechanism, actor/entity, unit/range와 analogy limitation | quality rubric와 disagreement log |
| Searcher/Forecaster experiment | agent contracts, replay and cost instrumentation | target schema, counterevidence policy, domain constraints | A/B/C preregistration과 run manifest |
| evaluation design | LLM-Judge orchestration, anonymization, order randomization | reasoning dimensions, type-specific outcome contracts, harm/non-inferiority rationale | primary/secondary analysis plan |
| analysis and paper | pipeline diagnostics, reproducibility and artifact packaging | interpretation, limitations, finance relevance | joint manuscript and release note |

Manual expert judging or annotation is not mandatory for any row. Domain input is needed to define questions, source validity and interpretation—not to turn manual pairwise judgment into a hidden primary endpoint.

---

## 19. Phases and milestone exit criteria

날짜나 예상 효과를 미리 제시하지 않는다. 다음 순서는 dependency와 exit criterion을 명시하는 milestone plan이다.

### Phase 0. Audit and preregistration

**Work:** 37 seed inventory, question-type labels, source/resolution timestamps, duplicate/event relation, primary/secondary estimands, A/B/C contrasts와 temporal/body-version policy를 preregister한다.

**Exit:** 모든 locked contract가 versioned protocol과 schema에 연결되고, audit에서 unresolved ambiguity와 exclusion reason이 명시된다.

### Phase 1. Seed quality gate

**Work:** 37 finance seed를 native pipeline으로 재실행 가능한 형태로 audit하고 GraphBuilder output, source grounding, resolution availability, node/path provenance와 quality edge cases를 검토한다.

**Exit:** seed를 pilot/seed로 사용할 수 있는 범위, 제외할 item, expansion에서 보완할 field가 서면으로 고정된다. 결과 품질이나 성능을 아직 주장하지 않는다.

### Phase 2. Finance-only expansion

**Work:** seed audit에서 정한 규칙으로 finance-only question → evidence → GraphBuilder expansion을 수행하고 type/domain coverage와 raw resolved DAG metadata를 기록한다.

**Exit:** growth target, resolution/source audit, same-event/shared-resolution/derived/near-duplicate cluster와 quality report가 analysis split 전에 freeze된다.

### Phase 3. Searcher contract

**Work:** A0 direct first pass와 A1 context/mechanism + eligible-DAG two pass를 구현·replay한다. Body-version gate, hard exclusion, Current Context Profile, counterevidence/open-world branch와 Evidence Pack hash를 검증한다.

**Exit:** A0/A1의 held-constant inputs와 Searcher estimand가 식별되고, admission/exclusion/cancellation/retry 로그를 deterministic하게 재현할 수 있다.

### Phase 4. Forecaster contract

**Work:** fixed Pack-B0에서 B0–B4 representation conditions를 실행한다. Forecaster output에는 supplied filtered evidence와 raw eligible historical DAG mapping만 허용하며 web access나 target graph artifact는 허용하지 않는다.

**Exit:** schema validation, scenario aggregation, probability normalization, uncertainty/analogy limitation과 invalid-output policy가 test artifact로 확인된다.

### Phase 5. Temporally filtered retrospective evaluation

**Work:** A/B/C preregistered conditions를 simulated cutoff에서 실행한다. Exact pre-cutoff body version을 독립 검증할 수 있는 evidence만 main retrospective pack에 admission하고, current ranking/snippet hindsight 및 unresolved parametric contamination을 limitations로 기록한다.

**Exit:** temporal exclusion audit, primary outcome-blind pairwise results, type-specific secondary outcomes, perturbation diagnostics와 cost/failure log가 함께 존재한다. Retrospective result는 `temporally filtered` 또는 `leakage-minimized`로만 labeled된다.

### Phase 6. Prospective strongest validation

**Work:** 아직 해결되지 않은 prospective questions에 동일한 Searcher-only live-web boundary를 적용하고, submission 전에 ground truth를 노출하지 않는다. Retrieval-time snapshot/hash와 provenance를 남긴다.

**Exit:** prospective run이 retrospective body-version assumptions와 분리된 clean-validation artifact를 제공하고, outcome realization 전 primary ex-ante judgments가 보존된다.

### Phase 7. Analysis and manuscript

**Work:** primary reasoning claim, type-specific safety outcomes, A/B component estimands, C bundle result, post-resolution/behavioral diagnostics, limitations와 licensing을 통합한다.

**Exit:** manuscript의 모든 empirical 문장이 결과 artifact 또는 proposed protocol로 표시되고, generic WR capability와 finance-specific claim이 혼동되지 않는다.

---

## 20. Success criteria

성공은 사전 등록된 reasoning improvement와 안전성·감사 가능성의 조합으로 판단한다. 특정 숫자 threshold, effect size, sample size 또는 judge agreement 수준을 이 문서에서 발명하지 않는다.

1. **Primary outcome-blind reasoning:** 동일한 question/cutoff/evidence view에서 primary anonymized pairwise LLM-Judge가 선택한 ex-ante justification이 preregistered baseline 대비 개선되는지 확인한다. A/B/C의 treatment와 held-constant inputs를 분리해 보고하고, tie, order inconsistency와 invalid output을 숨기지 않는다.
2. **Type-specific secondary safety/performance:** binary는 accuracy/Brier/log score/calibration, MCQ는 top-1/multiclass Brier/multiclass log score, quantity와 timeframe은 proposed predictive-quantile WIS와 original-unit/date error를 각각 보고한다. Outcome non-inferiority margin과 harm criterion은 main run 전 future preregistration에서 정하고, incompatible types를 pooling하지 않는다.
3. **Identifiable component estimands:** A는 direct first pass 대비 two-pass Searcher retrieval의 incremental Evidence Pack effect, B는 byte-identical Pack-B0에서 raw/full eligible DAG memory representation의 conditional Forecaster effect, C는 전체 bundle contrast로 해석한다. C의 total contrast를 Searcher 또는 Forecaster 하나의 effect로 재명명하지 않는다.
4. **Temporal exclusion audit:** admitted historical evidence는 exact pre-cutoff body version, body hash/equality, availability와 provenance를 갖고, unverifiable body는 제외 사유가 있다. Historical DAG는 `resolution_available_at < cutoff`와 source availability를 충족하며 same-event/shared-resolution/derived/near-duplicate가 ranking 전에 hard-exclude된다.
5. **Contract and auditability:** Evidence Pack, scenario/probability, citations, historical node/path mappings, uncertainty/disconfirmers와 analogy limitations가 schema 및 replay log로 연결된다. Perturbation 결과는 behavioral responsiveness/auditability로만 해석한다.
6. **Honest interpretation:** retrospective findings는 temporally filtered/leakage-minimized로 라벨하고 residual live-ranking, snippet, parametric contamination과 body attrition을 보고한다. Prospective evaluation을 strongest clean validation으로 별도 제시한다.

Primary reasoning improvement가 관찰되지 않아도, 잘 정의된 null result와 실패·비용·temporal exclusion report는 protocol 성공 조건을 충족할 수 있다. 반대로 outcome score가 좋아도 outcome-blind reasoning, provenance 또는 temporal audit을 통과하지 못하면 핵심 연구 성공으로 간주하지 않는다.

---

## 21. Risks and mitigations

| 위험 | 연구에 미치는 영향 | locked pipeline에 맞는 대응 |
|---|---|---|
| Body-version attrition | 과거 문서가 cutoff 이전 body로 검증되지 않아 usable evidence가 줄어듦 | publication date만으로 admission하지 않고 exact pre-cutoff body/version/hash를 검증한다. 검증 불가 item은 reason과 함께 제외하고 attrition을 보고한다. 별도 fixed web backend로 보충하지 않는다. |
| Live ranking hindsight | 현재 ranking, ordering 또는 popularity가 later event signal을 반영할 수 있음 | ranking/snippet은 evidence로 쓰지 않고 residual bias로 기록한다. retrospective를 leakage-free라고 부르지 않으며 prospective validation을 strongest clean check로 둔다. |
| Parametric contamination | model이 cutoff 이후 사실을 이미 알고 있을 수 있음 | model/API/version과 knowledge-cutoff policy를 preregister하고, evidence-only comparison과 prospective track을 분리한다. 완전 제거를 주장하지 않는다. |
| Analogy shortcut or outcome copying | historical outcome이나 표면 유사성이 current scenario를 자동으로 결정할 수 있음 | main raw DAG의 historical outcome은 명시적으로 label하고, masked/structural/raw-case/irrelevant-DAG ablation, counterevidence/open-world search, node/path mapping과 analogy limitation을 함께 평가한다. |
| Judge bias and position bias | 표현 길이, label order, model identity가 primary preference를 왜곡할 수 있음 | outcome/condition/model identity를 가리고 A/B order와 run order를 randomize한다. label-swapped repeat, tie/inconsistency/invalid 기록, absolute rubric diagnostics와 provisional panel agreement를 보고한다. |
| Type imbalance and small 37-seed pilot | seed의 question-type/domain 분포가 확장 전 불균형할 수 있음 | 37을 pilot/seed로만 해석하고 finance-only expansion 및 growth target을 preregister한다. binary/MCQ/quantity/timeframe metrics를 분리하고 pooled score를 만들지 않는다. |
| DAG quality and causal overclaim | source 없는 node/path, 선후관계를 causal edge로 과대 해석할 위험 | source/body provenance, epistemic status, event time, GraphBuilder quality audit를 통과시키고 graph를 causal identification으로 해석하지 않는다. |
| Retrieval anchoring and novelty loss | historical mechanism이 현재 novelty 또는 반대 신호를 가릴 수 있음 | first pass를 DAG-independent로 유지하고, diversity, open-world branch, counterevidence branch와 missing/uncertainty slot을 보존한다. |
| Compute/API reproducibility | provider 변경, timeout, cancellation, rate limit이 비교를 흔들 수 있음 | exact model/API/version, prompt, decoding, budget, seed, retry/cancellation policy, body hash, run manifest와 invalid log를 preregister하고 replay artifact를 보존한다. |

어떤 mitigation도 “leakage가 0” 또는 “reasoning이 내부적으로 causal”이라고 보증하지 않는다. 이 표의 목적은 residual risk를 측정·공개할 수 있게 하는 것이다.

---

## 22. Locked decisions versus remaining implementation parameters

다음 핵심 선택은 interview에서 이미 **locked** 되었다. 논문을 쓰는 동안 다시 열어 두지 않으며, remaining items는 구현·분석 파라미터일 뿐 연구 방향을 바꾸는 선택이 아니다.

### 22.1 Locked core decisions

| Locked decision | Exact contract |
|---|---|
| Finance-only WorldReasoner-native source | 주 데이터는 WorldReasoner native finance questions와 pipeline이다. v1.0의 built-graph finance questions **37개**를 pilot/seed로 audit하고 finance-only expansion을 한다. legacy local collections는 optional external validation일 뿐 main benchmark가 아니다. |
| Raw resolved DAG as main memory | Main Forecaster condition은 eligible **full/raw resolved historical DAG + historical outcome + episode metadata**를 그대로 받는다. |
| Structural/masked views | Outcome-masked, structural abstraction, domain template와 raw-case view는 ablation/diagnostic only이며 main raw memory를 대체하지 않는다. |
| Searcher-only web and two passes | Searcher가 first-pass DAG-independent direct/open search → temporal/body-version gate → flat Current Context Profile → hard-eligible diverse DAG retrieval → gap/counterevidence/open-world second pass를 수행한다. |
| Forecaster boundary | Forecaster는 supplied filtered Evidence Pack과 raw eligible historical DAG만 사용하고 web/search/ranking에 접근하지 않으며, current target에 대한 graph artifact를 생성하지 않는다. |
| No separate fixed backend | 별도의 fixed web corpus, fixed web database 또는 retrieval index를 연구 요구사항으로 만들지 않는다. Historical admission은 independently verified exact pre-cutoff body에만 허용한다. |
| Temporal admission | Current unresolved track은 retrieval-time snapshot/hash와 provenance를 남긴다. Historical backtest는 simulated cutoff를 사용하고 exact pre-cutoff body version이 검증되지 않으면 main experiment에서 제외한다. Historical DAG eligibility는 `resolution_available_at < cutoff`이며 same-event/shared-resolution/derived/near-duplicate는 hard exclusion이다. |
| Primary reasoning endpoint | Outcome-blind, anonymized, condition/model-identity-blind pairwise LLM-Judge preference가 primary다. Outcome-aware post-resolution judge와 absolute rubric은 별도 diagnostic이다. |
| Judge panel status | Heterogeneous three-judge panel은 provisional, configurable, low-priority engineering choice다. Panel 구성은 novelty claim이 아니며 agreement/tie/inconsistency를 보고한다. |
| Question-type reporting | Binary, MCQ, quantity, timeframe을 모두 포함하되 secondary metrics는 type별로 분리한다. Quantity/timeframe quantile/WIS protocol은 proposed preregistered extension이다. |

### 22.2 Remaining implementation parameters

Preregistration에서만 정할 항목은 다음으로 제한한다.

- outcome non-inferiority margin, harm margin/criterion, missing-output and invalid-output handling
- exact Forecaster/Searcher/Judge models, providers, API versions, prompts, decoding, knowledge-cutoff declaration and reproducibility manifest
- retrieval `top-k`, candidate budget, diversity rule, same-event cluster threshold and source policy
- Judge rubric wording/thresholds, tie and inconsistency aggregation, panel agreement statistic and planned statistical tests
- sample growth targets after the 37-seed audit, domain/type quotas, compute/API budget, timeout/retry policy
- artifact retention, licensing, privacy and any optional public release scope

이 목록 바깥의 core choice를 “향후 meeting decision”으로 되돌리지 않는다. Parameter 값은 실험 결과를 본 뒤 조정하지 않고 preregistration version에 고정한다.

---

## 23. Positioning and related work

WorldReasoner는 이 연구의 generic base다. 공식 repository, v1.0 release와 paper는 temporal forecasting, evidence/reasoning workflow와 선택적 graph-related capability를 제공하는 framework로 설명한다([repository](https://github.com/cyzus/worldreasoner), [release](https://github.com/cyzus/worldreasoner/releases/tag/v1.0.0), [paper](https://arxiv.org/abs/2606.11816)). Pinned finance config와 temporal documentation은 native pipeline과 cutoff-aware access 경계를 확인하는 primary artifacts다([config](https://github.com/cyzus/worldreasoner/blob/6fd8ed1208f45290b017160edc8f928f45336785/src/config/pipeline.py#L10-L24), [temporal docs](https://github.com/cyzus/worldreasoner/blob/6fd8ed1208f45290b017160edc8f928f45336785/docs/04_forecasting.md#L89-L99)). 우리는 이러한 generic capability를 새로 발명했다고 주장하지 않는다.

최근의 [Analogical Deep Research preprint](https://arxiv.org/abs/2607.13602)는 mechanism-aligned analogy와 research reasoning에 인접한 비교점이다. 본 연구는 그 방향과 경쟁 우선순위를 주장하기보다, 금융 질문에 대해 (i) probabilistic outcome/scenario forecasting, (ii) strict temporal/body-version access control, (iii) raw resolved historical-DAG memory와 eligibility, (iv) Searcher-only web access와 non-graph-building Forecaster separation, (v) outcome-blind reasoning-centered evaluation을 함께 검증하려는 확장으로 위치시킨다. 어떤 방법이 더 일반적이거나 먼저 제안되었다는 priority/novelty claim은 이 문서의 결론이 아니다.

Working paper의 한 문장 포지셔닝은 다음과 같다.

> We propose a finance-only WorldReasoner extension in which temporally eligible resolved historical DAGs support Searcher retrieval and Forecaster scenario reasoning, while outcome-blind reasoning evaluation and type-specific outcome safeguards keep the claim ex ante and auditable.

---

## 24. Conclusion

이 연구가 제안하는 핵심은 금융 질문에서 **outcome-blind ex-ante reasoning quality를 주 평가로 삼고, outcome metrics를 type-specific safety checks로 남기는 것**이다. 데이터 출발점은 WorldReasoner v1.0의 37 finance graph-built questions를 pilot/seed로 삼은 finance-only native expansion이다. Offline asset은 cutoff와 resolution/provenance가 감사된 raw resolved historical DAG memory다.

Online sequence는 고정되어 있다. Searcher가 question/cutoff에서 시작해 DAG-independent first pass, temporal/body-version admission, flat Current Context Profile, hard-eligible diverse historical-DAG retrieval, gap/counterevidence/open-world second pass를 수행하고 Evidence Pack과 raw eligible historical DAGs를 전달한다. Forecaster는 **supplied filtered evidence와 raw eligible historical DAGs를 입력으로 받아** scenarios, probabilities, citations, uncertainty/disconfirmers, historical node/path mappings와 analogy limitations를 출력한다. Forecaster는 web에 접근하지 않고, 현재 target에 대한 graph를 만들지 않는다.

Historical backtest는 independently verifiable exact pre-cutoff body version만 admission하는 `temporally filtered`/`leakage-minimized` protocol이며, 별도의 fixed web backend가 필요하지 않다. Prospective evaluation은 residual ranking and parametric limitations를 가진 retrospective track보다 강한 clean validation으로 별도 보고한다. Primary comparison은 outcome-blind pairwise LLM-Judge preference이고, A/B/C contrasts, type-specific outcomes, perturbation와 post-resolution diagnostics는 그 주장을 보완한다.

따라서 이 연구는 target-question graph output이나 graph-matching/edit-distance evaluation을 목표로 하지 않는다. Historical offline DAGs는 재사용 가능한 memory와 명시적 node/path mapping의 근거이고, 현재 질문의 output은 scenario/probability contract다. 이 구분과 locked temporal boundary가 지켜질 때, WorldReasoner의 generic foundation 위에 finance-specific reasoning transfer와 감사 가능한 평가를 제안할 수 있다. 아직 실험 전인 문서이므로 최종 contribution과 성능은 위 phases와 preregistered analyses가 완료된 뒤 판단한다.
