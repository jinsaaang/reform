# Hindsight-Guided Forecasting

이 문서는 현재 연구에서 사용하는 HGF 방법론과 실행 계약의 유일한
기준 문서다. 실험용 변형에서 얻은 아이디어는 이 문서에 반영되기 전까지
canonical 방법으로 간주하지 않는다.

## 연구 목적

반복적으로 발생하는 금융 이벤트는 같은 예측 목표와 비슷한 작동 구조를
공유하지만, 매번 관측되는 근거와 시장 상태는 달라진다. HGF는 과거 사건의
정답이나 확률을 복사하지 않는다. 과거 사건이 해결된 뒤 확인할 수 있었던
추론 구조를 현재 사건에서 다시 검사하여, LLM이 현재 근거를 더 구조적으로
해석하도록 한다.

핵심 가설은 단순하다. 과거 사건은 현재 답의 prior가 아니라 현재 증거를
어떤 순서와 관계로 확인할지 알려 주는 구조적 memory다.

## 오프라인 Blueprint 구축

Blueprint는 live forecasting 중에 생성하지 않는다. 과거에 해결된 금융
이벤트에 대해 WorldReasoner가 사후 자료를 사용해 만든 DAG가 출발점이다.
중립화 단계의 LLM은 node와 edge의 설명만 미래 사건에서도 사용할 수 있는
조건문으로 바꾼다. 이 모델은 path, node ID, edge ID, 방향, 관계, 순서를
생성하지 않는다. 이후 deterministic compiler가 원본 DAG에서 해당 topology를
그대로 다시 결합한다. 따라서 문장을 중립화하는 과정에서 graph 구조가 바뀔
수 없다. 이 과정은 다음 정보를 보존한다.

- 예측 요인의 역할과 node identity
- node 사이의 방향을 가진 edge
- root에서 target까지 이어지는 완전한 path
- 관계의 방향과 유형, 사용 가능한 support 정보와 contradiction condition
- 각 node와 edge가 요구하는 evidence type

원본 Blueprint bank는 `artifacts/hgf/blueprints`에 저장한다. Forecaster에
전달되는 중립화된 topology bank는
`artifacts/neutral_topology_templates`에 별도로 고정한다. 두 manifest는 source
DAG hash, compiler identity, artifact hash, topology coverage, acyclicity,
outcome leakage 검사를 기록한다. 이 bank는 forecasting model마다 다시 만들지
않고 모든 모델과 seed에서 동일한 연구 artifact로 재사용한다.

## Live forecasting 파이프라인

HGF의 예측은 하나의 end to end 흐름으로 실행한다.

1. 현재 질문의 metric, period, unit, comparison rule, option boundary를
   target contract로 고정한다.
2. Forecast cutoff 이전의 현재 evidence를 읽고 baseline, driver, timing,
   missing information을 포함한 evidence ledger를 만든다. 이 단계는 과거
   graph와 answer option을 보지 않는다.
3. 같은 event family와 target metric에 속하고 시간상 사용 가능한
   Blueprint들을 검색한다. 현재 evidence와 관련된 최대 세 개의 완전한
   root to target subgraph를 선택한다.
4. 선택한 subgraph의 topology를 바꾸지 않은 채 각 node와 edge를 현재
   evidence로 채운다. Node에는 current state, timing, evidence와 confidence를
   기록한다. Edge에는 preserved, reversed, contradicted, unverified 상태와
   current relation, lag assessment, evidence와 confidence를 기록한다.
5. 현재 상태가 채워진 graph를 따라 baseline, driver, mechanism,
   counterevidence, target bridge 순서로 reasoning을 작성한다. 현재 evidence가
   지지하지 않는 path는 기각하거나 uncertainty로 남긴다. Graph 밖에서만
   관측되는 현재 요인도 별도로 허용한다.
6. 별도 boundary mapper가 완성된 reasoning을 정확한 target unit과 option
   boundary에 대응시켜 하나의 probability distribution을 출력한다.

Blueprint는 고정된 과거 구조이고, current instantiated graph는 문항마다
새로 생성되는 현재 상태다. Historical answer, historical probability,
다른 방법의 prediction은 어떤 단계에도 입력하지 않는다.

## 정보 흐름 계약

- Blueprint retrieval과 subgraph routing은 final answer를 보지 않는다.
- Current evidence는 cutoff 이후 문서를 포함하지 않는다.
- 과거 topology는 current fact로 인용할 수 없다.
- Current factual claim은 현재 evidence ID를 인용해야 한다.
- Reasoning 단계는 option이나 probability를 출력하지 않는다.
- Probability는 마지막 boundary 단계에서 한 번만 생성한다.
- Probability pooling, posterior adjustment, modal boosting, 다른 실행의
  prediction 재사용을 금지한다.
- Worked historical answer나 resolved outcome을 forecaster에 제공하지 않는다.

## 출력 완결성과 기록

Provider가 hidden reasoning을 반환했더라도 forecaster가 실제로 소비한
structured reasoning이 비어 있으면 완결된 실행으로 간주하지 않는다. Raw
provider reasoning은 감사 목적으로 보존하되, prediction에 사용된 reasoning과
명확히 분리한다.

각 문항은 다음 정보를 저장해야 한다.

- 모든 model request와 수정되지 않은 raw response
- 실제 provider와 returned model identifier
- content, provider reasoning, reasoning details, finish reason
- stage별 token, cost, elapsed time, retry와 repair 기록
- 공급된 evidence ID와 reasoning 및 forecast가 실제 인용한 evidence ID
- 검색한 Blueprint ID와 실제 사용한 path, node, edge
- ledger, graph, reasoning, boundary의 완결성 검사 결과

Local validator는 빈 출력을 성공으로 바꾸기 위해 reasoning step이나 graph
state를 합성하지 않는다. 형식 오류는 전체 stage를 다시 생성한다. 최종
boundary를 복구하지 못한 문항은 균등 확률 fallback으로 성공 처리하지 않고
실패로 기록한다.

## Baseline 비교 계약

논문의 controlled comparison에서는 각 forecaster model이 현재 evidence를
독립적으로 검색한다. 한 모델 안에서는 모든 방법이 같은 질문, target
contract, model-specific current evidence와 retrieval manifest를 공유하고
표현 방식만 달리한다. 다른 모델이 만든 search 결과나 evidence ledger는
재사용하지 않는다.

| Method | 제공되는 과거 정보 |
|---|---|
| Structured Direct Forecasting | 없음. 동일한 현재 evidence와 구조화된 추론 및 probability boundary를 사용하는 강한 통제군 |
| DAG Forecasting | 현재 evidence로 새로 만든 prospective DAG |
| Direct DAG Retrieval | 같은 retrieval manifest의 outcome redacted DAG |
| Factor Memory | HGF와 동일한 중립화 graph에서 edge와 path를 제거한 factor만 제공 |
| Resolved Case | 같은 사건의 outcome redacted account 제공 |
| Forecasting Principles | 같은 사건에서 추출한 일반 예측 원칙 제공 |
| HGF | 같은 graph의 topology를 현재 evidence로 다시 채워 제공 |

Evidence retrieval의 효과를 별도로 연구하지 않는 main table에서는 모델별로
생성한 E1 evidence를 해당 모델의 모든 방법에 고정해 제공한다. 방법별 검색
전략을 비교하는 실험은 main table과 분리해 별도 ablation으로 보고한다.

## 현재 감사 결과

과거 100문항 실행에서 Accuracy 0.590, Brier 0.2039006이 관측되었다. Source
code와 Blueprint bank는 보존되어 있지만, 해당 실행에는 다음 완결성 문제가
있다.

- 68문항의 저장된 reasoning이 baseline과 target bridge 두 단계뿐이다.
- 78문항의 graph instantiation이 빈 model output을 local default state로
  채웠다.
- 30문항이 deterministic evidence ledger fallback을 사용했다.
- 31문항이 boundary validation 실패 뒤 neutral probability fallback을
  사용했다.

Neutral boundary fallback 문항의 Brier는 0.2294였고 나머지 문항은
0.1924였다. 따라서 0.2039가 fallback만으로 만들어진 수치는 아니다.
그럼에도 실제 prediction에 사용된 reasoning과 graph가 다수 문항에서
불완전하므로 이 한 번의 수치를 최종 논문 결과로 직접 보고하지 않는다.

Fallback 없이 모든 stage를 완결하도록 강제한 별도 100문항 실행은 Accuracy
0.530, Brier 0.2221이었다. 이 결과는 실행 엄밀성은 충족하지만 reasoning
길이를 7개에서 10개 단계로 과도하게 고정했다. 최종 구현은 원래 pipeline을
유지하면서 빈 응답과 기록 손실만 해결해야 하며, 결과를 높이기 위한 별도
probability 규칙을 추가하지 않는다.

### 0.2039 재현 게이트 진단

동일 code, Blueprint hash, model alias, temperature 0, medium effort, seed 0으로
새로 실행했을 때 36개 paired 문항의 Brier는 과거 0.2145에서 0.2419로
악화되었다. 따라서 과거 full 100의 0.2039는 아직 독립 재현된 결과가 아니다.

변동이 시작되는 단계를 찾기 위해 첫 50문항을 대상으로 과거 stage를 하나씩
고정하고 나머지 stage만 새로 생성했다. 이 진단 결과는 인과 분석용이며 논문
성적으로 사용할 수 없다.

| 고정한 과거 stage | 비교 문항 | 과거 Brier | 새 Brier | 새 Accuracy |
|---|---:|---:|---:|---:|
| evidence ledger | 50 | 0.2061 | 0.2246 | 0.460 |
| evidence ledger와 instantiated graph | 50 | 0.2061 | 0.2284 | 0.500 |
| evidence ledger, graph, reasoning | 43 | 0.2104 | 0.2082 | 0.558 |

마지막 행은 API key limit 이전에 완료된 43문항이다. 이 43문항에서는 boundary
입력 reasoning이 과거 실행과 모두 동일했지만 probability가 동일한 문항은
12개뿐이었다. 그럼에도 평균 Brier 차이는 -0.0023으로 작았다. 반면 과거
graph까지 동일하게 주고 reasoning을 새로 생성했을 때는 Brier가 0.0223
악화되었다. 현재 증거 정리의 변동도 downstream 입력을 바꾸지만, 이번 재생의
주된 성능 손실은 동일한 graph에서 생성된 reasoning의 변동과 여러 repair
경로에서 발생했다. Temperature 0과 seed 전달만으로 provider output이 완전히
결정적이지 않다는 점도 확인되었다.

Raw generation metadata를 다시 조회한 결과, 현재 Google 응답 67건은 동일한
model alias와 provider name 아래 두 개의 서로 다른 endpoint ID에 44건과
23건으로 나뉘어 처리되었다. OpenRouter의 기본 routing은 가용성과 가격에 따라
여러 endpoint를 load balance하며 fallback도 허용한다. 과거 0.2039 실행은
generation ID와 endpoint ID를 저장하지 않았으므로 당시 endpoint 조합을 다시
구성할 수 없다. 이는 method의 차이가 아니라 기존 실행 계약의 재현성 결함이다.

새 canonical 실행은 모델마다 하나의 exact provider tag를 고정하고 provider
fallback을 끄며, seed, response format, reasoning parameter를 모두 지원하는
endpoint만 허용한다. Request의 provider policy와 response의 endpoint ID를
문항별로 기록한다. Provider가 바뀌거나 endpoint ID가 허용 목록을 벗어난
문항은 성공으로 집계하지 않는다.

재현 실행은 개발 중인 dirty worktree에서도 분리한다. Base repository는
`27ff13c`의 detached clean worktree를 사용하고, HGF 구현은 Jin snapshot의
보존된 source를 overlay한다. Sidecar와 provider policy wrapper만 실행 계약으로
추가한다. 현재 준비된 runtime에서 HGF source 5개는 snapshot과 모두 동일하고,
Blueprint manifest SHA-256은
`897caad609dd5f17a0078708ad7a9a685f26ac2691a79ba6e721e1d58bc4c7c4`다.
따라서 개발 중인 baseline이나 retrieval 수정은 재현 실행에 들어가지 않는다.

`experiments/run_reproduction_gate.py`는 같은 endpoint와 seed로 5문항을 두 번
완전히 새로 호출한다. Endpoint, evidence input, retrieval result, ledger, memory,
final probability가 동일할 때 fresh full 100을 시작한다. Graph instantiation과
reasoning은 생성형 중간 산출물이므로 byte-level 동일성을 요구하지 않고 차이를
감사 대상으로 기록한다. 이 둘의 재현성은 독립 seed 반복에서 구조 사용률,
reasoning completeness, 성능 분포가 안정적인지로 판정한다. Full run 뒤에는
수치 재현과 publication integrity를 서로 분리해 판정한다.

고정된 Google Vertex endpoint에서 수행한 5문항 사전 점검에서는 두 실행의
endpoint와 입력, retrieval, ledger, memory, probability가 5문항 모두 정확히
일치했다. Graph와 reasoning은 한 문항에서 달랐지만 두 실행의 Accuracy,
Brier, NLL은 모두 동일했다. 이는 중간 자연어 trace의 제한된 변동과 최종
예측의 실행 안정성을 구분해야 함을 보여준다.

이 진단만으로는 0.2039를 재현했다고 주장할 수 없었다. 같은 frozen code와
고정 endpoint를 사용한 독립 full 100 결과가 필요했고, 수치 재현과 trace
완결성을 분리해 판정하기로 했다.

## 단일 endpoint 100문항 재현 결과

고정된 Google Vertex endpoint에서 frozen code를 100문항에 다시 실행했다.
첫 실행은 96문항을 완료했고, 네 문항은 모두 OpenRouter 응답 본문을 SDK가
해석하지 못한 `JSONDecodeError`로 실패했다. 같은 model, endpoint, seed와
generation setting으로 네 문항을 다시 호출해 모두 완료했으며, 원 실행과
재시도 호출을 별도로 보존한 뒤 실패 문항만 교체한 recovery manifest를
작성했다.

| 실행 | 문항 | Accuracy | Brier | NLL |
|---|---:|---:|---:|---:|
| 과거 frozen 결과 | 100 | 0.590 | 0.2039 | 0.8664 |
| 단일 endpoint fresh 재현 | 100 | 0.570 | 0.2069 | 0.8931 |

Fresh 재현은 protocol gate와 numerical reproduction gate를 모두 통과했다.
Accuracy 차이는 -0.020, Brier 차이는 +0.0030, NLL 차이는 +0.0267이다. 정상
응답과 재시도 응답은 모두 동일한 endpoint ID에서 생성되었다. 따라서 저장된
Procedural Topology HGF의 성능 수준은 재현되었다고 판단한다.

Publication integrity는 통과하지 못했다. Frozen runtime은 provider가 내부
reasoning만 반환하고 structured content를 비운 채 `finish_reason=error`로
종료한 응답을 빈 JSON 객체로 처리했다. 이 때문에 evidence ledger, graph,
reasoning, boundary stage에서 실제 생성 실패가 downstream default로 바뀌었다.
Recovered 100문항에서는 ledger fallback 2건, default-only graph 14건, boundary
fallback 9건이 관찰되었다. 기존 고정형 audit는 활성 path가 없는 정당한
반대 근거 trace에도 driver와 mechanism을 요구했으므로 incomplete reasoning을
82건으로 과대 계상했다. Raw response를 다시 분석하면 structured reasoning
content 자체가 비어 있던 문항은 30건이었다.

Canonical runtime은 방법론을 바꾸지 않고 이 실행 결함만 수정한다. 빈 content,
provider error finish, SDK transport JSON error를 정상 결과로 받지 않고 같은
등록 effort로 최대 네 번 재시도한다. Forecast prompt, subgraph topology,
evidence, boundary mapping, probability는 바꾸지 않는다. Reasoning completeness는
고정된 step 수로 정의하지 않는다. 활성 path가 있으면 current driver와
mechanism을 모두 요구하고, 모든 path가 기각되었으면 baseline, 명시적인 path
assessment, target bridge로 완결된 trace를 허용한다. 이 조건은 DAG를 억지로
활성화하지 않으면서 prediction에 실제 사용된 구조적 reasoning을 기록하기
위한 최소 조건이다.

## 논문 결과 승인 조건

최종 표에 들어가는 실행은 다음 조건을 모두 만족해야 한다.

- 고정된 code와 artifact hash로 독립 seed 세 번 실행
- 모든 문항에서 fallback 0건과 outcome contamination 0건
- 실제 prediction에 사용된 reasoning과 evidence가 저장됨
- 각 모델 안에서 같은 E1 evidence와 같은 retrieval manifest를 사용한
  baseline 비교
- 모델별 exact identifier와 provider routing policy 고정

- Accuracy, Brier, NLL과 category breakdown 기록
- token, cost, elapsed time, retry rate 기록
- 한 seed의 우연한 최고점이 아니라 평균과 표준편차 보고

위 조건은 최종적인 다중 seed 주장에 필요한 기준이다. 2026년 8월 1일에
고정한 현재 registry는 seed 0 한 번만 포함하므로 평균과 표준편차를 보고하는
실행으로 취급하지 않는다. 나머지 완결성 조건은 문항별 raw audit와 최종
summary에서 검사한다.

## Minimal Clean Procedural Topology HGF

과거 `0.2039` 실행은 간결한 end to end 흐름의 성능 가능성을 보여 주었지만,
원본 Blueprint 문장에 과거 실현을 암시하는 표현이 남아 있었고 일부 빈 생성이
local default로 처리되었다. 이후 strict 구현은 이 문제만 고친 것이 아니라
reasoning 길이와 graph 사용량까지 강제하여 원래 방법과 다른 동작을 만들었다.
따라서 두 실행의 차이만으로 topology transfer의 효과를 판단할 수 없다.

`src/hgf_e2e_topology_clean`은 이 두 요인을 분리하기 위한 최종 최소 변경
구현이다. 단계 순서는 현재 증거 정리, 동일 event family의 과거 subgraph
검색, 현재 증거를 이용한 node와 edge 상태 채우기, 구조를 이용한 reasoning,
동일 probability boundary의 순서다. 과거 예측, 확률, 정답, 실현값, worked
conclusion은 어느 단계에도 입력되지 않는다.

과거 DAG의 node와 edge 문장은 한 번만 outcome neutral한 조건문으로 바꾸고
frozen artifact로 저장한다. ID, node 순서, edge endpoint, relationship,
directionality, lag, support, confidence, path 순서는 원본 Blueprint와 정확히
같아야 한다. 구조의 의미를 지우기 위해 topology를 요약하거나 새 graph로
다시 생성하지 않는다.

현재 evidence ledger를 먼저 만든 뒤 여러 과거 DAG에서 관련된 complete path를
선택한다. 선택된 path의 모든 node와 edge는 현재 evidence로 다시 확인한다.
Forecaster는 현재 graph에서 active로 확인된 path 중 실제 예측에 필요한 일부만
사용할 수 있다. 모든 active path를 쓰거나 일정한 수의 driver, mechanism,
reasoning step을 만들도록 강제하지 않는다. 다만 사용한다고 선언한 path는
current driver와 mechanism을 모두 설명해야 하며, prediction에 사용한 evidence
ID와 path ID를 그대로 기록해야 한다.

빈 provider content, 불완전한 graph, 생성되지 않은 reasoning, boundary validation
실패는 중립값이나 local default로 바꾸지 않는다. 같은 stage contract로 다시
생성하고 끝내 유효한 출력이 없으면 해당 문항은 실패로 남긴다. Probability
pooling, posterior adjustment, 다른 method의 prediction 참조는 금지한다. 실행은
source, selection, model-specific E1 evidence, retrieval manifest, neutral topology
bank의 hash가 모두 같은 경우에만 resume할 수 있다.

이 구현의 판정 목적은 과거의 좋은 수치를 그대로 복구하는 것이 아니다. 과거
실현 문장과 fallback을 제거한 뒤에도 간결한 topology transfer가 강한 baseline을
넘는지를 확인하는 것이다. 개발 비교에는 결과를 보기 전에 고정한 균형 40문항을
사용한다. 이 단계에서 충분한 우월성과 무실패 실행을 확인한 동일 버전만 full
100으로 승격한다. 최종 목표는 다섯 모델 평균 Brier `0.215` 미만이며, 이 기준은
결과를 본 뒤 prompt나 확률을 조정하는 데 사용하지 않는다.

## 고정 균형 40문항과 full 100 승격 규칙

개발 비교용 문항은 `data/questions/selection_balanced_40.json`에 고정한다. 다섯
금융 카테고리에서 각각 여덟 문항을 선택하고, 각 카테고리의 여덟 event family를
한 번씩만 포함한다. Easy 20문항과 hard 20문항의 구분에는 새로운 HGF 결과나
이번에 평가하는 두 baseline 결과를 사용하지 않는다. 이미 등록된 seed 0 결과
중 네 primary model과 Direct Forecasting, DAG Forecasting, Factor Memory,
Forecasting Principles의 문항별 Brier만 사용한다. Binary와 three option 문항을
비교할 수 있도록 각 Brier를 uniform forecast Brier로 나눈 뒤 평균한다. 각
카테고리에서 평균 난이도가 낮은 네 family와 높은 네 family를 정하고, easy
family에서는 가장 쉬운 문항을, hard family에서는 가장 어려운 문항을 한 개
고른다. 입력 파일과 모든 source result hash는
`experiments/subsets/balanced_40_difficulty_manifest.json`에 기록한다.

HGF 비교 대상은 Outcome Redacted Case Retrieval과 Outcome Neutral Direct DAG
Retrieval 두 방법이다. 세 방법은 모델별로 동일한 frozen E1 evidence selection과
동일한 historical retrieval manifest에서 시작한다. HGF 후보는 40문항의 결과를
본 뒤 같은 버전 안에서 수정하지 않는다. 새로운 아이디어는 새 source package,
새 method identifier, 새 output directory에서 실행한다.

Full 100 승격은 다음 조건을 모두 만족할 때만 허용한다.

- 다섯 모델의 200개 HGF 예측이 모두 성공하고 reportability audit를 통과함
- pooled Brier가 두 baseline 각각보다 낮고, 가장 강한 baseline 대비 상대 개선이
  최소 5 percent에 도달함
- 다섯 모델 중 최소 네 모델에서 가장 강한 baseline보다 Brier가 나쁘지 않으며,
  최소 세 모델에서는 더 낮음
- paired bootstrap으로 계산한 pooled Brier 차이의 95 percent 상한이 0보다 작음
- 확률 사후조정, baseline prediction 참조, 결과별 provider 교체, 문항 교체가 없음

승격 후에는 source, prompt, validator, provider policy, seed 0, input manifest를
바꾸지 않고 원래 100문항 selection에서 실행한다. Full 100이 40문항 결과를
재현하지 못하면 full 결과를 최종 판단으로 사용하며, 40문항 성능만 논문 성능
주장에 사용하지 않는다.

## Clean v1.1 reliability revision

`src/hgf_e2e_topology_clean_v1_1`은 Minimal Clean HGF의 방법론을 바꾸지 않는
출력 신뢰성 버전이다. 모델이 baseline step, adaptive middle reasoning, target
bridge를 각각 생성하고 runtime은 세 필드를 순서대로 이어 붙이기만 한다. 문장,
evidence, path, effect는 runtime이 만들거나 고치지 않는다. Reasoning source로
사용할 수 있는 DAG path ID는 current graph에서 active인 path로 제한한다. Active
path가 없으면 current evidence는 `CURRENT_NEW`, target operation은
`TARGET_CONTRACT`로 기록하고, unresolved path는 counterevidence와 uncertainty에만
기록한다. 이 변경은 Llama와 MiniMax가 inactive path를 driver source로 적어
유효한 예측 전체가 폐기되던 serialization 오류를 제거하기 위한 것이다.

## Grounded path revision

Clean v1.1의 고정 40문항 실행을 점검하면 current evidence가 여러 node와 edge를
직접 지지해도 root to target path 전체가 `ACTIVE`가 아니라는 이유로 topology가
reasoning에서 빠지는 경우가 많다. 이는 DAG가 현재 관측을 해석하는 구조라는
논문의 목적과 맞지 않는 all or nothing 병목이다. 관측되지 않은 downstream
mediator 하나가 있다는 이유로 현재 확인된 driver와 인접 mechanism까지 버리면
HGF가 사실상 current evidence forecasting으로 축소된다.

`src/hgf_e2e_topology_clean_v1_2`는 확률이나 답을 건드리지 않고 이 병목만 고친
별도 버전이다. Graph instantiation이 끝난 뒤 deterministic support register를
만든다. Complete path가 현재 지지되면 `FULL`이다. Complete path는 unresolved라도
현재 evidence가 적어도 두 node와 연속된 두 edge를 함께 지지하고 contradiction이
없으면 `PARTIAL`이다. 단일 factor와 edge만 확인된 경우에는 구조 이전이라고
부풀리지 않고 current factor로 처리한다. Contradicted edge가 있거나 연속된
current mechanism이 확인되지 않으면 해당 path는 driver나 mechanism으로 사용할
수 없다.

Forecaster는 `FULL` 또는 `PARTIAL` path가 있으면 그중 적어도 하나를 사용한다.
`PARTIAL` path에서는 현재 evidence가 채운 segment만 factual driver로 취급하고,
관측되지 않은 나머지 edge sequence는 조건부 mechanism과 uncertainty로 남긴다.
관측되지 않은 연결은 evidence가 없는 `missing_link` step으로 명시하여, 구조적
가정과 현재 확인된 mechanism이 로그에서 섞이지 않게 한다.
같은 source DAG에서 checkpoint, edge, mechanism이 모두 같은 중복 path는 graph에
그대로 보존하되 첫 path만 reasoning 후보로 사용한다. 이는 topology를 요약하는
절차가 아니라 동일한 구조가 prompt와 trace에서 반복되는 것을 막는 결정적 제거다.
이 구조는 방향을 설명할 수 있지만 현재 target value나 magnitude를 제공하지
않는다. 모든 factual statement는 여전히 current article ID를 인용해야 하며,
current evidence가 historical topology보다 우선한다. 사용한 grounded path에는
current driver와 mechanism step을 모두 기록한다.

이 변경은 adaptive topology transfer를 실제로 작동시키기 위한 routing 규칙이다.
Historical answer, realization, probability, 다른 baseline의 예측은 입력되지 않는다.
Probability pooling과 posterior adjustment도 없다. v1.1 결과는 수정하지 않고
reference run으로 보존하며, v1.2는 새 method ID와 output directory에서 같은 고정
40문항으로 비교한다. v1.2가 승격 조건을 통과할 때만 source와 설정을 그대로
동결하여 full 100을 실행한다.

## 2026년 8월 1일 최종 등록 실행

현재 main registry에는 다섯 모델, 일곱 방법, 100문항으로 구성된 3,500개
예측이 있다. 논문의 주 모델인 Gemini 2.5 Flash Lite, GPT-5 mini, DeepSeek
V3.2, Llama 4 Maverick은 2,800개이며 MiniMax M2.5의 700개는 보조 결과다.
3,500개 모두 결과, 실제 prediction에 사용한 evidence ID, written reasoning,
raw request와 response, token, cost, elapsed time을 보존하고 reportability
검사를 통과했다.

| Model | HGF Accuracy | HGF Brier | HGF NLL | 최저 Brier baseline |
|---|---:|---:|---:|---|
| Gemini 2.5 Flash Lite | 0.510 | 0.2368 | 0.9771 | Structured Direct Forecasting 0.2324 |
| GPT-5 mini | 0.520 | 0.2316 | 1.0174 | Direct DAG Retrieval 0.2276 |
| DeepSeek V3.2 | 0.480 | 0.2282 | 0.9891 | Resolved Case 0.2065 |
| Llama 4 Maverick | 0.460 | 0.2362 | 0.9855 | Factor Memory 0.2240 |
| MiniMax M2.5 | 0.480 | 0.2333 | 0.9749 | Resolved Case 0.2177 |

따라서 이 single seed 표는 실행 완결성과 여러 모델에서의 적용 가능성을
보여주지만, HGF가 가장 강한 baseline보다 일관되게 우수하다는 주장을
지지하지 않는다. 이 제한은 결과 해석과 논문 본문에 그대로 반영해야 한다.

Strict Historical Live HGF 비교는 네 모델에서 400개 모두 reportable하다.
반면 Original Main HGF는 400개 출력을 생성했지만 최종 probability 이전에
예측 값이 reasoning stage에 노출된 경우가 많아 모델별 5개, 총 20개만 최종
reportability 계약을 만족했다. 해당 400개 성적은 진단용으로만 보존하고
논문 표에는 사용하지 않는다.

현재 등록 실행의 중요한 예외는 다음과 같다.

- Gemini의 transport recovery 일부는 동일 model, method, seed와 request
  contract를 유지하면서 Google과 Google AI Studio 표기로 기록된 endpoint를
  모두 포함한다.
- Llama 4 Maverick은 같은 structured reasoning prompt를 사용했지만 선택된
  endpoint가 native reasoning effort를 제공하지 않아 effort 값은 none이다.
- 모든 결과는 seed 0 한 번의 fresh call이다. 다중 seed 안정성을 주장하려면
  seed 1과 2를 별도로 실행해야 한다.

추적 가능한 축약 결과와 source hash는
`experiments/final_results/sync_manifest.json`에 고정한다.
