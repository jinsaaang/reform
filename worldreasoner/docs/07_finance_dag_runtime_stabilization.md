# WorldReasoner 금융 DAG 실행 안정화 계획

> 범위: 현재 WorldReasoner 파이프라인이 오래 걸리거나 중간에서 반복 실패하는 문제
>
> 원칙: 검색 -> 증거 수집 -> 인과 설명 -> DAG 생성이라는 기존 골격은 그대로 둔다.

## 1. 목표

이 문서의 목적은 DAG 품질 체계를 새로 설계하는 것이 아니다. 현재 WorldReasoner가
의도한 작업을 다음 조건으로 안정적으로 끝내게 하는 것이다.

- 모델이 검색 결과나 tool output 자료형을 직접 추측하지 않는다.
- 이미 끝난 단계는 다시 실행하지 않는다.
- 같은 오류와 같은 payload를 반복하지 않는다.
- 긴 기사 본문과 inspector 출력 때문에 context가 불필요하게 커지지 않는다.
- 한 질문이 실패해도 저장된 중간 결과를 이용해 정확한 단계부터 재개한다.
- 기사 목표 10건, DAG 최소 6노드 같은 실행 설정을 한 곳에서 일관되게 사용한다.

## 2. 유지할 현재 골격

```text
Question
  -> HindsightAgent
     -> evidence_collector의 agentic web search
     -> article_collector 저장
     -> causal explanation 저장
  -> GraphBuilderAgent
     -> propose_subgraph
     -> graph_inspector
     -> graph_built
  -> JSON/DB export
```

유지 사항:

- HindsightAgent와 GraphBuilderAgent를 그대로 사용한다.
- web search를 한 번의 고정 검색으로 바꾸지 않는다.
- agent가 필요에 따라 검색어를 바꾸고 반복 수집하는 방식은 유지한다.
- Event, CausalHypothesis, Article 및 SQLite 구조는 유지한다.
- 프롬프트의 목적이나 인과 설명 형식은 전면 수정하지 않는다.
- OpenRouter의 Gemini 2.5 Flash 호출 방식은 유지한다.

## 3. 이번 실행에서 실제로 시간을 소모한 지점

### 3.1 검색 결과가 backend마다 다른 자료형으로 반환됨

`WebSearchTool`은 SearXNG를 사용할 때 내부적으로 structured result를 만들지만 최종
출력은 markdown 문자열이다. fallback search도 markdown 문자열을 그대로 반환하며,
structured result 및 auto-collect 경로를 사용할 수 없다.

그 결과 evidence agent가 다음 값을 문자열에서 직접 읽어야 했다.

- 제목
- URL
- 게시일
- publisher

게시일을 찾지 못하면 모델이 질문 시작일인 `2026-03-28`을 넣었다. 실제로는 다수의
기사가 4월 30일 이후 게시됐지만 9건이 모두 3월 28일로 저장되었다. 이 잘못된 날짜가
뒤의 GraphBuilder chronology validation을 반복 실패시켰다.

### 3.2 typed tool output을 모델이 dict처럼 검사함

`article_retrieval`은 `ArticleRetrievalOutput` 객체를 반환한다. 이번 HindsightAgent는
다음과 같은 방식으로 응답을 검사했다.

```python
if article_content and "content" in article_content:
    ...
```

`OutputModelBase`에는 `__getitem__`, `get`, `keys`가 있지만 `__contains__`가 없다.
따라서 실제 본문이 존재하는데도 9개 기사 조회를 전부 실패로 판단했고, 같은 작업을
다시 한 번 반복했다. 이후에는 본문이 없다고 잘못 판단한 상태로 인과 설명을 썼다.

이 한 문제로 불필요한 agent step 두 개와 기사 18회 조회가 발생했다. 해당 run의
HindsightAgent 누적 입력 토큰도 약 42만 토큰까지 증가했다.

### 3.3 생성한 article alias가 output schema에서 사라짐

`QuestionArticlesTool`은 내부 결과에 `alias`를 넣지만 `ArticleListItem` 모델에는
`alias` 필드가 없다. Pydantic 변환 과정에서 이 값이 버려진다.

GraphBuilder는 첫 단계에서 `article.alias`를 읽다가 실패하고, 다음 단계에서 article
ID를 alias 대신 사용하는 우회 코드를 다시 만들었다. URL도 tool 내부 결과에 넣지
않아서 `url=None`으로 반환된다.

### 3.4 잘못된 graph payload를 거의 그대로 반복함

잘못된 기사 날짜 때문에 event validation이 실패했을 때 GraphBuilder가 비슷한
payload를 여러 번 다시 제출했다. `propose_subgraph`는 일부 항목이 성공하면
`partial_success`로 DB에 남기므로 재시도 중 중복 event/hypothesis도 생겼다.

현재는 다음 장치가 없다.

- 동일 payload + 동일 오류 감지
- 저장 전 dry-run validation
- batch 실패 시 전체 rollback
- 실패한 항목만 정확히 재시도하는 자동 경로

### 3.5 agent context가 계속 커짐

긴 기사 본문, 전체 subgraph payload, 전체 graph inspector 출력이 이후 모든 agent
step의 입력 history에 누적된다. 이번 GraphBuilder는 마지막 step에서 누적 입력이 약
27만 토큰까지 증가했다.

GraphBuilder의 판단에 필요한 것은 대부분 다음뿐이다.

- 기사 ID, alias, 제목, 날짜, 짧은 관련 excerpt
- 생성 성공/실패 항목
- 노드 수, 깊이, orphan 및 validation error

전체 본문과 전체 graph report를 매 step 다시 볼 필요는 없다.

## 4. 최소 보완 사항

우선순위는 실행 중 실제로 막힌 순서다.

### P0-1. WebSearchTool 출력 하나로 통일

대상:

- `src/tools/collectors/web_search.py`
- `src/tools/base/output_models.py`

변경:

1. SearXNG와 fallback search 모두 동일한 `WebSearchOutput`을 반환한다.
2. 결과는 `results[]` 아래에 `title`, `url`, `description`, `published_date`,
   `source`를 갖는다.
3. markdown은 화면 표시용 문자열로만 만들고 agent의 데이터 입력으로 사용하지 않는다.
4. auto-collect도 동일한 `results[]`를 사용한다.
5. 게시일이 없는 결과는 모델이 날짜를 채우게 하지 않고 article fetch 단계로 넘긴다.

중요한 점은 agentic search를 없애는 것이 아니다. agent는 여전히 검색어와 반복 여부를
결정하지만, 검색 결과 파싱은 Python tool이 책임진다.

완료 기준:

- search backend를 바꿔도 agent가 받는 object schema가 같다.
- 모델 코드에 markdown/URL/날짜 정규식 파싱이 등장하지 않는다.

### P0-2. 게시일을 ArticleCollector가 확정

대상:

- `src/tools/collectors/article_collector.py`
- `src/tools/collectors/web_fetch.py`

변경:

1. `published_date` 입력을 모델의 확정값이 아니라 hint로 취급한다.
2. fetch한 페이지에서 다음 순서로 날짜를 추출한다.
   - JSON-LD `datePublished`
   - OpenGraph/article metadata
   - `<time datetime>`
   - URL의 연/월/일
   - 본문 상단의 명시적 게시일
3. 모델 입력 날짜와 페이지 날짜가 다르면 페이지 근거를 우선하고 warning을 남긴다.
4. 날짜를 확정하지 못하면 질문 시작일로 대체하지 않는다.
5. 날짜와 날짜 출처를 Article metadata에 함께 저장한다.

완료 기준:

- Apple 샘플의 4월 30일 기사들이 3월 28일로 저장되지 않는다.
- 날짜 불명 기사는 명시적으로 표시되어 재검색 대상으로 돌아간다.

### P0-3. Tool output의 dict/object 호환성 수정

대상:

- `src/tools/base/output_models.py`
- 관련 unit test

변경:

1. `OutputModelBase`에 `__contains__`를 추가한다.
2. nested output도 attribute access와 dict access가 동일하게 동작하는지 테스트한다.
3. `json.dumps`, `.get()`, `result["field"]`, `result.field`,
   `"field" in result`를 모두 지원한다.

이 보완은 agent에게 사용법을 프롬프트로 다시 가르치는 방식이 아니다. 모델이 흔히
생성하는 두 접근법을 tool contract가 모두 안전하게 처리하게 한다.

완료 기준:

- 현재 HindsightAgent가 생성했던 본문 조회 코드가 수정 없이 9건을 정상 인식한다.
- 동일 기사 조회를 실패로 오판해 반복하지 않는다.

### P0-4. QuestionArticlesOutput에서 alias와 URL 보존

대상:

- `src/tools/base/output_models.py`
- `src/tools/generators/question_articles.py`

변경:

1. `ArticleListItem`에 `alias`를 추가한다.
2. `QuestionArticlesTool` 결과에 실제 `url`을 넣는다.
3. output model 변환 전후의 alias, ID, URL이 동일한지 테스트한다.

완료 기준:

- GraphBuilder의 첫 호출에서 `article.alias`가 바로 동작한다.
- alias를 찾기 위한 추가 agent step이 발생하지 않는다.

### P0-5. Graph batch의 사전 검증과 반복 차단

대상:

- `src/tools/reasoning/propose_subgraph.py`
- `src/tools/reasoning/event_identifier.py`
- `src/tools/reasoning/causal_reasoner.py`

변경:

1. event/edge 전체를 저장 전에 검증하는 dry-run 단계를 둔다.
2. 하나라도 잘못되면 canonical DB에 아무것도 저장하지 않는다.
3. `source_article_ids`가 들어오면 `article_ids`로 정규화한다.
4. 검증 오류는 실패한 item index, alias, 필드 및 이유만 짧게 반환한다.
5. runner가 `payload hash + error hash`를 기록한다.
6. 같은 조합이 두 번 나오면 세 번째 동일 제출을 실행하지 않는다.
7. 반복 감지 시 실패한 item만 individual tool로 재시도하거나 현재 graph stage를
   실패 상태로 checkpoint한다.

고정 wall-clock timeout을 다시 넣지는 않는다. 시간으로 중단하는 대신 진전 없는
동일 행동만 중단한다.

완료 기준:

- 동일 chronology error가 세 번 반복되지 않는다.
- 실패한 batch가 중복 event/hypothesis를 남기지 않는다.

### P1-1. Agent에 필요한 만큼만 본문 반환

대상:

- `src/tools/inspectors/article_retrieval.py`
- `src/tools/inspectors/graph_inspector.py`

변경:

1. `article_retrieval`에 선택 인자 `query`, `max_chars`를 추가한다.
2. query가 있으면 관련 구간과 앞뒤 문맥만 반환한다.
3. 원문 전체는 DB에 유지하고 agent history에는 기본 상한을 적용한다.
4. `graph_inspector`에 `compact=true` 출력을 추가한다.
5. 중간 검사에는 노드 수, 깊이, orphan, 오류만 반환하고 전체 report는 최종 한 번만
   사용한다.

완료 기준:

- HindsightAgent 누적 입력 토큰을 현재 약 42만에서 10만 이하로 줄인다.
- GraphBuilder 누적 입력 토큰을 현재 약 27만에서 10만 이하로 줄인다.
- 인과 설명과 최종 graph JSON에 필요한 원문 정보는 유실되지 않는다.

### P1-2. 단계별 checkpoint와 resume

대상:

- `scripts/finance/run_dag_sample.py`
- evidence/graph pipeline executor
- `run_summary.json`

상태:

```text
question_loaded
evidence_collecting
evidence_complete
explanation_complete
graph_building
graph_complete
export_complete
```

변경:

1. 각 단계가 끝날 때 DB와 run summary에 checkpoint를 쓴다.
2. 재실행 시 기사 11건이 이미 있으면 search를 다시 하지 않는다.
3. causal explanation이 있으면 HindsightAgent를 다시 돌리지 않는다.
4. graph만 실패했다면 GraphBuilder 단계부터 재개한다.
5. partial graph는 staging 단위로 정리하고 정상 evidence는 보존한다.
6. 기사 목표, 최소 노드, 최소 깊이를 동일한 run config에서 읽는다.

완료 기준:

- 프로세스를 어느 단계에서 종료해도 다음 실행이 그 단계부터 이어진다.
- 이미 수집한 11개 기사를 다시 검색하거나 fetch하지 않는다.

### P1-3. 수집 병렬화

대상:

- `src/tools/collectors/web_search.py`
- `src/tools/collectors/article_collector.py`

변경:

1. 한 검색 결과에서 선택된 URL들을 현재 auto-collect 방식처럼 병렬 fetch한다.
2. URL deduplication을 fetch 전에 수행한다.
3. 한 사이트의 timeout이 다른 URL 수집을 막지 않게 한다.
4. SQLite write는 fetch 완료 후 짧은 순차 transaction으로 처리한다.

질문 여러 개의 병렬 실행은 단일 질문 안정화 후 적용한다. 우선 10개 qualification
질문을 각각 독립 DB로 실행하고 마지막에 merge하는 방식을 사용한다.

## 5. 보완 후 예상 실행 흐름

```text
1. Runner가 DB/checkpoint 확인
2. evidence가 부족하면 HindsightAgent 시작
3. evidence_collector가 structured web search 수행
4. 후보 URL을 병렬 fetch하고 게시일을 tool이 확정
5. 11건 저장 즉시 evidence checkpoint
6. manager가 key article excerpt만 조회해 causal explanation 저장
7. GraphBuilder가 alias가 포함된 article list를 한 번 조회
8. subgraph를 dry-run 검증 후 한 번에 commit
9. compact inspector로 노드 10개/깊이 3 확인
10. graph/export checkpoint 후 종료
```

## 6. 적용 결과: 금융 기본 실행 모드

금융 runner는 이제 다음 세 모드를 제공한다.

| 모드 | 증거 검색 | 설명 작성 | DAG 생성 | 용도 |
|---|---|---|---|---|
| `hybrid` (기본) | ToolCallingAgent | 기존 CodeAgent | 기존 CodeAgent | 금융 운영 |
| `code` | 기존 CodeAgent | 기존 CodeAgent | 기존 CodeAgent | WorldReasoner baseline 재현 |
| `tool` | ToolCallingAgent | ToolCallingAgent | ToolCallingAgent | 실험용 |

`hybrid`의 증거 검색에서는 모델이 검색어와 페이지, 재검색 여부만 결정한다. 실제
검색결과 순회, 병렬 fetch, 게시일 검증, 중복 제거, 질문별 provenance 저장은
`WebSearchTool`과 `ArticleCollectorTool`의 고정 코드가 수행한다. evidence agent에는
개별 `article_collector`나 Python executor를 노출하지 않는다.

전체를 `tool` 모드로 바꾸는 실험도 수행했지만 기본값으로 채택하지 않았다. 긴 causal
explanation을 수정할 때 전체 본문을 tool argument로 반복 전송했고, 대형 subgraph JSON
문자열 내부의 escaping 오류도 남았다. 따라서 이번 변경은 실제 병목이었던 agentic
search 경계만 좁혀 적용한다.

실행 예시:

```bash
PYTHONPATH=. .venv/bin/python scripts/finance/run_dag_sample.py \
  --agent-mode hybrid \
  --min-evidence-articles 10 \
  --min-graph-events 8 \
  --browser-concurrency 2
```

## 7. 2026-07-17 검증 결과

한 문항 Apple Q2 샘플의 단계별 integration run 결과:

- 모델: `google/gemini-2.5-flash` via OpenRouter
- 증거: 12건, 고유 source 12개
- 게시일 출처: page metadata 5건, URL 4건, 본문 3건
- agent가 임의로 넣은 날짜: 0건
- DAG: 평가 이벤트 14개, 인과 간선 15개, 최대 깊이 3
- 실제 outcome 유입 간선: 1개
- outcome impact: 13/13
- cycle: 없음
- 날짜 역행 edge: 없음
- 최종 상태: completed

산출물:

- `research/forecaster/experiments/worldreasoner_tool_agent_e11_n10_smoke_2/`

남은 병목도 있다. GraphBuilder의 누적 입력은 약 32만 토큰이었고, 15개 간선 중
`evidence_article_ids`가 직접 저장된 간선은 1개였다. 노드 13개에는 모두 기사 근거가
있지만 edge-level provenance는 약하다. 이는 이번 검색 실행 안정화와 별개의 다음
개선 범위다.

실패 시:

```text
- 검색 결과 부족 -> 새 검색어로 계속 수집
- URL fetch 실패 -> 그 URL만 제외하고 다음 후보 사용
- 게시일 불명 -> 날짜 확정 후보로 교체
- graph item validation 실패 -> 해당 item만 수정
- 같은 graph 오류 2회 -> 반복 중단 및 graph checkpoint
- 재실행 -> 마지막 정상 checkpoint부터 재개
```

## 6. 구현 순서

### 1단계: 즉시 고칠 네 가지

1. `OutputModelBase.__contains__`
2. `ArticleListItem.alias`와 URL 보존
3. article 게시일 자동 추출
4. search output 구조 통일

이 단계만으로 이번 샘플에서 발생한 잘못된 날짜, 본문 18회 오판, alias 접근 실패를
제거할 수 있다.

### 2단계: GraphBuilder 반복 제거

1. subgraph dry-run/atomic commit
2. payload/error 반복 감지
3. 실패 item만 재시도
4. compact inspector

### 3단계: 재개와 속도

1. 단계별 checkpoint
2. article excerpt 제한
3. URL 병렬 fetch
4. 10개 질문을 독립 DB로 병렬 qualification

## 7. 완료 기준

Apple 샘플을 새 DB에서 사람의 수동 수정 없이 실행했을 때:

- [ ] 검색 결과 제목, URL, 날짜를 모델이 문자열에서 파싱하지 않는다.
- [ ] 11개 기사 날짜가 실제 페이지 metadata와 일치한다.
- [ ] `article_retrieval` 결과 9개를 0개라고 오판하지 않는다.
- [ ] article alias 접근 오류가 발생하지 않는다.
- [ ] 동일 validation error가 세 번 반복되지 않는다.
- [ ] 실패한 graph batch가 DB에 일부 노드를 남기지 않는다.
- [ ] evidence 단계가 완료된 뒤 graph 재실행 시 검색을 반복하지 않는다.
- [ ] HindsightAgent와 GraphBuilder가 각각 누적 입력 10만 토큰 이내로 끝난다.
- [ ] evidence 수집부터 DAG export까지 중단 없이 완료된다.
- [ ] 기존 WorldReasoner의 단계 순서와 agent 역할은 바뀌지 않는다.

이 기준을 통과한 뒤에만 10개 샘플을 실행한다. 10개가 모두 수동 개입 없이 끝나면
질문별 독립 DB와 제한된 worker concurrency로 300개 실행을 시작한다.
