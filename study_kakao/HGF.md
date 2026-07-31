# HGF

HGF (Hindsight-Guided Forecasting)는 과거에 해결된 forecasting 사건의
정답을 재사용하지 않고, 그 사건의 refined hindsight DAG에서 검증된
인과 구조와 증거 요구조건을 현재 예측의 절차적 memory로 재사용한다.
현재 구현에서 HGF는 하나의 canonical 방법이며 별도 실험 variant가 아니다.

## 1. 전체 구조

```text
과거 해결 사건 + 사후 근거
        ↓
Refined Hindsight DAG
        ↓ deterministic compile + sanitize + topology validation
Blueprint
        ↓ cutoff-safe historical evidence로 생성
Exemplar
        ↓ fixed retrieval
현재 질문 + 현재 cutoff 이전 E1 증거
        ↓
family_id + target_metric compatibility 검사
        ├─ compatible   → Blueprint + sanitized Exemplar 사용
        └─ incompatible → historical memory 완전 제거
        ↓
현재 target operator에 맞춘 구조화 reasoning
        ↓ boundary validation / schema repair
모델이 출력한 최종 확률
```

마지막 확률에는 temperature scaling, modal boost, calibration boost 등
별도의 probability postprocessing을 적용하지 않는다. Schema repair와 생성
실패 시 neutral fallback은 실행 안정성을 위한 검증 경로이며 확률을 성능
향상 목적으로 재조정하는 단계가 아니다.

## 2. Blueprint 구축

입력은 200개의 refined DAG와 memory question metadata이다. 각 DAG는
deterministic compiler를 거쳐 다음 요소로 변환된다.

- `target_definition`: metric, operation, horizon, unit, comparator
- `checkpoints`: 원 DAG 노드의 causal role, evidence requirement,
  contradiction signal
- `causal_paths`: root-to-target 연결, mechanism, applicability 및 failure
  condition
- `alternative_hypotheses`: 경쟁 경로와 이를 구분할 증거
- `forecast_audit_questions`: 현재 reasoning을 점검할 질문
- `topology_validation`: edge coverage, path precision, acyclic 및 leakage
  검사 결과

Compiler는 과거 outcome, answer label, realized target value와 절대
period를 제거한다. 값과 시점은 현재 사건에 직접 복사할 수 없는 일반
조건으로 치환하지만, DAG의 노드 역할·edge·root-to-target path는 유지한다.
Canonical artifact의 내부 호환성 schema는
`hgf_blueprint_topology_v2`로 남아 있으나 외부 method 이름은 항상 `HGF`다.

`hgf-build-memory`는 source graph hash, compiler, Blueprint hash,
checkpoint/edge/path 통계를 manifest에 기록한다. V1 card fallback은 없다.
현재 bank의 완료 조건은 200/200, edge coverage 1.0, path precision 1.0,
acyclic, outcome/value/period leakage 0이다.

## 3. Exemplar 구축

Exemplar는 Blueprint와 독립적인 memory가 아니라, 해당 Blueprint를
어떻게 current-case reasoning으로 실행하는지 보여주는 cutoff-safe
demonstration이다. 따라서 각 memory Exemplar는 자신이 결합된 Blueprint의
hash를 기록한다.

Generator는 historical question의 forecast cutoff 이전 article만 사용할 수
있다. 결과에는 target semantics, forecast-time evidence, structured reasoning,
counterevidence, prospective estimate, option mapping과 uncertainty가 포함된다.
`hgf-build-exemplars`는 다음을 보장한다.

- memory Exemplar 200개
- 고정 평가 질문 100개의 top-1 case wrapper
- 모든 citation이 해당 historical cutoff 이전 evidence에 포함
- Blueprint hash와 Exemplar wrapper의 일치
- 기존의 유효한 결과를 보존하고 누락분만 생성하는 resume

## 4. 현재 질문에서의 실행

1. 현재 질문의 cutoff를 확정하고 E1 evidence를 cutoff-safe하게 로드한다.
2. 고정 mapping으로 historical memory를 선택한다.
3. 현재 질문과 memory의 `family_id`와 `target_metric`을 정확히 비교한다.
4. 불일치하면 Blueprint와 Exemplar를 모두 제거하고 현재 증거만 사용한다.
5. 호환되면 Blueprint 전체 topology와 sanitized demonstration을 Expert
   Memory로 컴파일한다.
6. 각 reasoning step은 실제로 충족한 checkpoint ID, `CURRENT_NEW`, 또는
   public arithmetic용 `TARGET_CONTRACT`를 기록한다.
7. 현재 증거가 checkpoint 요구조건을 충족하지 않으면 임의의 첫
   checkpoint로 연결하지 않는다. 정책에 따라 validation error가 되거나,
   memory rejection이 허용된 경우 `CURRENT_NEW`로 남긴다.
8. exact target operator가 level, change, return, growth acceleration 등을
   구분하고 target-period quantity와 comparator의 단위·산술을 검사한다.
9. boundary mapper가 option별 확률과 prediction의 schema 및 합계를
   검증한다. 통과한 raw probability가 최종 출력이다.

Canonical HGF가 사용하는 semantic lesson은 고정된 일반 원칙뿐이다.
Historical lesson distillation이나 semantic cache는 사용하지 않는다.
또한 이번 구현에는 mechanism-level applicability gate나 calibration
단계가 추가되어 있지 않다.

## 5. Baseline과의 입력 분리

공유되는 것은 question, cutoff, evidence contract, scorer와 refined DAG
loader이다. Memory payload는 분리된다.

| Method | Memory 입력 |
|---|---|
| HGF | `artifacts/hgf/blueprints` + matching Exemplar |
| Factor-Memory | `artifacts/baselines/factor_memory`의 frozen V1 card |
| Case/Text/Direct DAG | 각 baseline 정의에 따른 독립 payload |
| Search-only/Prospective DAG | historical HGF memory 없음 |

Factor-Memory가 HGF Blueprint의 mutable 목록을 공유하지 않으므로 HGF
변경이 baseline 정의를 바꾸지 않는다.

## 6. 이전 구현과의 차이

- 일부 stored guidance와 generic fallback의 혼합 대신 200개 전체 DAG를
  동일 compiler로 처리한다.
- DAG를 소수 factor로 평탄화하지 않고 checkpoint와 causal path를 보존한다.
- Blueprint와 Exemplar의 hash 결합을 검증한다.
- 정확한 family/metric 호환성 검사와 incompatible-memory 제거가 항상
  활성화된다.
- arbitrary checkpoint repair를 금지한다.
- historical semantic distillation/cache와 probability postprocessing을
  제거한다.
- `hgf-replay`와 `hgf-main-table`이 동일한 runtime 함수를 사용한다.
- 부분 Blueprint override 대신 완전한 `--hgf-artifact-root`만 허용한다.

이전 HGF의 코드, V1 cards, Exemplar, semantic lessons와 결과는
[legacy/original_hgf](legacy/original_hgf/README.md)에 실행 가능한 보관본으로
격리되어 있다.
