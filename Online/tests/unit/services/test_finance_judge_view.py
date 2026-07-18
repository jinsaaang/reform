"""Blind finance judge-view construction tests."""

from dataclasses import replace
from importlib.util import find_spec

import pytest

from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_results import (
    ArmSucceeded,
    ArmUnavailable,
    ArmUnavailableReason,
    TreatmentAudit,
)
from src.domain.finance.experiment_snapshot import FinanceEvidenceSnapshot
from src.domain.finance.forecast import (
    ForecasterInput,
    ForecastResult,
    OutcomeProbability,
    Scenario,
)
from src.domain.finance.judge_views import NeutralCandidate
from src.domain.finance.judging import JudgeCallOrientation
from src.domain.finance.memory import (
    OutcomeLabel,
    ScenarioId,
)
from src.domain.finance.sanitized_artifact import TransientForbiddenValueRegistry
from src.domain.finance.search import (
    EvidenceItem,
    EvidencePack,
    TargetProfile,
)
from src.services.finance_judge_view import (
    FinanceJudgeViewBuilder,
    JudgeCandidateSource,
    JudgeViewError,
    JudgeViewFailureReason,
    JudgeViewRequest,
    build_blind_judge_payload,
)
from tests.unit.domain.finance.test_judging import (
    DAG_SENTINEL,
    EVIDENCE_SENTINEL,
    make_bounded_judge_episode,
    make_judge_evidence,
    make_judge_target,
)


def test_finance_judge_view_service_exists() -> None:
    # Given / When
    specification = find_spec("src.services.finance_judge_view")

    # Then
    assert specification is not None


def test_judge_view_service_exposes_typed_builder() -> None:
    # Given
    builder = FinanceJudgeViewBuilder(alias_salt=b"s" * 32)

    # When
    salt = builder.alias_salt

    # Then
    assert salt == b"s" * 32


def _source(
    arm: ForecastArm,
    evidence: EvidenceItem,
    target: TargetProfile | None = None,
) -> JudgeCandidateSource:
    episode = make_bounded_judge_episode()
    memory = (episode,) if arm is ForecastArm.SEARCH_DAG else ()
    memory_reference = str(episode.dag_id) if memory else "public context"
    evidence_items = () if arm is ForecastArm.DIRECT else (evidence,)
    pack = EvidencePack(items=evidence_items, historical_dag_references=())
    forecast_input = ForecasterInput(
        target_profile=target or make_judge_target(),
        evidence_pack=pack,
        historical_memory=memory,
    )
    outcomes = (
        OutcomeProbability(label=OutcomeLabel("No"), probability=0.4),
        OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.6),
    )
    forecast = ForecastResult(
        scenarios=(
            Scenario(
                scenario_id=ScenarioId(f"scenario-{arm.value}"),
                name=f"Scenario for {arm.value}",
                reasoning_steps=(f"Use {EVIDENCE_SENTINEL} and {memory_reference}.",),
                probability=1.0,
                conditional_outcomes=outcomes,
                evidence_ids=tuple(item.evidence_id for item in evidence_items),
                historical_dag_references=(episode.reference,) if memory else (),
                assumptions=(f"Assume {arm.value} remains relevant.",),
                triggers=(f"Trigger {EVIDENCE_SENTINEL}.",),
                disconfirmers=(f"Disconfirm {memory_reference}.",),
                uncertainty="Residual uncertainty remains.",
            ),
        ),
        outcome_probabilities=outcomes,
        explanation=f"Explanation from {arm.name} with {evidence.content_hash}.",
    )
    snapshot = FinanceEvidenceSnapshot(items=evidence_items)
    treatment = TreatmentAudit(
        arm=arm,
        evidence_snapshot_digest=snapshot.digest if evidence_items else None,
        evidence_snapshot_bytes=snapshot.byte_length if evidence_items else 0,
        evidence_item_count=len(evidence_items),
        historical_memory_episode_count=len(memory),
    )
    return JudgeCandidateSource(
        arm_result=ArmSucceeded(
            arm=arm,
            attempts=(),
            treatment=treatment,
            forecast=forecast,
        ),
        forecast_input=forecast_input,
    )


def test_direct_pair_omits_arm_values_without_rejecting_direction_key() -> None:
    evidence = make_judge_evidence().model_copy(
        update={"claim": f"Claim cites {EVIDENCE_SENTINEL}."}
    )
    direct = _source(ForecastArm.DIRECT, evidence)
    search_only = _source(ForecastArm.SEARCH_ONLY, evidence)

    pair = FinanceJudgeViewBuilder(alias_salt=b"d" * 32).build(
        JudgeViewRequest((direct, search_only))
    )
    serialized_values = pair.model_dump(mode="json")
    serialized = pair.model_dump_json()

    assert '"direction"' in serialized
    assert "arm" not in serialized_values["candidate_1"]
    assert "arm" not in serialized_values["candidate_2"]
    assert '"direct"' not in serialized
    assert '"search_only"' not in serialized


def test_pair_namespaces_identical_provider_scenario_ids() -> None:
    evidence = make_judge_evidence().model_copy(
        update={"claim": f"Claim cites {EVIDENCE_SENTINEL}."}
    )
    direct = _source(ForecastArm.DIRECT, evidence)
    search_only = _source(ForecastArm.SEARCH_ONLY, evidence)
    assert isinstance(direct.arm_result, ArmSucceeded)
    assert isinstance(search_only.arm_result, ArmSucceeded)
    shared_id = ScenarioId("provider-scenario-1")
    direct_scenario = direct.arm_result.forecast.scenarios[0].model_copy(
        update={"scenario_id": shared_id}
    )
    search_scenario = search_only.arm_result.forecast.scenarios[0].model_copy(
        update={"scenario_id": shared_id}
    )
    direct = replace(
        direct,
        arm_result=direct.arm_result.model_copy(
            update={
                "forecast": direct.arm_result.forecast.model_copy(
                    update={"scenarios": (direct_scenario,)}
                )
            }
        ),
    )
    search_only = replace(
        search_only,
        arm_result=search_only.arm_result.model_copy(
            update={
                "forecast": search_only.arm_result.forecast.model_copy(
                    update={"scenarios": (search_scenario,)}
                )
            }
        ),
    )

    pair = FinanceJudgeViewBuilder(alias_salt=b"n" * 32).build(
        JudgeViewRequest((direct, search_only))
    )

    assert pair.candidate_1.scenarios[0].alias == "candidate_1_scenario_001"
    assert pair.candidate_2.scenarios[0].alias == "candidate_2_scenario_001"


def test_pair_is_order_independent_bounded_and_has_no_dangling_aliases() -> None:
    # Given
    search_only = _source(ForecastArm.SEARCH_ONLY, make_judge_evidence())
    search_dag = _source(ForecastArm.SEARCH_DAG, make_judge_evidence())
    builder = FinanceJudgeViewBuilder(alias_salt=b"r" * 32)

    # When
    forward = builder.build(JudgeViewRequest((search_only, search_dag)))
    reverse = builder.build(JudgeViewRequest((search_dag, search_only)))

    # Then
    assert forward == reverse
    memory = forward.memory[0]
    assert (len(memory.nodes), len(memory.edges), len(memory.impacts)) == (12, 16, 8)
    assert (
        memory.omitted_node_count,
        memory.omitted_edge_count,
        memory.omitted_impact_count,
    ) == (2, 2, 2)
    assert memory.edges[0].alias == "memory_001_edge_002"
    assert memory.impacts[0].alias == "memory_001_impact_002"
    node_aliases = {node.alias for node in memory.nodes}
    edge_aliases = {edge.alias for edge in memory.edges}
    assert all(
        edge.source_node_alias in node_aliases
        and edge.target_node_alias in node_aliases
        for edge in memory.edges
    )
    assert all(
        impact.event_alias in node_aliases
        and impact.outcome_event_alias in node_aliases
        and set(impact.causal_edge_aliases).issubset(edge_aliases)
        for impact in memory.impacts
    )


def test_pair_strips_identities_hashes_and_preserves_historical_grounding() -> None:
    # Given
    pair = FinanceJudgeViewBuilder(alias_salt=b"h" * 32).build(
        JudgeViewRequest(
            (
                _source(ForecastArm.SEARCH_ONLY, make_judge_evidence()),
                _source(ForecastArm.SEARCH_DAG, make_judge_evidence()),
            )
        )
    )

    # When
    serialized = pair.model_dump_json()
    canonical = build_blind_judge_payload(pair, JudgeCallOrientation.CANONICAL)
    swapped = build_blind_judge_payload(pair, JudgeCallOrientation.SWAPPED)

    # Then
    assert all(
        forbidden not in serialized
        for forbidden in (EVIDENCE_SENTINEL, DAG_SENTINEL, "current-target", "a" * 64)
    )
    assert "content_hash" not in serialized
    assert pair.memory[0].question.question_text == "Historical rate question"
    assert pair.memory[0].resolved_historical_outcome == (
        "Resolved Yes under memory_001_episode"
    )
    assert canonical.answer_a.alias is NeutralCandidate.CANDIDATE_1
    assert swapped.answer_a.alias is NeutralCandidate.CANDIDATE_2


def test_view_rejects_target_abstention_timing_and_hash_conflicts() -> None:
    # Given
    good = _source(ForecastArm.SEARCH_ONLY, make_judge_evidence())
    mismatched = _source(
        ForecastArm.SEARCH_DAG,
        make_judge_evidence(),
        make_judge_target("different-target"),
    )
    cutoff_evidence = make_judge_evidence().model_copy(
        update={
            "available_at": make_judge_target().cutoff,
            "retrieved_at": make_judge_target().cutoff,
        }
    )
    post_cutoff = _source(ForecastArm.SEARCH_DAG, cutoff_evidence)
    conflicting = _source(
        ForecastArm.SEARCH_DAG,
        make_judge_evidence("b" * 64),
    )
    assert isinstance(good.arm_result, ArmSucceeded)
    abstained = JudgeCandidateSource(
        arm_result=ArmUnavailable(
            arm=good.arm_result.arm,
            attempts=(),
            treatment=good.arm_result.treatment,
            reason=ArmUnavailableReason.NOT_ATTEMPTED,
        ),
        forecast_input=good.forecast_input,
    )
    cases = (
        ((good, mismatched), JudgeViewFailureReason.TARGET_MISMATCH),
        ((abstained, conflicting), JudgeViewFailureReason.ARM_NOT_SUCCEEDED),
        ((good, post_cutoff), JudgeViewFailureReason.POST_CUTOFF_EVIDENCE),
        ((good, conflicting), JudgeViewFailureReason.CONFLICTING_EVIDENCE),
    )

    # When / Then
    for candidates, reason in cases:
        with pytest.raises(JudgeViewError) as captured:
            FinanceJudgeViewBuilder(alias_salt=b"x" * 32).build(
                JudgeViewRequest(candidates)
            )
        assert captured.value.reason is reason


def test_view_rejects_transient_raw_body_and_invalid_salt() -> None:
    # Given
    raw_body = "unsafe exact provider body"
    raw_source = _source(ForecastArm.SEARCH_DAG, make_judge_evidence())
    assert isinstance(raw_source.arm_result, ArmSucceeded)
    raw_forecast = raw_source.arm_result.forecast.model_copy(
        update={"explanation": raw_body}
    )
    raw_source = replace(
        raw_source,
        arm_result=raw_source.arm_result.model_copy(update={"forecast": raw_forecast}),
    )
    request = JudgeViewRequest(
        (
            _source(ForecastArm.SEARCH_ONLY, make_judge_evidence()),
            raw_source,
        ),
        transient_forbidden=TransientForbiddenValueRegistry(
            raw_values=(raw_body,),
        ),
    )

    # When / Then
    with pytest.raises(JudgeViewError) as raw_error:
        FinanceJudgeViewBuilder(alias_salt=b"x" * 32).build(request)
    assert raw_error.value.reason is JudgeViewFailureReason.FORBIDDEN_CONTENT
    with pytest.raises(JudgeViewError) as salt_error:
        FinanceJudgeViewBuilder(alias_salt=b"x" * 31).build(
            JudgeViewRequest(request.candidates)
        )
    assert salt_error.value.reason is JudgeViewFailureReason.INVALID_ALIAS_SALT
