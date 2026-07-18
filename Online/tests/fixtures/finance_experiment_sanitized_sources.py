"""Sentinel-bearing sources projected through the real Todo 3 sanitizer."""

from src.domain.finance.experiment import ArmSucceeded, ForecastArm, TreatmentAudit
from src.domain.finance.experiment_snapshot import FinanceEvidenceSnapshot
from src.domain.finance.forecast import (
    ForecasterInput,
    ForecastResult,
    OutcomeProbability,
    Scenario,
)
from src.domain.finance.judge_views import (
    SanitizedCandidateView,
    SanitizedEvidenceView,
    SanitizedMemoryView,
    SanitizedTargetView,
)
from src.domain.finance.memory import OutcomeLabel, ScenarioId
from src.domain.finance.sanitized_artifact import (
    FinanceAliasAudit,
    TransientForbiddenValueRegistry,
)
from src.domain.finance.search import EvidenceItem, EvidencePack, TargetProfile
from src.services.finance_judge_view import (
    FinanceJudgeViewBuilder,
    JudgeCandidateSource,
    JudgeViewRequest,
)
from tests.unit.domain.finance._experiment_factories import make_manifest
from tests.unit.domain.finance.test_judging import (
    DAG_SENTINEL,
    EVIDENCE_SENTINEL,
    make_bounded_judge_episode,
    make_judge_evidence,
)


def _candidate_source(
    arm: ForecastArm,
    evidence: EvidenceItem,
    target: TargetProfile,
) -> JudgeCandidateSource:
    episode = make_bounded_judge_episode()
    memory = (episode,) if arm is ForecastArm.SEARCH_DAG else ()
    pack = EvidencePack(items=(evidence,), historical_dag_references=())
    outcomes = (
        OutcomeProbability(label=OutcomeLabel("No"), probability=0.4),
        OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.6),
    )
    forecast = ForecastResult(
        scenarios=(
            Scenario(
                scenario_id=ScenarioId(f"scenario-{arm.value}"),
                name=f"Scenario for {arm.value}",
                reasoning_steps=(f"Use {EVIDENCE_SENTINEL} and {DAG_SENTINEL}.",),
                probability=1.0,
                conditional_outcomes=outcomes,
                evidence_ids=(evidence.evidence_id,),
                historical_dag_references=(episode.reference,) if memory else (),
                assumptions=(f"Assume {arm.value} remains relevant.",),
                triggers=(f"Trigger {EVIDENCE_SENTINEL}.",),
                disconfirmers=(f"Disconfirm {DAG_SENTINEL}.",),
                uncertainty="Residual uncertainty remains.",
            ),
        ),
        outcome_probabilities=outcomes,
        explanation=f"Explanation from {arm.name} with {evidence.content_hash}.",
    )
    snapshot = FinanceEvidenceSnapshot(items=(evidence,))
    return JudgeCandidateSource(
        arm_result=ArmSucceeded(
            arm=arm,
            attempts=(),
            treatment=TreatmentAudit(
                arm=arm,
                evidence_snapshot_digest=snapshot.digest,
                evidence_snapshot_bytes=snapshot.byte_length,
                evidence_item_count=1,
                historical_memory_episode_count=len(memory),
            ),
            forecast=forecast,
        ),
        forecast_input=ForecasterInput(
            target_profile=target,
            evidence_pack=pack,
            historical_memory=memory,
        ),
    )


def make_source_sanitized_pair() -> tuple[
    SanitizedTargetView,
    tuple[SanitizedEvidenceView, ...],
    tuple[SanitizedMemoryView, ...],
    SanitizedCandidateView,
    SanitizedCandidateView,
    FinanceAliasAudit,
]:
    """Sanitize original IDs while transient body/credential values stay absent."""
    manifest = make_manifest()
    question = manifest.questions[0]
    target = TargetProfile(
        question_id=question.question_id,
        question_text=question.question_text,
        question_type=question.question_type,
        domain=question.domain,
        context=question.context,
        cutoff=manifest.cutoff,
        outcome_space=question.outcome_space,
        resolution_rule=question.resolution_rule,
    )
    raw_body = "RAW_BODY_SENTINEL_91bd"
    credential = "sk-" + "or-v1-" + "Q" * 16
    pair = FinanceJudgeViewBuilder(alias_salt=b"e" * 32).build(
        JudgeViewRequest(
            candidates=(
                _candidate_source(
                    ForecastArm.SEARCH_ONLY,
                    make_judge_evidence(),
                    target,
                ),
                _candidate_source(
                    ForecastArm.SEARCH_DAG,
                    make_judge_evidence(),
                    target,
                ),
            ),
            transient_forbidden=TransientForbiddenValueRegistry(
                raw_values=(raw_body, credential),
            ),
        )
    )
    return (
        pair.target,
        pair.evidence,
        pair.memory,
        pair.candidate_1,
        pair.candidate_2,
        pair.alias_audit,
    )


__all__ = ["make_source_sanitized_pair"]
