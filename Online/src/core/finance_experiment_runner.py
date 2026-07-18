"""DAG-independent fixed-pack finance experiment orchestration."""

from dataclasses import dataclass

from src.agents.finance_forecast_agent import FinanceForecastAgent
from src.agents.finance_search_agent import FinanceSearchAgent
from src.core.finance_experiment_runtime import (
    FinanceExperimentRequestError,
    FinanceExperimentRetrievalError,
    FinanceExperimentStage,
    FinanceExperimentTrialRequest,
    FinanceExperimentTrialRun,
    HistoricalRetriever,
    SeedIdentityVerifier,
)
from src.domain.finance.experiment import (
    ArmResult,
    ArmUnavailableReason,
    FinanceEvidenceSnapshot,
    FinanceExperimentManifest,
    FinanceExperimentTrial,
    ForecastArm,
)
from src.domain.finance.memory import ResolvedDagEpisode
from src.domain.finance.pipeline import (
    SearchMergeInput,
    SearchPassFailed,
    retrieval_matches_canonical_memory,
)
from src.domain.finance.provider import InitialSearchRequest, SearchQueryIntent
from src.domain.finance.retrieval import (
    DuplicateEpisodeIdError,
    HistoricalDagQuery,
    HistoricalEpisodeSource,
    SeedAssetMismatchError,
    SeedDatabaseReadError,
    SeedDataError,
    SeedSchemaError,
    TopK,
    UnsafeSeedPathError,
    WritableSeedAssetError,
)
from src.domain.finance.search import EvidencePack, TargetProfile
from src.services.finance_experiment_arm import (
    FinanceArmForecastRequest,
    direct_treatment,
    execute_finance_forecast_arm,
    memory_unavailable,
    snapshot_treatment,
    target_profile,
    unavailable_search_arms,
)


@dataclass(frozen=True, slots=True)
class FinanceExperimentRunner:
    """Run Direct, Search-only, and Search+DAG as recoverable siblings."""

    searcher: FinanceSearchAgent
    forecaster: FinanceForecastAgent
    repository: HistoricalEpisodeSource
    retriever: HistoricalRetriever
    identity_verifier: SeedIdentityVerifier

    def run_trial(
        self,
        request: FinanceExperimentTrialRequest,
    ) -> FinanceExperimentTrialRun:
        """Run one fixed-pack trial after manifest and DB identity preflight."""
        manifest = FinanceExperimentManifest.model_validate_json(
            request.manifest.model_dump_json()
        )
        if request.question not in manifest.questions:
            raise FinanceExperimentRequestError("question is absent from manifest")
        if not 0 <= request.repetition_index < manifest.repetitions:
            raise FinanceExperimentRequestError("repetition index is out of range")
        self.identity_verifier.verify_identity()
        target = target_profile(manifest, request.question)
        stages = [FinanceExperimentStage.METADATA_PREFLIGHT]
        arms: list[ArmResult] = []

        def terminal(
            snapshot: FinanceEvidenceSnapshot | None,
        ) -> FinanceExperimentTrialRun:
            trial = FinanceExperimentTrial(
                trial_id=request.trial_id,
                question_id=request.question.question_id,
                repetition_index=request.repetition_index,
                preparation_attempts=(),
                arms=tuple(arms),
            )
            return FinanceExperimentTrialRun(
                trial=trial,
                stage_order=tuple(stages),
                evidence_snapshot=snapshot,
            )

        empty_pack = EvidencePack(items=(), historical_dag_references=())
        stages.append(FinanceExperimentStage.DIRECT)
        arms.append(
            execute_finance_forecast_arm(
                self.forecaster,
                FinanceArmForecastRequest(
                    arm=ForecastArm.DIRECT,
                    target_profile=target,
                    evidence_pack=empty_pack,
                    historical_memory=(),
                    treatment=direct_treatment(),
                ),
            )
        )

        stages.append(FinanceExperimentStage.INITIAL_SEARCH)
        initial = self.searcher.search(
            InitialSearchRequest(
                run_mode=manifest.run_mode,
                target_profile=target,
                source_policy=manifest.search.source_policy,
                query_intents=tuple(SearchQueryIntent)[:4],
            )
        )
        if isinstance(initial, SearchPassFailed):
            arms.extend(
                unavailable_search_arms(ArmUnavailableReason.SEARCH_UNAVAILABLE)
            )
            return terminal(None)
        admitted = self.searcher.merge_and_admit(
            SearchMergeInput(
                target_profile=target,
                historical_dag_references=(),
                passes=(initial,),
            )
        )

        snapshot = FinanceEvidenceSnapshot.from_evidence_pack(
            admitted.searcher_result.evidence_pack
        )
        stages.append(FinanceExperimentStage.FREEZE)
        if not snapshot.items:
            arms.extend(
                unavailable_search_arms(ArmUnavailableReason.NO_ADMITTED_EVIDENCE)
            )
            return terminal(snapshot)
        frozen_pack = EvidencePack(
            items=snapshot.items,
            historical_dag_references=(),
        )
        search_audit = snapshot_treatment(ForecastArm.SEARCH_ONLY, snapshot, 0)
        stages.append(FinanceExperimentStage.SEARCH_ONLY)
        arms.append(
            execute_finance_forecast_arm(
                self.forecaster,
                FinanceArmForecastRequest(
                    arm=ForecastArm.SEARCH_ONLY,
                    target_profile=target,
                    evidence_pack=frozen_pack,
                    historical_memory=(),
                    treatment=search_audit,
                ),
            )
        )

        stages.append(FinanceExperimentStage.MEMORY_RETRIEVAL)
        memory = self._load_selected_memory(manifest, target)
        if not memory:
            arms.append(memory_unavailable())
            return terminal(snapshot)
        dag_audit = snapshot_treatment(
            ForecastArm.SEARCH_DAG,
            snapshot,
            len(memory),
        )
        stages.append(FinanceExperimentStage.SEARCH_DAG)
        arms.append(
            execute_finance_forecast_arm(
                self.forecaster,
                FinanceArmForecastRequest(
                    arm=ForecastArm.SEARCH_DAG,
                    target_profile=target,
                    evidence_pack=frozen_pack,
                    historical_memory=memory,
                    treatment=dag_audit,
                ),
            )
        )
        return terminal(snapshot)

    def _load_selected_memory(
        self,
        manifest: FinanceExperimentManifest,
        target: TargetProfile,
    ) -> tuple[ResolvedDagEpisode, ...]:
        try:
            episodes = self.repository.load_episodes()
        except (
            DuplicateEpisodeIdError,
            SeedAssetMismatchError,
            SeedDataError,
            SeedDatabaseReadError,
            SeedSchemaError,
            UnsafeSeedPathError,
            WritableSeedAssetError,
        ):
            return ()
        if not episodes:
            return ()
        top_k = TopK(manifest.retrieval.top_k)
        try:
            result = self.retriever.retrieve_from(
                episodes,
                HistoricalDagQuery(
                    target_profile=target,
                    policy=manifest.retrieval.eligibility_policy,
                    top_k=top_k,
                ),
            )
        except FinanceExperimentRetrievalError:
            return ()
        if not retrieval_matches_canonical_memory(episodes, result, top_k):
            return ()
        return tuple(item.episode for item in result.selected)


__all__ = [
    "FinanceExperimentRunner",
    "FinanceExperimentRequestError",
    "FinanceExperimentRetrievalError",
    "FinanceExperimentStage",
    "FinanceExperimentTrialRequest",
    "FinanceExperimentTrialRun",
    "HistoricalRetriever",
    "SeedIdentityVerifier",
]
