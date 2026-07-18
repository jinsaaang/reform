"""Pure hard eligibility and deterministic retrieval result values."""

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final, Protocol, assert_never

from typing_extensions import override

from src.domain.finance.memory import QuestionId, RelationState, ResolvedDagEpisode
from src.domain.finance.search import TargetProfile


@dataclass(frozen=True, slots=True)
class FinanceSeedAssetSpec:
    """Expected immutable asset identity and canonical episode count."""

    byte_size: int
    sha256: str
    schema_version: int
    episode_count: int


FINANCE_SEED_V1_SPEC: Final = FinanceSeedAssetSpec(
    byte_size=55_554_048,
    sha256="94ffd8cca51906edec0b05f7e94e78de80d26f268f082521bded80d0aed06fab",
    schema_version=8,
    episode_count=37,
)


@dataclass(frozen=True, slots=True)
class UnsafeSeedPathError(Exception):
    path: Path

    @override
    def __str__(self) -> str:
        return f"seed location must be a filesystem path, got {self.path}"


@dataclass(frozen=True, slots=True)
class SeedAssetMismatchError(Exception):
    path: Path
    attribute: str
    expected: int | str
    actual: int | str

    @override
    def __str__(self) -> str:
        return (
            f"seed {self.attribute} mismatch for {self.path}: "
            f"expected {self.expected}, got {self.actual}"
        )


@dataclass(frozen=True, slots=True)
class WritableSeedAssetError(Exception):
    path: Path

    @override
    def __str__(self) -> str:
        return f"seed asset must be read-only: {self.path}"


@dataclass(frozen=True, slots=True)
class SeedSchemaError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"seed schema invalid for {self.path}: {self.detail}"


@dataclass(frozen=True, slots=True)
class SeedDataError(Exception):
    table: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"invalid seed data in {self.table}: {self.detail}"


@dataclass(frozen=True, slots=True)
class DuplicateEpisodeIdError(Exception):
    question_id: QuestionId

    @override
    def __str__(self) -> str:
        return f"duplicate canonical finance question ID: {self.question_id}"


@dataclass(frozen=True, slots=True)
class SeedDatabaseReadError(Exception):
    path: Path
    detail: str

    @override
    def __str__(self) -> str:
        return f"unable to read immutable seed {self.path}: {self.detail}"


@unique
class EligibilityPolicy(StrEnum):
    """Explicit policy for unavailable public relation metadata."""

    STRICT = "strict"
    PUBLIC_DB_BOOTSTRAP = "public_db_bootstrap"


@unique
class ExclusionReason(StrEnum):
    """Hard reasons that remove a candidate before ranking."""

    RESOLUTION_NOT_STRICTLY_BEFORE_CUTOFF = "resolution_not_strictly_before_cutoff"
    SAME_UNDERLYING_EVENT = "same_underlying_event"
    SHARED_RESOLUTION = "shared_resolution"
    DERIVED_QUESTION = "derived_question"
    NEAR_DUPLICATE = "near_duplicate"
    RELATION_METADATA_UNAVAILABLE = "relation_metadata_unavailable"


@unique
class AuditMarker(StrEnum):
    """Non-exclusion audit markers retained on selected candidates."""

    UNVERIFIED_RELATION_METADATA = "unverified_relation_metadata"


@unique
class RetrievalFeature(StrEnum):
    """Closed lexical feature groups used by the baseline ranker."""

    QUESTION = "question"
    CONTEXT = "context"
    DOMAIN = "domain"
    QUESTION_TYPE = "question_type"


@dataclass(frozen=True, slots=True)
class InvalidTopKError(Exception):
    """Raised when a requested retrieval bound is not a positive integer."""

    supplied_value: int | str

    @override
    def __str__(self) -> str:
        return f"top-k must be positive, got {self.supplied_value!r}"


@dataclass(frozen=True, slots=True)
class TopK:
    """Parsed positive bound for deterministic historical retrieval."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise InvalidTopKError(supplied_value=self.value)

    @classmethod
    def parse(cls, raw_value: int | str) -> "TopK":
        """Parse an integer or decimal text into a positive bound."""
        match raw_value:
            case bool():
                raise InvalidTopKError(supplied_value=raw_value)
            case int() as value:
                return cls(value=value)
            case str() as text:
                try:
                    return cls(value=int(text))
                except ValueError as error:
                    raise InvalidTopKError(supplied_value=text) from error
            case _ as unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class HistoricalDagQuery:
    """Complete typed input to historical DAG retrieval."""

    target_profile: TargetProfile
    policy: EligibilityPolicy
    top_k: TopK


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """Hard-gate result recorded before any lexical score exists."""

    exclusion_reasons: tuple[ExclusionReason, ...]
    audit_markers: tuple[AuditMarker, ...]

    @property
    def eligible(self) -> bool:
        """Return whether the candidate may enter the ranking pool."""
        return not self.exclusion_reasons


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    """Deterministic contribution from one target feature group."""

    feature: RetrievalFeature
    weight: int
    matched_terms: tuple[str, ...]

    @property
    def contribution(self) -> int:
        """Return the integer lexical contribution without floating drift."""
        return self.weight * len(self.matched_terms)


@dataclass(frozen=True, slots=True)
class RankedHistoricalDag:
    """Eligible historical episode with reproducible score detail."""

    episode: ResolvedDagEpisode
    score: int
    score_components: tuple[ScoreComponent, ...]
    matched_terms: tuple[str, ...]
    audit_markers: tuple[AuditMarker, ...]


@dataclass(frozen=True, slots=True)
class ExcludedHistoricalDag:
    """Hard-excluded episode without a ranking score."""

    episode: ResolvedDagEpisode
    decision: EligibilityDecision


@dataclass(frozen=True, slots=True)
class HistoricalDagRetrieval:
    """Complete selected, ranked, and hard-excluded retrieval audit."""

    selected: tuple[RankedHistoricalDag, ...]
    ranked_candidates: tuple[RankedHistoricalDag, ...]
    excluded: tuple[ExcludedHistoricalDag, ...]


class HistoricalEpisodeSource(Protocol):
    """Read-only source of canonical resolved historical episodes."""

    def load_episodes(self) -> tuple[ResolvedDagEpisode, ...]:
        """Load episodes in stable source order."""
        ...


def assess_eligibility(
    episode: ResolvedDagEpisode,
    target_profile: TargetProfile,
    policy: EligibilityPolicy,
) -> EligibilityDecision:
    """Apply temporal and relation hard gates before ranking."""
    reasons: list[ExclusionReason] = []
    if episode.historical_resolution.resolution_date_proxy >= target_profile.cutoff:
        reasons.append(ExclusionReason.RESOLUTION_NOT_STRICTLY_BEFORE_CUTOFF)

    relation_checks = (
        (
            episode.relation_metadata.same_underlying_event,
            ExclusionReason.SAME_UNDERLYING_EVENT,
        ),
        (
            episode.relation_metadata.shared_resolution,
            ExclusionReason.SHARED_RESOLUTION,
        ),
        (
            episode.relation_metadata.derived_question,
            ExclusionReason.DERIVED_QUESTION,
        ),
        (
            episode.relation_metadata.near_duplicate,
            ExclusionReason.NEAR_DUPLICATE,
        ),
    )
    metadata_unavailable = False
    for state, known_true_reason in relation_checks:
        match state:
            case RelationState.KNOWN_TRUE:
                reasons.append(known_true_reason)
            case RelationState.KNOWN_FALSE:
                continue
            case RelationState.UNAVAILABLE:
                metadata_unavailable = True
            case _ as unreachable:
                assert_never(unreachable)

    markers: tuple[AuditMarker, ...] = ()
    match policy:
        case EligibilityPolicy.STRICT:
            if metadata_unavailable:
                reasons.append(ExclusionReason.RELATION_METADATA_UNAVAILABLE)
        case EligibilityPolicy.PUBLIC_DB_BOOTSTRAP:
            if metadata_unavailable and not reasons:
                markers = (AuditMarker.UNVERIFIED_RELATION_METADATA,)
        case _ as unreachable:
            assert_never(unreachable)

    return EligibilityDecision(
        exclusion_reasons=tuple(reasons),
        audit_markers=markers,
    )
