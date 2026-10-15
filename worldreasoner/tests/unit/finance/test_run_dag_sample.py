"""Unit tests for the finance DAG sample runner helpers."""

import sys
from datetime import datetime, timezone

import pytest

from scripts.finance.run_dag_sample import (
    associate_family_articles,
    create_structured_resolution_evidence,
    create_round_search_tracker,
    compact_pipeline_results,
    get_finance_category,
    parse_args,
    read_questions,
    select_questions,
    should_collect_resolution_fallback,
    should_run_near_threshold_rescue,
    validate_resolved_question,
)
from src.core.database import GenericDatabase
from src.domain.models import Article, Domain
from src.domain.models import Question
from src.services.question_monitor_service import QuestionMonitorService
from src.config.pipeline import EvidenceSatisfactionConfig
from src.pipelines.base import PipelineStageResult, PipelineStageStatus
from src.pipelines.prompts.hindsight_causal_analysis import get_prompt
from src.tools.inspectors.article_inspector import ArticleInspectorTool
from tests.conftest import create_test_question


def test_compact_pipeline_results_omits_article_content():
    article = Article(
        id="art_finance_20260430_001_test",
        title="Apple results",
        content="large body " * 1_000,
        url="https://example.com/apple-results",
        source="Example",
        published_date=datetime(2026, 4, 30, tzinfo=timezone.utc),
        domain=Domain.FINANCE,
    )
    now = datetime.now(timezone.utc)
    result = PipelineStageResult(
        stage_name="AgentBasedEvidence",
        status=PipelineStageStatus.COMPLETED,
        items_processed=1,
        items_output=1,
        outputs=[article],
        started_at=now,
        completed_at=now,
    )

    compact = compact_pipeline_results([result])

    assert "outputs" not in compact[0]
    assert compact[0]["output_ids"] == [article.id]
    assert "large body" not in str(compact)


def test_finance_runner_defaults_to_hybrid_agent_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dag_sample.py"])

    args = parse_args()

    assert args.agent_mode == "hybrid"
    assert args.min_evidence_articles == 10
    assert args.min_graph_events == 8
    assert args.graph_agent_max_steps == 12
    assert args.graph_agent_max_output_tokens == 24_000
    assert args.category is None


@pytest.mark.parametrize("article_count", [8, 9])
def test_near_threshold_rescue_only_allows_covered_near_misses(article_count):
    assert should_run_near_threshold_rescue(article_count, {"gaps": ["missing"]})


@pytest.mark.parametrize("article_count", [0, 7, 10])
def test_near_threshold_rescue_rejects_other_article_counts(article_count):
    assert not should_run_near_threshold_rescue(article_count, {"gaps": ["missing"]})
    assert not should_run_near_threshold_rescue(8, {"gaps": []})


def test_resolution_fallback_runs_for_any_unsatisfied_evidence_count():
    assert should_collect_resolution_fallback(False)
    assert not should_collect_resolution_fallback(True)


def test_normal_evidence_rounds_receive_independent_query_budgets(tmp_path):
    first = create_round_search_tracker(tmp_path / "evidence.db", "q", 10, 2)
    second = create_round_search_tracker(tmp_path / "evidence.db", "q", 10, 2)

    assert first is not second
    assert first.allow_search("first query")[0]
    first.record_search(query="first query")
    assert first.allow_search("second query")[0]
    first.record_search(query="second query")
    assert not first.allow_search("third query")[0]
    assert second.allow_search("third query")[0]
    assert second.max_queries == 2


def test_hindsight_prompt_has_upper_cutoff_but_no_lower_date_bound():
    question = create_test_question(
        estimated_start_time=datetime(2024, 3, 1, tzinfo=timezone.utc),
        resolution_date=datetime(2024, 4, 15, tzinfo=timezone.utc),
        ground_truth=True,
    )

    prompt = get_prompt(
        question,
        min_evidence_articles=10,
        evidence_window_days=45,
        min_graph_depth=3,
        confidence_threshold=0.7,
    )

    assert "There is no lower date bound" in prompt
    assert "older filings, official" in prompt
    assert "strictly before 2024-04-15" in prompt


def test_select_questions_filters_category_before_limit():
    questions = [
        create_test_question(
            id="q_macro_1",
            metadata={"finfactorbench": {"original_domain": "macro"}},
        ),
        create_test_question(
            id="q_earnings_1",
            metadata={
                "finfactorbench": {"original_domain": "corporate_earnings"}
            },
        ),
        create_test_question(
            id="q_macro_2",
            metadata={"finfactorbench": {"original_domain": "macro"}},
        ),
    ]

    selected = select_questions(questions, category="macro", limit=1)

    assert [question.id for question in selected] == ["q_macro_1"]
    assert get_finance_category(selected[0]) == "macro"


def test_select_questions_reads_active_finance_category():
    questions = [
        create_test_question(
            id="q_macro_active",
            metadata={"finance": {"category": "macro"}},
        ),
        create_test_question(
            id="q_market_active",
            metadata={"finance": {"category": "market_fx_credit"}},
        ),
    ]

    selected = select_questions(questions, category="macro", limit=10)

    assert [question.id for question in selected] == ["q_macro_active"]


def test_select_questions_supports_explicit_question_ids():
    questions = [
        create_test_question(
            id="q_macro_1",
            metadata={"finfactorbench": {"original_domain": "macro"}},
        ),
        create_test_question(
            id="q_earnings_1",
            metadata={
                "finfactorbench": {"original_domain": "corporate_earnings"}
            },
        ),
    ]

    selected = select_questions(
        questions,
        category=None,
        limit=10,
        question_ids=["q_earnings_1"],
    )

    assert [question.id for question in selected] == ["q_earnings_1"]


def test_finance_category_allows_missing_metadata():
    question = create_test_question(metadata=None)

    assert get_finance_category(question) is None


def test_resolved_question_validation_rejects_unsupported_timeframe():
    question = create_test_question(
        id="q_timeframe",
        question_type="timeframe",
        ground_truth="2026-04-01",
    )

    with pytest.raises(ValueError, match="unsupported timeframe"):
        validate_resolved_question(question)


def test_read_questions_rejects_duplicate_ids(tmp_path):
    question = create_test_question(id="duplicate", ground_truth=True)
    serialized = question.model_dump_json()
    path = tmp_path / "duplicates.jsonl"
    path.write_text(f"{serialized}\n{serialized}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate question ID"):
        read_questions(path)


def test_select_questions_shards_without_overlap():
    questions = [create_test_question(id=f"q_{index}") for index in range(7)]

    shards = [
        select_questions(
            questions,
            category=None,
            limit=7,
            shard_index=index,
            shard_count=3,
        )
        for index in range(3)
    ]

    ids = [[question.id for question in shard] for shard in shards]
    assert ids == [["q_0", "q_1", "q_2"], ["q_3", "q_4", "q_5"], ["q_6"]]
    assert len({question_id for shard in ids for question_id in shard}) == 7


def test_family_articles_are_reused_without_changing_original_owner(tmp_path):
    db = GenericDatabase(str(tmp_path / "family.sqlite"))
    db.create_table(Question)
    db.create_table(Article)
    metadata = {"finfactorbench": {"resolution_doc_id": "sec:adbe:2026q1"}}
    window_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    resolution = datetime(2026, 2, 1, tzinfo=timezone.utc)
    first = create_test_question(
        id="q_first",
        metadata=metadata,
        estimated_start_time=window_start,
        resolution_date=resolution,
    )
    second = create_test_question(
        id="q_second",
        metadata=metadata,
        estimated_start_time=window_start,
        resolution_date=resolution,
    )
    db.save(Question, first)
    db.save(Question, second)
    article = Article(
        id="art_finance_20240101_001_family",
        title="Adobe reports quarterly financial results",
        content="Quarterly financial results and operating details. " * 10,
        url="https://example.com/adobe-results",
        source="Example",
        published_date=window_start,
        domain=Domain.FINANCE,
        collected_for_question_id=first.id,
        metadata={"related_question_ids": [first.id]},
    )
    db.save(Article, article)

    assert associate_family_articles(db, second) == 1
    saved = db.get(Article, article.id)
    assert saved.collected_for_question_id == first.id
    assert set(saved.metadata["related_question_ids"]) == {first.id, second.id}
    monitor = QuestionMonitorService(
        db, EvidenceSatisfactionConfig(min_articles=1)
    )
    assert monitor.check_satisfaction(second.id).article_count == 1


def test_monetary_period_articles_are_reused_across_factor_families(tmp_path):
    db = GenericDatabase(str(tmp_path / "monetary-period.sqlite"))
    db.create_table(Question)
    db.create_table(Article)
    resolution = datetime(2025, 3, 15, tzinfo=timezone.utc)
    first = create_test_question(
        id="q_two_year",
        metadata={
            "finance": {
                "event_cluster_id": "treasury_2y_2025-02",
                "category": "monetary_policy",
                "region": "US",
                "target_period": "2025-02",
            }
        },
        resolution_date=resolution,
    )
    second = create_test_question(
        id="q_real_yield",
        metadata={
            "finance": {
                "event_cluster_id": "treasury_real_10y_2025-02",
                "category": "monetary_policy",
                "region": "US",
                "target_period": "2025-02",
            }
        },
        resolution_date=resolution,
    )
    db.save(Question, first)
    db.save(Question, second)
    db.save(
        Article,
        Article(
            id="art_finance_20250201_001_policy",
            title="Federal Reserve policy and Treasury yields",
            content="Federal Reserve policy, inflation, and Treasury yields. " * 10,
            url="https://example.com/fed-treasury-yields",
            source="Example",
            published_date=datetime(2025, 2, 1, tzinfo=timezone.utc),
            domain=Domain.FINANCE,
            collected_for_question_id=first.id,
            metadata={"related_question_ids": [first.id]},
        )
    )

    assert associate_family_articles(db, second) == 1
    saved = db.get(Article, "art_finance_20250201_001_policy")
    assert set(saved.metadata["related_question_ids"]) == {first.id, second.id}


def test_finance_inspector_keeps_evidence_older_than_question_start(tmp_path):
    db_path = tmp_path / "old-evidence.sqlite"
    db = GenericDatabase(str(db_path))
    db.create_table(Question)
    db.create_table(Article)
    question = create_test_question(
        id="q_intel",
        estimated_start_time=datetime(2026, 3, 28, tzinfo=timezone.utc),
        resolution_date=datetime(2026, 4, 19, tzinfo=timezone.utc),
        metadata={"finfactorbench": {"original_domain": "corporate_earnings"}},
    )
    db.save(Question, question)
    db.save(
        Article,
        Article(
            id="art_intel_10k",
            title="Intel 2025 annual report",
            content="Historical filing evidence. " * 10,
            url="https://example.com/intel-10k",
            source="SEC EDGAR",
            published_date=datetime(2026, 1, 23, tzinfo=timezone.utc),
            domain=Domain.FINANCE,
            collected_for_question_id=question.id,
        ),
    )

    report = ArticleInspectorTool(
        db_path=str(db_path),
        question_id=question.id,
        satisfaction_config=EvidenceSatisfactionConfig(min_articles=1),
        require_causal_explanation=False,
    ).forward()

    assert "Total Articles" in report
    assert "STATUS: No Articles Collected" not in report


def test_structured_official_fallback_is_auditable_and_deterministic(tmp_path):
    db = GenericDatabase(str(tmp_path / "official.sqlite"))
    db.create_table(Question)
    db.create_table(Article)
    start = datetime(2025, 12, 1, tzinfo=timezone.utc)
    resolution = datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc)
    question = create_test_question(
        id="fred_dgs10_test",
        estimated_start_time=start,
        resolution_date=resolution,
        resolution_criteria="Use the final DGS10 observation.",
        resolution_reasoning="The official value was 4.18.",
        metadata={
            "finfactorbench": {
                "source_quality": "high",
                "resolution_value": "4.18",
                "resolution_document": {
                    "published_at": "2025-12-31",
                    "publisher": "Federal Reserve Bank of St. Louis",
                    "title": "FRED DGS10 official series",
                    "url": "https://fred.stlouisfed.org/series/DGS10",
                },
            }
        },
    )
    db.save(Question, question)

    created = create_structured_resolution_evidence(db, question)
    repeated = create_structured_resolution_evidence(db, question)

    assert created["status"] == "structured_created"
    assert repeated == {"status": "structured_existing", "id": created["id"]}
    article = db.get(Article, created["id"])
    assert article.metadata["structured_resolution_evidence"] is True
    assert "not model-generated evidence" in article.content


def test_structured_official_fallback_expands_short_title(tmp_path):
    db = GenericDatabase(str(tmp_path / "short-title.sqlite"))
    db.create_table(Question)
    db.create_table(Article)
    question = create_test_question(
        id="fred_obfr_test",
        estimated_start_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
        resolution_date=datetime(2026, 3, 31, 23, 59, tzinfo=timezone.utc),
        metadata={
            "finfactorbench": {
                "source_quality": "high",
                "resolution_value": "3.64",
                "resolution_document": {
                    "published_at": "2026-03-31",
                    "publisher": "Federal Reserve Bank of St. Louis",
                    "title": "FRED OBFR",
                    "url": "https://fred.stlouisfed.org/series/OBFR",
                },
            }
        },
    )
    db.save(Question, question)

    created = create_structured_resolution_evidence(db, question)

    article = db.get(Article, created["id"])
    assert article.title == "FRED OBFR official series"


def test_select_questions_rejects_missing_category():
    questions = [
        create_test_question(
            metadata={"finfactorbench": {"original_domain": "macro"}}
        )
    ]

    with pytest.raises(ValueError, match="available category counts"):
        select_questions(questions, category="corporate_earnings", limit=1)


def test_finance_runner_accepts_explicit_category(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_dag_sample.py", "--category", "corporate_earnings", "--limit", "1"],
    )

    args = parse_args()

    assert args.category == "corporate_earnings"
    assert args.limit == 1
