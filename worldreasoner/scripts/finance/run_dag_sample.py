#!/usr/bin/env python3
"""Run the WorldReasoner hindsight-to-DAG pipeline on finance samples.

The run is intentionally resumable: it keeps a SQLite database and rewrites a
small JSON summary after every question. Source prompts and graph rules are not
modified by this adapter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import traceback
import math
import hashlib
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.analysis.graph_analysis import calculate_graph_quality, resolve_target_event_id
from src.config.database import DatabaseConfig
from src.config.pipeline import EvidencePipelineConfig, EvidenceSatisfactionConfig
from src.core.database import GenericDatabase
from src.core.db_init import init_and_migrate
from src.domain.models import (
    Article,
    CausalHypothesis,
    Event,
    EventOutcomeImpact,
    Question,
    QuestionType,
)
from src.pipelines.evidence.pipeline import EvidencePipeline
from src.pipelines.graph_builder.audit import GraphAuditPipeline
from src.pipelines.graph_builder.pipeline import GraphBuilderPipeline
from src.services.question_monitor_service import QuestionMonitorService
from src.tools.collectors.article_collector import ArticleCollectorTool
from src.tools import SearchCoverageTracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT.parent
DEFAULT_SAMPLE = (
    RESEARCH_ROOT / "data" / "worldreasoner" / "finance_questions_sample_10.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "forecaster"
    / "experiments"
    / "worldreasoner_dag_gemini_2_5_flash_10"
)
FINANCE_CATEGORIES = (
    "corporate_earnings",
    "energy_commodities",
    "macro",
    "market_fx_credit",
    "monetary_policy",
)


def read_questions(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as handle:
        questions = [
            Question.model_validate_json(line) for line in handle if line.strip()
        ]

    seen_ids: set[str] = set()
    for question in questions:
        if question.id in seen_ids:
            raise ValueError(f"Duplicate question ID: {question.id}")
        seen_ids.add(question.id)
        validate_resolved_question(question)
    return questions


def validate_resolved_question(question: Question) -> None:
    """Reject inputs that cannot produce an anchored hindsight DAG."""
    if question.ground_truth is None:
        raise ValueError(f"Question {question.id} has no resolved ground truth")

    if question.question_type == QuestionType.TIMEFRAME:
        raise ValueError(
            f"Question {question.id} uses unsupported timeframe outcomes"
        )

    if question.question_type == QuestionType.BINARY:
        normalized = str(question.ground_truth).strip().strip("\"'").lower()
        numeric_truth = (
            isinstance(question.ground_truth, (int, float))
            and not isinstance(question.ground_truth, bool)
            and question.ground_truth in {0, 1}
        )
        if (
            not isinstance(question.ground_truth, bool)
            and not numeric_truth
            and normalized
            not in {"yes", "no", "true", "false", "y", "n", "1", "0"}
        ):
            raise ValueError(
                f"Question {question.id} has invalid binary ground truth: "
                f"{question.ground_truth!r}"
            )

    if question.question_type == QuestionType.MCQ:
        options = question.options or []
        if len(options) < 2:
            raise ValueError(f"Question {question.id} has fewer than two MCQ options")
        if isinstance(question.ground_truth, int):
            aligned = 0 <= question.ground_truth < len(options)
        else:
            truth = str(question.ground_truth).strip().casefold()
            aligned = any(str(option).strip().casefold() == truth for option in options)
        if not aligned:
            raise ValueError(
                f"Question {question.id} ground truth does not match its MCQ options"
            )


def get_finance_category(question: Question) -> Optional[str]:
    """Return the active or legacy finance category for a question."""
    metadata = question.metadata or {}
    source_metadata = metadata.get("finance") or metadata.get("finfactorbench") or {}
    category = source_metadata.get("category") or source_metadata.get("original_domain")
    return category if isinstance(category, str) else None


def select_questions(
    questions: list[Question],
    category: Optional[str],
    limit: int,
    question_ids: Optional[list[str]] = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[Question]:
    """Select questions deterministically, applying the limit after filtering."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, received {limit}")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), received {shard_index}"
        )

    eligible = questions
    if question_ids:
        requested = set(question_ids)
        eligible = [question for question in eligible if question.id in requested]
        missing = requested - {question.id for question in eligible}
        if missing:
            raise ValueError("Unknown question IDs: " + ", ".join(sorted(missing)))
    if category is not None:
        unfiltered = eligible
        eligible = [
            question
            for question in eligible
            if get_finance_category(question) == category
        ]
        if not eligible:
            available = Counter(
                item
                for question in unfiltered
                if (item := get_finance_category(question)) is not None
            )
            raise ValueError(
                f"No questions found for category {category!r}; "
                f"available category counts: {dict(sorted(available.items()))}"
            )

    shard_size = math.ceil(len(eligible) / shard_count)
    shard_start = shard_index * shard_size
    shard_end = min(shard_start + shard_size, len(eligible))
    return eligible[shard_start:shard_end][:limit]


def _resolution_family_key(question: Question) -> Optional[str]:
    root_metadata = question.metadata or {}
    metadata = root_metadata.get("finance") or root_metadata.get("finfactorbench") or {}
    if not isinstance(metadata, dict):
        return None
    document_id = metadata.get("resolution_doc_id")
    document = metadata.get("resolution_document", {})
    document_url = document.get("url") if isinstance(document, dict) else None
    key = document_id or document_url
    if key:
        return str(key)

    # Yield-curve, real-rate, breakeven, money-supply, and Fed-balance-sheet
    # questions for the same US monetary-policy month are driven by a common
    # macro information set. Reuse those already-collected articles instead of
    # repeatedly hitting rate-limited public search backends. The association
    # path still re-checks every article against the consumer question's cutoff.
    category = str(
        metadata.get("category") or metadata.get("original_domain") or ""
    ).strip().lower()
    region = str(metadata.get("region") or "").strip().upper()
    target_period = str(metadata.get("target_period") or "").strip()
    if category == "monetary_policy" and region and target_period:
        return f"monetary_policy:{region}:{target_period}"

    event_cluster_id = metadata.get("event_cluster_id")
    return str(event_cluster_id) if event_cluster_id else None


def associate_family_articles(db: GenericDatabase, question: Question) -> int:
    """Authorize already-collected evidence for questions sharing one source report."""
    family_key = _resolution_family_key(question)
    if not family_key:
        return 0
    family_question_ids = {
        candidate.id
        for candidate in db.get_many(Question)
        if _resolution_family_key(candidate) == family_key
    }
    if not family_question_ids:
        return 0

    associated = 0
    for article in db.get_many(Article):
        related_ids = set(article.metadata.get("related_question_ids", []))
        owners = related_ids | {article.collected_for_question_id}
        if not (owners & family_question_ids):
            continue
        if article.published_date >= question.resolution_date:
            continue
        if question.id in related_ids:
            associated += 1
            continue
        related_ids.add(question.id)
        metadata = dict(article.metadata or {})
        metadata["related_question_ids"] = sorted(related_ids)
        article.metadata = metadata
        db.save(Article, article)
        associated += 1
    return associated


def collect_resolution_document_fallback(
    db_path: Path, question: Question
) -> Optional[dict[str, Any]]:
    """Collect the dated official resolution release only after search found zero."""
    root_metadata = question.metadata or {}
    metadata = root_metadata.get("finance") or root_metadata.get("finfactorbench") or {}
    private = root_metadata.get("benchmark_private") or {}
    if not isinstance(metadata, dict):
        return None
    document = metadata.get("resolution_document") or private.get("resolution_document") or {}
    if not isinstance(document, dict):
        return None
    url = document.get("url")
    published_at = document.get("published_at") or metadata.get("resolution_available_at")
    if not url or not published_at:
        return None
    try:
        published_date = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        if published_date.tzinfo is None:
            published_date = published_date.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    # Historical finance evidence has no lower time bound.  A prior filing or
    # macro release can remain predictive long after the question was opened;
    # only the target question's strict upper cutoff matters here.
    if published_date >= question.resolution_date:
        return None

    collector = ArticleCollectorTool(
        db_path=str(db_path), question_id=question.id
    )
    result = collector.forward(
        url=str(url),
        title=str(document.get("title") or question.question_text),
        source=str(document.get("publisher") or "Official source"),
        published_date=published_date.isoformat(),
        domain="finance",
    )
    return result.model_dump(mode="json")


def create_structured_resolution_evidence(
    db: GenericDatabase, question: Question
) -> Optional[dict[str, Any]]:
    """Persist auditable official-series metadata when its landing page is undated."""
    root_metadata = question.metadata or {}
    metadata = root_metadata.get("finance") or root_metadata.get("finfactorbench") or {}
    private = root_metadata.get("benchmark_private") or {}
    is_official_active_source = str(metadata.get("source_type") or "").startswith(
        ("fred_", "sec_")
    )
    if not isinstance(metadata, dict) or not (
        metadata.get("source_quality") == "high" or is_official_active_source
    ):
        return None
    document = metadata.get("resolution_document") or private.get("resolution_document") or {}
    if not isinstance(document, dict):
        return None
    published_at = document.get("published_at") or metadata.get("resolution_available_at")
    resolution_value = metadata.get("resolution_value", private.get("resolution_value"))
    if not published_at or resolution_value is None or not document.get("url"):
        return None
    try:
        published_date = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        if published_date.tzinfo is None:
            published_date = published_date.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if published_date >= question.resolution_date:
        return None

    article_id = (
        "art_finance_"
        + published_date.strftime("%Y%m%d")
        + "_official_"
        + hashlib.sha1(question.id.encode()).hexdigest()[:10]
    )
    existing = db.get(Article, article_id)
    if existing:
        return {"status": "structured_existing", "id": article_id}
    signal_names = [
        str(item.get("signal"))
        for item in metadata.get("candidate_outcome_relevant_signals", [])
        if isinstance(item, dict) and item.get("signal")
    ]
    content = (
        f"Official structured resolution record for {document.get('title')}. "
        f"Publisher: {document.get('publisher')}. Published: {published_at}. "
        f"Recorded value: {resolution_value}. Resolution rule: "
        f"{question.resolution_criteria or 'See the official source.'} "
        f"Resolution audit: {question.resolution_reasoning or 'No additional note.'} "
        f"Benchmark-listed pre-resolution factor families: "
        f"{'; '.join(signal_names) or 'none listed'}. "
        "This record is a deterministic rendering of the benchmark's cited official "
        "source metadata, not model-generated evidence."
    )
    document_title = str(
        document.get("title") or f"Official resolution for {question.id}"
    )
    if len(document_title) < 10:
        document_title = f"{document_title} official series"
    article = Article(
        id=article_id,
        title=document_title,
        content=content,
        url=str(document["url"]),
        source=str(document.get("publisher") or "Official source"),
        published_date=published_date,
        domain="finance",
        collected_for_question_id=question.id,
        word_count=len(content.split()),
        metadata={
            "evidence_type": "hindsight",
            "structured_resolution_evidence": True,
            "related_question_ids": [question.id],
        },
    )
    db.save(Article, article)
    return {"status": "structured_created", "id": article_id}


def should_run_near_threshold_rescue(
    article_count: int, coverage: dict[str, Any]
) -> bool:
    """Allow exactly one bounded extension for an 8--9 article near miss."""
    return article_count in {8, 9} and bool(coverage.get("gaps"))


def should_collect_resolution_fallback(evidence_satisfied: bool) -> bool:
    """Use one official fallback only while the strict evidence gate is unmet."""
    return not evidence_satisfied


def create_round_search_tracker(
    db_path: Path,
    question_id: str,
    min_articles: int,
    query_budget: int,
) -> SearchCoverageTracker:
    """Create an independent base budget for one normal evidence round."""
    return SearchCoverageTracker(
        db_path=str(db_path),
        question_id=question_id,
        min_articles=min_articles,
        max_queries=query_budget,
    )


def _question_hypotheses(db: GenericDatabase, question_id: str) -> list[CausalHypothesis]:
    return db.get_many(
        CausalHypothesis,
        filters={"discovered_by_question_ids__like": f'%"{question_id}"%'},
    )


def export_question_graph(
    db: GenericDatabase,
    question_id: str,
    satisfaction_config: EvidenceSatisfactionConfig,
) -> dict[str, Any]:
    question = db.get(Question, question_id)
    monitor = QuestionMonitorService(db, satisfaction_config)
    articles = monitor.get_evidence_articles(question_id)
    hypotheses = _question_hypotheses(db, question_id)
    impacts = db.get_many(EventOutcomeImpact, filters={"question_id": question_id})

    event_ids = set(question.outcome_event_ids)
    for hypothesis in hypotheses:
        event_ids.add(hypothesis.source_event_id)
        event_ids.add(hypothesis.target_event_id)
    events = [event for event_id in sorted(event_ids) if (event := db.get(Event, event_id))]

    target_event_id = resolve_target_event_id(question, db, hypotheses)
    metrics = calculate_graph_quality(hypotheses, target_event_id)
    evidence_status = monitor.check_satisfaction(question_id)
    graph_status = monitor.check_graph_satisfaction(question_id)
    graph_validation = GraphAuditPipeline(str(db.db_path)).audit_question(question_id)

    return {
        "question": question.model_dump(mode="json"),
        "actual_outcome_event_id": target_event_id,
        "evidence": {
            "satisfied": evidence_status.is_satisfied,
            "article_count": evidence_status.article_count,
            "missing_requirements": evidence_status.missing_requirements,
            "articles": [
                {
                    "id": article.id,
                    "title": article.title,
                    "url": article.url,
                    "source": article.source,
                    "published_date": article.published_date.isoformat(),
                }
                for article in articles
            ],
        },
        "graph": {
            "built": question.graph_built,
            "satisfied": graph_status.is_satisfied,
            "validation": graph_validation,
            "missing_requirements": graph_status.missing_requirements,
            "metrics": metrics,
            "nodes": [event.model_dump(mode="json") for event in events],
            "edges": [hypothesis.model_dump(mode="json") for hypothesis in hypotheses],
            "outcome_impacts": [impact.model_dump(mode="json") for impact in impacts],
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compact_pipeline_results(results: list[Any]) -> list[dict[str, Any]]:
    """Serialize stage progress without copying full article bodies."""
    compact = []
    for result in results:
        data = result.model_dump(mode="json", exclude={"outputs"})
        data["output_ids"] = [
            item.id for item in result.outputs if getattr(item, "id", None)
        ]
        compact.append(data)
    return compact


async def run(args: argparse.Namespace) -> dict[str, Any]:
    input_questions = read_questions(args.sample)
    input_category_counts = Counter(
        category
        for question in input_questions
        if (category := get_finance_category(question)) is not None
    )
    questions = select_questions(
        input_questions,
        args.category,
        args.limit,
        question_ids=args.question_id,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    selected_question_ids = [question.id for question in questions]
    print(
        "Selected "
        f"{len(questions)} question(s) "
        f"for category={args.category or 'all'}: "
        f"{', '.join(selected_question_ids)}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "worldreasoner_finance_10.sqlite"
    init_and_migrate(str(db_path))
    db = GenericDatabase(str(db_path))

    for question in questions:
        if db.get(Question, question.id) is None:
            db.save(Question, question)

    summary_path = args.output_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["question_count"] = len(questions)
        summary.pop("completed_at", None)
        summary.pop("counts", None)
    else:
        summary = {
            "model": args.model,
            "sample": str(args.sample),
            "database": str(db_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "question_count": len(questions),
            "settings": {
                "agent_max_steps": args.agent_max_steps,
                "graph_agent_max_steps": args.graph_agent_max_steps,
                "graph_agent_max_output_tokens": args.graph_agent_max_output_tokens,
                "min_articles": args.min_evidence_articles,
                "min_graph_depth": args.min_graph_depth,
                "min_graph_events": args.min_graph_events,
                "max_evidence_rounds": args.max_evidence_rounds,
                "agent_mode": args.agent_mode,
                "search_provider": args.search_provider,
                "search_query_mode": args.search_query_mode,
                "browser_concurrency": args.browser_concurrency,
                "search_query_budget": args.search_query_budget,
            },
            "questions": {},
        }
    summary["settings"] = {
        "agent_max_steps": args.agent_max_steps,
        "graph_agent_max_steps": args.graph_agent_max_steps,
        "graph_agent_max_output_tokens": args.graph_agent_max_output_tokens,
        "min_articles": args.min_evidence_articles,
        "min_graph_depth": args.min_graph_depth,
        "min_graph_events": args.min_graph_events,
        "max_evidence_rounds": args.max_evidence_rounds,
        "agent_mode": args.agent_mode,
        "search_provider": args.search_provider,
        "search_query_mode": args.search_query_mode,
        "browser_concurrency": args.browser_concurrency,
        "search_query_budget": args.search_query_budget,
        "category": args.category,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    summary["selection"] = {
        "category": args.category,
        "input_question_count": len(input_questions),
        "input_category_counts": dict(sorted(input_category_counts.items())),
        "selected_question_ids": selected_question_ids,
        "explicit_question_ids": args.question_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    write_json(summary_path, summary)

    evidence_config = EvidencePipelineConfig(
        satisfaction=EvidenceSatisfactionConfig(
            min_articles=args.min_evidence_articles,
            min_graph_depth=args.min_graph_depth,
            min_graph_events=args.min_graph_events,
        )
    )
    database_config = DatabaseConfig(db_path=str(db_path), timeout=30.0)

    for index, original_question in enumerate(questions, 1):
        question_id = original_question.id
        existing_question = db.get(Question, question_id)
        question_summary: dict[str, Any] = {
            "index": index,
            "question_id": question_id,
            "category": get_finance_category(original_question),
            "status": "running",
            "stage": "question_loaded",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        summary["questions"][question_id] = question_summary
        write_json(summary_path, summary)

        try:
            question = db.get(Question, question_id)
            question_summary["family_articles_reused"] = associate_family_articles(
                db, question
            )
            monitor = QuestionMonitorService(db, evidence_config.satisfaction)
            last_round_search_tracker: SearchCoverageTracker | None = None
            evidence_satisfaction = monitor.check_satisfaction(question_id)
            question_summary["evidence_rounds"] = []
            refreshed = db.get(Question, question_id)
            graph_evidence_available = evidence_satisfaction.is_satisfied
            question_summary["evidence_coverage_target_met"] = (
                evidence_satisfaction.is_satisfied
            )
            question_summary["graph_evidence_available"] = graph_evidence_available

            if graph_evidence_available:
                question_summary["evidence_pipeline"] = []
                question_summary["evidence_pipeline_skipped"] = (
                    "Existing evidence already satisfies this run's threshold."
                )
            else:
                evidence_round = 0
                while not evidence_satisfaction.is_satisfied and (
                    args.max_evidence_rounds == 0
                    or evidence_round < args.max_evidence_rounds
                ):
                    evidence_round += 1
                    count_before = evidence_satisfaction.article_count
                    round_started_at = datetime.now(timezone.utc).isoformat()
                    question_summary["stage"] = (
                        f"evidence_collecting_round_{evidence_round}"
                    )
                    write_json(summary_path, summary)
                    round_search_tracker = create_round_search_tracker(
                        db_path=db_path,
                        question_id=question_id,
                        min_articles=args.min_evidence_articles,
                        query_budget=args.search_query_budget,
                    )
                    last_round_search_tracker = round_search_tracker
                    evidence_pipeline = EvidencePipeline(
                        evidence_config=evidence_config,
                        database_config=database_config,
                        enable_persistence=True,
                        max_concurrent_questions=1,
                        agent_max_steps=args.agent_max_steps,
                        min_graph_depth=args.min_graph_depth,
                        use_code_agents=args.agent_mode != "tool",
                        evidence_agent_is_code=args.agent_mode == "code",
                        evidence_agent_max_steps=args.evidence_agent_max_steps,
                        model_id=args.model,
                        search_query_budget=args.search_query_budget,
                        search_coverage_tracker=round_search_tracker,
                    )
                    evidence_results = await evidence_pipeline.run([question])
                    compact_results = compact_pipeline_results(evidence_results)
                    evidence_satisfaction = monitor.check_satisfaction(question_id)
                    round_summary = {
                        "round": evidence_round,
                        "started_at": round_started_at,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "article_count_before": count_before,
                        "article_count_after": evidence_satisfaction.article_count,
                        "satisfied": evidence_satisfaction.is_satisfied,
                        "missing_requirements": (
                            evidence_satisfaction.missing_requirements
                        ),
                        "pipeline_results": compact_results,
                        "search_coverage": (
                            evidence_pipeline.search_coverage_by_question.get(
                                question_id, {}
                            )
                        ),
                    }
                    question_summary["evidence_rounds"].append(round_summary)
                    question_summary["evidence_pipeline"] = compact_results
                    question_summary["evidence_satisfaction"] = asdict(
                        evidence_satisfaction
                    )
                    write_json(summary_path, summary)

                rescue_search_tracker = (
                    last_round_search_tracker
                    or create_round_search_tracker(
                        db_path=db_path,
                        question_id=question_id,
                        min_articles=args.min_evidence_articles,
                        query_budget=args.search_query_budget,
                    )
                )
                rescue_coverage = rescue_search_tracker.snapshot()
                if (
                    not evidence_satisfaction.is_satisfied
                    and should_run_near_threshold_rescue(
                        evidence_satisfaction.article_count, rescue_coverage
                    )
                ):
                    rescue_search_tracker.extend_query_budget(3)
                    question_summary["stage"] = "evidence_collecting_near_threshold_rescue"
                    write_json(summary_path, summary)
                    rescue_pipeline = EvidencePipeline(
                        evidence_config=evidence_config,
                        database_config=database_config,
                        enable_persistence=True,
                        max_concurrent_questions=1,
                        agent_max_steps=args.agent_max_steps,
                        min_graph_depth=args.min_graph_depth,
                        use_code_agents=args.agent_mode != "tool",
                        evidence_agent_is_code=args.agent_mode == "code",
                        evidence_agent_max_steps=args.evidence_agent_max_steps,
                        model_id=args.model,
                        search_query_budget=args.search_query_budget,
                        search_coverage_tracker=rescue_search_tracker,
                    )
                    rescue_results = await rescue_pipeline.run([question])
                    evidence_satisfaction = monitor.check_satisfaction(question_id)
                    question_summary["evidence_rounds"].append(
                        {
                            "round": "near_threshold_rescue",
                            "article_count_after": evidence_satisfaction.article_count,
                            "satisfied": evidence_satisfaction.is_satisfied,
                            "missing_requirements": evidence_satisfaction.missing_requirements,
                            "pipeline_results": compact_pipeline_results(rescue_results),
                            "search_coverage": rescue_pipeline.search_coverage_by_question.get(
                                question_id, {}
                            ),
                        }
                    )
                    question_summary["evidence_satisfaction"] = asdict(
                        evidence_satisfaction
                    )
                    write_json(summary_path, summary)

                if should_collect_resolution_fallback(evidence_satisfaction.is_satisfied):
                    fallback = collect_resolution_document_fallback(
                        db_path, db.get(Question, question_id)
                    )
                    if not monitor.has_evidence_articles(question_id):
                        structured_fallback = create_structured_resolution_evidence(
                            db, db.get(Question, question_id)
                        )
                        if structured_fallback:
                            fallback = structured_fallback
                    question_summary["resolution_document_fallback"] = fallback
                    if fallback and fallback.get("status") in {
                        "created",
                        "existing",
                        "structured_created",
                        "structured_existing",
                    }:
                        # Re-enter only the manager stage. The evidence agent is
                        # explicitly disabled so an underfilled strict gate cannot
                        # silently open a fresh search budget after fallback.
                        explanation_pipeline = EvidencePipeline(
                            evidence_config=evidence_config,
                            database_config=database_config,
                            enable_persistence=True,
                            max_concurrent_questions=1,
                            agent_max_steps=args.agent_max_steps,
                            min_graph_depth=args.min_graph_depth,
                            use_code_agents=args.agent_mode != "tool",
                            evidence_agent_is_code=args.agent_mode == "code",
                            evidence_agent_max_steps=args.evidence_agent_max_steps,
                            model_id=args.model,
                            search_query_budget=args.search_query_budget,
                            enable_evidence_agent=False,
                        )
                        explanation_results = await explanation_pipeline.run(
                            [db.get(Question, question_id)]
                        )
                        question_summary["resolution_fallback_pipeline"] = (
                            compact_pipeline_results(explanation_results)
                        )

            refreshed = db.get(Question, question_id)
            evidence_satisfaction = monitor.check_satisfaction(question_id)
            graph_evidence_available = evidence_satisfaction.is_satisfied
            question_summary["evidence_coverage_target_met"] = (
                evidence_satisfaction.is_satisfied
            )
            question_summary["graph_evidence_available"] = graph_evidence_available
            question_summary["evidence_satisfaction"] = asdict(
                evidence_satisfaction
            )
            if evidence_satisfaction.is_satisfied:
                question_summary["stage"] = "evidence_complete"
                write_json(summary_path, summary)

            if graph_evidence_available:
                if refreshed.graph_built:
                    existing_graph_status = monitor.check_graph_satisfaction(
                        question_id
                    )
                    existing_graph_validation = GraphAuditPipeline(
                        str(db_path)
                    ).audit_question(question_id)
                    graph_success = (
                        existing_graph_status.is_satisfied
                        and existing_graph_validation.get("status") == "pass"
                    )
                    if graph_success:
                        if refreshed.graph_build_error is not None:
                            refreshed.graph_build_error = None
                            db.save(Question, refreshed)
                        question_summary["graph_pipeline_skipped"] = (
                            "Existing graph satisfies the current validation gates."
                        )
                    else:
                        refreshed.graph_built = False
                        refreshed.graph_build_error = (
                            "Existing graph failed current validation: "
                            f"{existing_graph_validation.get('issues', [])}; "
                            f"{existing_graph_status.missing_requirements}"
                        )
                        db.save(Question, refreshed)
                        question_summary["graph_pipeline_skipped"] = (
                            refreshed.graph_build_error
                        )
                else:
                    question_summary["stage"] = "graph_building"
                    write_json(summary_path, summary)
                    graph_pipeline = GraphBuilderPipeline(
                        db_path=str(db_path),
                        model_id=args.model,
                        temperature=0.2,
                        min_evidence_articles=args.min_evidence_articles,
                        min_graph_depth=args.min_graph_depth,
                        min_events=args.min_graph_events,
                        agent_mode=(
                            "tool" if args.agent_mode == "tool" else "code"
                        ),
                        agent_max_steps=args.graph_agent_max_steps,
                        agent_max_output_tokens=args.graph_agent_max_output_tokens,
                    )
                    graph_success = graph_pipeline._process_single_question(refreshed)
            else:
                graph_success = False
                question_summary["graph_pipeline_skipped"] = (
                    "The configured evidence threshold or causal explanation requirement "
                    "was not satisfied after the evidence-search rounds."
                )
            question_summary["graph_pipeline_success"] = graph_success
            question_summary["stage"] = (
                "graph_complete" if graph_success else "graph_failed"
            )
            write_json(summary_path, summary)

            graph_export = export_question_graph(
                db,
                question_id,
                evidence_config.satisfaction,
            )
            write_json(args.output_dir / "graphs" / f"{question_id}.json", graph_export)
            question_summary["status"] = (
                "completed"
                if (
                    graph_export["evidence"]["satisfied"]
                    and graph_export["graph"]["built"]
                    and graph_export["graph"]["satisfied"]
                    and graph_export["graph"]["validation"].get("status") == "pass"
                )
                else "failed"
            )
            question_summary["article_count"] = graph_export["evidence"][
                "article_count"
            ]
            question_summary["graph_metrics"] = graph_export["graph"]["metrics"]
            question_summary["graph_error"] = graph_export["question"].get(
                "graph_build_error"
            )
            question_summary["stage"] = "export_complete"
        except Exception as exc:
            question_summary["status"] = "exception"
            question_summary["error"] = str(exc)
            question_summary["traceback"] = traceback.format_exc()

        question_summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, summary)

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    summary["counts"] = {
        status: sum(
            item["status"] == status for item in summary["questions"].values()
        )
        for status in ("completed", "failed", "exception")
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="google/gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--question-id",
        action="append",
        default=None,
        help="Run a specific question ID; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--category",
        choices=FINANCE_CATEGORIES,
        default=None,
        help=(
            "Run only one FinFactorBench original_domain category. The limit is "
            "applied after this filter, and input order is preserved."
        ),
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=20,
        help=(
            "Maximum hindsight-manager steps. Evidence search has its own "
            "separate step and query budgets."
        ),
    )
    parser.add_argument(
        "--graph-agent-max-steps",
        type=int,
        default=12,
        help=(
            "Maximum steps for each graph-build attempt. A failed draft can still "
            "receive the pipeline's separate bounded repair attempt. Keep this "
            "bounded so a repeated invalid event/edge call starts a fresh repair "
            "context instead of growing the same agent trace indefinitely."
        ),
    )
    parser.add_argument(
        "--graph-agent-max-output-tokens",
        type=int,
        default=24_000,
        help=(
            "Maximum output tokens for one graph-builder model step. This "
            "bounds pathological repeated JSON/code generation without "
            "changing evidence, node, depth, or audit requirements."
        ),
    )
    parser.add_argument(
        "--evidence-agent-max-steps",
        type=int,
        default=15,
        help="Maximum search/inspection steps inside the evidence sub-agent.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=("hybrid", "tool", "code"),
        default="hybrid",
        help=(
            "Finance runner agent mode. 'hybrid' keeps the stable baseline "
            "manager/graph agents but makes evidence search code-free; 'tool' "
            "uses native tool calls everywhere; 'code' reproduces the original "
            "baseline."
        ),
    )
    parser.add_argument("--min-evidence-articles", type=int, default=10)
    parser.add_argument(
        "--search-query-budget",
        type=int,
        default=10,
        help="Maximum distinct query/page/provider combinations per evidence agent run.",
    )
    parser.add_argument(
        "--search-provider",
        choices=("auto", "google_news", "gdelt", "ddgs", "smolagents"),
        default="auto",
        help=(
            "Search transport: auto uses Google News RSS for dated finance "
            "news, then DDGS and GDELT as fallbacks; the other choices force "
            "one backend."
        ),
    )
    parser.add_argument(
        "--search-query-mode",
        choices=("original", "finance_variants"),
        default="finance_variants",
        help=(
            "Transport-side query policy. finance_variants deterministically "
            "adds period, country, CPI, and causal-factor searches."
        ),
    )
    parser.add_argument(
        "--browser-concurrency",
        type=int,
        default=2,
        help="Maximum concurrent browser fetches per process (recommended: 1-3).",
    )
    parser.add_argument("--min-graph-depth", type=int, default=3)
    parser.add_argument("--min-graph-events", type=int, default=8)
    parser.add_argument(
        "--max-evidence-rounds",
        type=int,
        default=3,
        help="Maximum DB-verified evidence collection rounds; 0 means unlimited",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["WEB_SEARCH_FALLBACK_PROVIDER"] = args.search_provider
    os.environ["WEB_SEARCH_QUERY_MODE"] = args.search_query_mode
    os.environ["WEB_SEARCH_MAX_FETCH_WORKERS"] = str(args.browser_concurrency)
    summary = asyncio.run(run(args))
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
