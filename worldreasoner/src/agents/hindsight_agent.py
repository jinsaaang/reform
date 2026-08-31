from datetime import datetime
from typing import Optional

from smolagents import CodeAgent, ToolCallingAgent

from src.agents.base import BaseAgent, create_llm_model, _uses_structured_outputs
from src.config import Config, get_config
from src.config.pipeline import EvidenceSatisfactionConfig
from src.tools import (
    # Evidence
    ArticleCollectorTool,
    ArticleRetrievalTool,
    WebFetchTool,
    WebSearchTool,
    # Inspector
    ArticleInspectorTool,
    # NL Explanation
    SaveExplanationTool,
    SearchCoverageTool,
    SearchCoverageTracker,
)
from src.tools.generators.question_articles import QuestionArticlesTool
from src.pipelines.prompts.hindsight_causal_analysis import (
    EVIDENCE_AGENT_DESCRIPTION,
)


class HindsightAgent(BaseAgent):
    """Manager agent for building deep causal explanations with hindsight.

    This agent orchestrates:
    1. Evidence collection (delegated to Evidence Agent)
    2. Writing a natural-language causal explanation (save_explanation)

    The GraphBuilderAgent converts the saved explanation into a structured
    graph asynchronously.
    """

    def __init__(
        self,
        config: Config = None,
        tools: list = [],
        max_steps: int = 30,
        is_code: bool = True,
        db_path: str = "worldreasoner.db",
        question_id: Optional[str] = None,
        target_event_id: Optional[str] = None,
        satisfaction_config: Optional[EvidenceSatisfactionConfig] = None,
        evidence_agent_is_code: Optional[bool] = None,
        evidence_agent_max_steps: int = 15,
        domain: str = "general",
        model_id: Optional[str] = None,
        search_query_budget: int = 10,
        search_coverage_tracker: Optional[SearchCoverageTracker] = None,
        enable_evidence_agent: bool = True,
    ):
        """Initialize the HindsightAgent.

        Args:
            config: Configuration object
            tools: Additional tools for the manager agent
            max_steps: Maximum steps for the manager agent
            is_code: Whether to use CodeAgent
            db_path: Path to the database
            question_id: Question ID for provenance tracking (passed to all tools)
            target_event_id: Target event ID for causal graph building
            satisfaction_config: Evidence and graph satisfaction thresholds
            evidence_agent_is_code: Override the evidence sub-agent type. If None,
                it follows ``is_code``.
            domain: Domain assigned to automatically collected articles.
            model_id: Optional model override shared by the evidence sub-agent and
                hindsight manager.
            search_query_budget: Maximum distinct query/page/provider combinations.
            enable_evidence_agent: Attach the web-search sub-agent when evidence
                coverage is incomplete. Set False only for a manager-only
                explanation pass after deterministic fallback collection.
        """
        if config is None:
            config = get_config()
        if model_id:
            config = config.model_copy(
                update={"llm": config.llm.model_copy(update={"model": model_id})}
            )

        self.question_id = question_id
        self.target_event_id = target_event_id
        min_articles = satisfaction_config.min_articles if satisfaction_config else 20
        coverage_tracker = search_coverage_tracker or SearchCoverageTracker(
            db_path=db_path,
            question_id=question_id,
            min_articles=min_articles,
            max_queries=search_query_budget,
        )
        self.search_coverage_tracker = coverage_tracker
        evidence_agent_is_code = (
            is_code if evidence_agent_is_code is None else evidence_agent_is_code
        )

        llm_model = create_llm_model(config)

        if evidence_agent_is_code:
            date_instructions = (
                f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
                "All question resolution dates and evidence windows are in the past. "
                "Search the web normally — do NOT skip searches assuming events are in "
                "the future. Use the structured WebSearchTool result directly. If the "
                "publication date is absent, let article_collector verify it from the "
                "page; never substitute an event or reporting-period date."
            )
        else:
            date_instructions = (
                f"Today's date is {datetime.now().strftime('%Y-%m-%d')}. "
                "All question resolution dates and evidence windows are in the past. "
                "Use WebSearchTool directly through tool calls. Every search automatically "
                "fetches, validates, deduplicates, and stores eligible results. Check "
                "article_inspector after each search, then change the query or page when "
                "coverage is still insufficient. Do not parse URLs or publication dates "
                "yourself."
            )

        # Evidence gathering specialist (web search, article collection)
        # Tools get question_id for provenance tracking
        if evidence_agent_is_code:
            evidence_tools = [
                ArticleCollectorTool(db_path=db_path, question_id=question_id),
                ArticleInspectorTool(
                    db_path=db_path,
                    question_id=question_id,
                    satisfaction_config=satisfaction_config,
                    require_causal_explanation=False,
                ),
                WebFetchTool(),
                SearchCoverageTool(coverage_tracker),
                WebSearchTool(
                    db_path=db_path,
                    question_id=question_id,
                    coverage_tracker=coverage_tracker,
                    enforce_upper_only_dates=True,
                ),
            ]
        else:
            # The model chooses only query/page. Search, parallel fetch, date
            # validation, deduplication, and persistence stay deterministic.
            evidence_tools = [
                ArticleInspectorTool(
                    db_path=db_path,
                    question_id=question_id,
                    satisfaction_config=satisfaction_config,
                    require_causal_explanation=False,
                ),
                SearchCoverageTool(coverage_tracker),
                WebSearchTool(
                    db_path=db_path,
                    question_id=question_id,
                    auto_collect_enabled=True,
                    max_auto_collect=10,
                    domain=domain,
                    coverage_tracker=coverage_tracker,
                    enforce_upper_only_dates=True,
                ),
            ]

        evidence_agent_kwargs = dict(
            model=llm_model,
            tools=evidence_tools,
            max_steps=evidence_agent_max_steps,
            stream_outputs=False,
            name="evidence_collector",
            description=EVIDENCE_AGENT_DESCRIPTION.format(
                min_evidence_articles=(
                    min_articles
                ),
                search_query_budget=search_query_budget,
            ),
            instructions=date_instructions,
        )
        if evidence_agent_is_code:
            evidence_agent_kwargs["additional_authorized_imports"] = [
                "json",
                "datetime",
                "typing",
            ]
            evidence_agent_kwargs["executor_kwargs"] = {"timeout_seconds": None}
            if _uses_structured_outputs(llm_model.model_id):
                evidence_agent_kwargs["use_structured_outputs_internally"] = True
            evidence_agent = CodeAgent(**evidence_agent_kwargs)
        else:
            evidence_agent_kwargs["max_tool_threads"] = 1
            evidence_agent = ToolCallingAgent(**evidence_agent_kwargs)

        manager_tools = [
            ArticleInspectorTool(
                db_path=db_path,
                question_id=question_id,
                satisfaction_config=satisfaction_config,
                require_causal_explanation=False,
            ),  # Check coverage
            ArticleRetrievalTool(db_path=db_path),  # Read full article content
            SaveExplanationTool(db_path=db_path, question_id=question_id),
            QuestionArticlesTool(db_path=db_path, question_id=question_id),
            SearchCoverageTool(coverage_tracker),
        ]
        if is_code:
            manager_tools.insert(
                0, ArticleCollectorTool(db_path=db_path, question_id=question_id)
            )

        agent_kwargs = {}
        if is_code:
            agent_kwargs["executor_kwargs"] = {"timeout_seconds": None}
        else:
            agent_kwargs["max_tool_threads"] = 1

        from src.core.database import GenericDatabase
        from src.services.question_monitor_service import QuestionMonitorService

        monitor = QuestionMonitorService(GenericDatabase(db_path), satisfaction_config)
        has_sufficient_article_coverage = monitor.has_sufficient_evidence_articles(
            question_id
        )

        super().__init__(
            config=config,
            tools=tools + manager_tools,
            max_steps=max_steps,
            is_code=is_code,
            managed_agents=(
                []
                if has_sufficient_article_coverage or not enable_evidence_agent
                else [evidence_agent]
            ),
            instructions=date_instructions,
            **agent_kwargs,
        )
