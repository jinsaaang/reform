"""Graph Builder pipeline for asynchronous background processing."""

from typing import Dict, Any, List, Optional

from src.config import get_config
from src.core.database import GenericDatabase
from src.domain.models import CausalHypothesis, Event, Question
from src.agents.graph_builder_agent import GraphBuilderAgentFactory
from src.pipelines.graph_builder.audit import GraphAuditPipeline
from src.pipelines.prompts import graph_builder as graph_builder_prompts
from src.config.pipeline import EvidenceSatisfactionConfig, SATISFACTION_DEFAULTS
from src.services.question_monitor_service import QuestionMonitorService
from src.utils.logging import logger


class GraphBuilderPipeline:
    """Pipeline for converting causal explanations into structured graphs."""

    def __init__(
        self,
        db_path: str,
        model_id: Optional[str] = None,
        temperature: float = 0.2,
        min_evidence_articles: int = SATISFACTION_DEFAULTS.min_articles,
        min_graph_depth: int = SATISFACTION_DEFAULTS.min_graph_depth,
        min_events: int = SATISFACTION_DEFAULTS.min_graph_events,
        agent_mode: str = "code",
        agent_max_steps: int = 20,
        agent_max_output_tokens: int = 24_000,
    ):
        """Initialize the pipeline."""
        self.db_path = db_path
        self.db = GenericDatabase(db_path)
        self.model_id = model_id or get_config().llm.model
        self.temperature = temperature
        self.min_evidence_articles = min_evidence_articles
        self.min_graph_depth = min_graph_depth
        self.min_events = min_events
        self.agent_mode = agent_mode
        self.agent_max_steps = agent_max_steps
        self.agent_max_output_tokens = agent_max_output_tokens

    def _load_pending_questions(self) -> List[Question]:
        """Find graph-eligible questions waiting for graph building."""
        all_questions = self.db.get_many(Question)
        pending = [q for q in all_questions if q.causal_explanation and not q.graph_built]
        monitor = QuestionMonitorService(
            self.db,
            EvidenceSatisfactionConfig(min_articles=self.min_evidence_articles),
        )
        eligible = [
            q for q in pending if monitor.has_sufficient_evidence_articles(q.id)
        ]

        skipped = len(pending) - len(eligible)
        if skipped > 0:
            logger.info(
                f"Skipping {skipped} pending question(s) below the evidence threshold."
            )

        return eligible

    def process_pending(self, limit: int = 10) -> Dict[str, Any]:
        """Process all pending questions."""
        pending = self._load_pending_questions()
        if not pending:
            logger.info("No questions waiting for graph building.")
            return {"processed": 0, "success": 0, "failed": 0}

        pending = pending[:limit]
        logger.info(f"Found {len(pending)} pending questions for graph building.")

        results = {"processed": 0, "success": 0, "failed": 0}

        for q in pending:
            logger.info(f"Building graph for question {q.id}...")
            success = self._process_single_question(q)
            results["processed"] += 1
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1

        return results

    def _process_single_question(self, question: Question) -> bool:
        """Process a single question's graph."""
        try:
            monitor = QuestionMonitorService(
                self.db,
                EvidenceSatisfactionConfig(min_articles=self.min_evidence_articles),
            )
            if not monitor.has_sufficient_evidence_articles(question.id):
                article_count = len(monitor.get_evidence_articles(question.id))
                reason = (
                    "Evidence article threshold not met for this question: "
                    f"{article_count} < {self.min_evidence_articles}. "
                    "Run `wr evidence run` first."
                )
                logger.warning(f"Skipping graph build for {question.id}: {reason}")
                question.graph_build_error = reason
                self.db.save(Question, question)
                return False

            # 1. Find actual outcome event
            events = self.db.get_many(Event)
            actual_outcome_id = "unknown"

            # Use explicit fallback
            outcome_events = []
            if getattr(question, "outcome_event_ids", None):
                outcome_events = [
                    e for e in events if e.id in question.outcome_event_ids
                ]
            else:
                outcome_events = [
                    e
                    for e in events
                    if getattr(e, "extracted_for_question_id", None) == question.id
                    and getattr(e, "is_outcome", False)
                ]

            actual_outcomes = [
                event
                for event in outcome_events
                if getattr(event, "is_actual_outcome", False)
            ]
            if len(actual_outcomes) == 1:
                actual_outcome_id = actual_outcomes[0].id

            if actual_outcome_id == "unknown":
                reason = (
                    "Question must have exactly one actual outcome event; "
                    f"found {len(actual_outcomes)}."
                )
                logger.error(f"Skipping graph build for {question.id}: {reason}")
                question.graph_built = False
                question.graph_build_error = reason
                self.db.save(Question, question)
                return False

            # A prior attempt may have persisted a fully valid graph but missed
            # only the agent's final completion call. Resume from the
            # deterministic graph state before spending another model request.
            initial_prompt = None
            existing_hypotheses = self.db.get_many(
                CausalHypothesis,
                filters={
                    "discovered_by_question_ids__like": f'%"{question.id}"%'
                },
            )
            if existing_hypotheses:
                preflight_monitor = QuestionMonitorService(
                    self.db,
                    EvidenceSatisfactionConfig(
                        min_graph_depth=self.min_graph_depth,
                        min_graph_events=self.min_events,
                    ),
                )
                preflight_sat = preflight_monitor.check_graph_satisfaction(
                    question.id
                )
                preflight_audit = GraphAuditPipeline(
                    self.db_path
                ).audit_question(question.id)
                if (
                    preflight_sat.is_satisfied
                    and preflight_audit.get("status") == "pass"
                ):
                    question.graph_built = True
                    question.graph_build_error = None
                    self.db.save(Question, question)
                    logger.info(
                        f"Reused deterministically valid persisted graph for "
                        f"Q: {question.id}"
                    )
                    return True
                preflight_issues = list(
                    preflight_sat.missing_requirements
                ) + list(preflight_audit.get("issues", []))
                if not question.graph_built:
                    preflight_issues.insert(
                        0,
                        "The persisted draft was not marked complete.",
                    )
                initial_prompt = graph_builder_prompts.get_repair_prompt(
                    question=question,
                    actual_outcome_event_id=actual_outcome_id,
                    issues=preflight_issues,
                    min_graph_depth=self.min_graph_depth,
                    min_events=self.min_events,
                )
                logger.info(
                    f"Resuming persisted partial graph with targeted repair for "
                    f"Q: {question.id}"
                )

            # 2. Build Prompt
            prompt = initial_prompt or graph_builder_prompts.get_prompt(
                question=question,
                actual_outcome_event_id=actual_outcome_id,
                min_graph_depth=self.min_graph_depth,
                min_events=self.min_events,
            )

            # 3. Create Agent
            agent = GraphBuilderAgentFactory.create(
                model_id=self.model_id,
                temperature=self.temperature,
                db_path=self.db_path,
                question_id=question.id,
                agent_mode=self.agent_mode,
                min_graph_depth=self.min_graph_depth,
                min_events=self.min_events,
                max_steps=self.agent_max_steps,
                max_output_tokens=self.agent_max_output_tokens,
            )

            # 4. Run Agent
            logger.info(f"Starting GraphBuilderAgent for Q: {question.id}")
            _ = agent.run(prompt)

            # Validate the initial draft independently. A bounded repair is useful
            # both when the agent explicitly marks failure and when it marks a graph
            # complete that still misses structural or audit requirements.
            refreshed_q = self.db.get(Question, question.id)
            monitor = QuestionMonitorService(
                self.db,
                EvidenceSatisfactionConfig(
                    min_graph_depth=self.min_graph_depth,
                    min_graph_events=self.min_events,
                ),
            )
            graph_sat = monitor.check_graph_satisfaction(question.id)
            audit = GraphAuditPipeline(self.db_path).audit_question(question.id)
            initial_valid = (
                graph_sat.is_satisfied and audit.get("status") == "pass"
            )
            if initial_valid and not refreshed_q.graph_built:
                # ``graph_built`` is an agent completion signal, not an
                # independent validity check. Normalize it from the persisted
                # graph's deterministic structural/evidence audit so a missed
                # final tool call cannot discard an otherwise strict graph.
                refreshed_q.graph_built = True
                refreshed_q.graph_build_error = None
                self.db.save(Question, refreshed_q)
            needs_repair = not initial_valid
            if needs_repair:
                repair_issues = []
                if not refreshed_q.graph_built:
                    repair_issues.append(
                        "The initial draft was not marked complete by the graph agent."
                    )
                repair_issues.extend(graph_sat.missing_requirements)
                repair_issues.extend(audit.get("issues", []))
                logger.warning(
                    f"[{question.id}] Graph validation failed; running one bounded "
                    f"repair: {repair_issues}"
                )
                # Keep the persisted graph/DB state, but discard the initial
                # agent's failed trace so repair does not inherit a large,
                # malformed code-generation context.
                repair_agent = GraphBuilderAgentFactory.create(
                    model_id=self.model_id,
                    temperature=self.temperature,
                    db_path=self.db_path,
                    question_id=question.id,
                    agent_mode=self.agent_mode,
                    min_graph_depth=self.min_graph_depth,
                    min_events=self.min_events,
                    max_steps=self.agent_max_steps,
                    max_output_tokens=self.agent_max_output_tokens,
                )
                _ = repair_agent.run(
                    graph_builder_prompts.get_repair_prompt(
                        question=question,
                        actual_outcome_event_id=actual_outcome_id,
                        issues=repair_issues,
                        min_graph_depth=self.min_graph_depth,
                        min_events=self.min_events,
                    )
                )
                refreshed_q = self.db.get(Question, question.id)
                graph_sat = monitor.check_graph_satisfaction(question.id)
                audit = GraphAuditPipeline(self.db_path).audit_question(question.id)
                repaired_valid = (
                    graph_sat.is_satisfied and audit.get("status") == "pass"
                )
                if not repaired_valid:
                    reason = (
                        f"Graph validation failed after one repair: "
                        f"{audit.get('issues', [])}; "
                        f"{graph_sat.missing_requirements}"
                    )
                    logger.error(f"[{question.id}] {reason}")
                    refreshed_q.graph_built = False
                    refreshed_q.graph_build_error = reason
                    self.db.save(Question, refreshed_q)
                    return False
                if not refreshed_q.graph_built:
                    refreshed_q.graph_built = True
                    refreshed_q.graph_build_error = None
                    self.db.save(Question, refreshed_q)

            refreshed_q = self.db.get(Question, question.id)
            if refreshed_q.graph_build_error is not None:
                refreshed_q.graph_build_error = None
                self.db.save(Question, refreshed_q)
            logger.info(f"Successfully built graph for Q: {question.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process graph for {question.id}: {str(e)}")
            question.graph_build_error = str(e)
            self.db.save(Question, question)
            return False
