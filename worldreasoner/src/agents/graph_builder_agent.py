"""Graph builder agent for translating NL causal explanations into structured graphs."""

from typing import Union

from smolagents import ActionStep, CodeAgent, Tool, ToolCallingAgent

from src.core.database import GenericDatabase
from src.core.alias_registry import AliasRegistry
from src.domain.models import Question
from src.config.pipeline import EvidenceSatisfactionConfig

from src.tools import (
    ArticleRetrievalTool,
    EventIdentifierTool,
    CausalReasonerTool,
    GraphInspectorTool,
    RecordOutcomeImpactTool,
    DeleteEventTool,
    DeleteHypothesisTool,
)
from src.tools.generators.question_articles import QuestionArticlesTool
from src.tools.reasoning.propose_subgraph import ProposeSubgraphTool

from .factory import AgentFactory
from .base import _uses_structured_outputs


def _compact_graph_builder_history(_memory_step, *, agent) -> None:
    """Bound repeated graph-repair traces without discarding useful state.

    CodeAgent's Python executor and every graph mutation persist independently
    of the chat transcript. Keeping the first two actions preserves the article
    inventory/aliases, while the last two preserve the current repair context.
    Large intermediate code payloads in between otherwise get resent on every
    model call and can grow a single graph build to hundreds of thousands of
    input tokens.
    """

    action_steps = [
        step for step in agent.memory.steps if isinstance(step, ActionStep)
    ]
    if len(action_steps) <= 4:
        return

    retained = {
        id(step) for step in action_steps[:2] + action_steps[-2:]
    }
    for step in action_steps:
        if id(step) in retained or getattr(step, "_graph_trace_compacted", False):
            continue
        step.model_input_messages = None
        step.model_output_message = None
        step.model_output = (
            "Earlier intermediate graph-building action compacted. "
            "The Python executor and persisted database state remain available; "
            "use graph_inspector before any repair."
        )
        step.code_action = None
        step.tool_calls = None
        step.observations = None
        step.observations_images = None
        step.action_output = None
        step.error = None
        step._graph_trace_compacted = True


class MarkGraphBuiltTool(Tool):
    """Internal tool for the GraphBuilderAgent to mark its job as done."""

    name = "mark_graph_built"
    description = "Mark the graph as fully built and verified."
    inputs = {
        "success": {
            "type": "boolean",
            "description": "True if built successfully, False if unrecoverable errors",
        }
    }
    output_type = "string"

    def __init__(self, db_path: str, question_id: str):
        super().__init__()
        self.db = GenericDatabase(db_path)
        self.question_id = question_id

    def forward(self, success: bool) -> str:
        q = self.db.get(Question, self.question_id)
        if q:
            q.graph_built = success
            if success:
                q.graph_build_error = None
            else:
                q.graph_build_error = "Agent marked graph build as failed."
            self.db.save(Question, q)
            return "Graph marked successfully."
        return "Question not found."


class GraphBuilderAgentFactory(AgentFactory):
    """Factory for the GraphBuilder Agent."""

    @classmethod
    def create(
        cls,
        model_id: str,
        temperature: float = 0.2,
        db_path: str = None,
        question_id: str = None,
        agent_mode: str = "code",
        **kwargs,
    ) -> Union[CodeAgent, ToolCallingAgent]:
        """Create the graph builder agent with specialized graph batching tools."""

        if agent_mode not in {"code", "tool"}:
            raise ValueError("agent_mode must be 'code' or 'tool'")

        # Instantiate base LLM model
        llm_model = cls._create_model(model_id, temperature)
        model_kwargs = getattr(llm_model, "kwargs", None)
        graph_max_output_tokens = int(
            kwargs.get("max_output_tokens", 24_000)
        )
        if isinstance(model_kwargs, dict):
            configured_max = model_kwargs.get("max_tokens")
            if configured_max is None or int(configured_max) > graph_max_output_tokens:
                model_kwargs["max_tokens"] = graph_max_output_tokens

        # Shared components
        alias_registry = AliasRegistry()
        satisfaction_config = EvidenceSatisfactionConfig(
            min_graph_depth=kwargs.get("min_graph_depth", 3),
            min_graph_events=kwargs.get("min_events", 10),
        )

        # We need instances of base tools to pass to propose_subgraph
        evt_tool = EventIdentifierTool(
            db_path=db_path, question_id=question_id, alias_registry=alias_registry
        )
        reason_tool = CausalReasonerTool(
            db_path=db_path, question_id=question_id, alias_registry=alias_registry
        )

        tools = [
            QuestionArticlesTool(
                db_path=db_path,
                question_id=question_id,
                alias_registry=alias_registry,
            ),
            ArticleRetrievalTool(db_path=db_path),
            ProposeSubgraphTool(
                event_identifier_tool=evt_tool,
                causal_reasoner_tool=reason_tool,
                alias_registry=alias_registry,
                db_path=db_path,
                question_id=question_id,
            ),
            # Individual fallback tools using same alias registry
            evt_tool,
            reason_tool,
            RecordOutcomeImpactTool(db_path=db_path, question_id=question_id),
            GraphInspectorTool(
                db_path=db_path,
                question_id=question_id,
                default_compact=True,
                satisfaction_config=satisfaction_config,
            ),
            DeleteEventTool(db_path=db_path),
            DeleteHypothesisTool(db_path=db_path),
            MarkGraphBuiltTool(db_path=db_path, question_id=question_id),
        ]

        agent_kwargs = dict(
            model=llm_model,
            tools=tools,
            max_steps=kwargs.get("max_steps", 30),
            stream_outputs=False,
            step_callbacks=[_compact_graph_builder_history],
        )
        if agent_mode == "code":
            agent_kwargs["additional_authorized_imports"] = [
                "json",
                "datetime",
                "typing",
            ]
            # Keep enough output for aliases and validation feedback without
            # retaining repeated full graph payloads for every agent step.
            agent_kwargs["max_print_outputs_length"] = 20_000
            agent_kwargs["executor_kwargs"] = {"timeout_seconds": None}
            if _uses_structured_outputs(llm_model.model_id):
                agent_kwargs["use_structured_outputs_internally"] = True
            agent = CodeAgent(**agent_kwargs)
        else:
            # Keep DB-mutating graph calls sequential while removing generated
            # Python from the orchestration loop.
            agent_kwargs["max_tool_threads"] = 1
            agent = ToolCallingAgent(**agent_kwargs)

        return agent
