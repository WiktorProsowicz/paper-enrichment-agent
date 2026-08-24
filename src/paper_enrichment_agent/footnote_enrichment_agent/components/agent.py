"""Contains the implementation of the LLM agent for the `footnote_enrichment_agent` service.

The module defines the agent invocation graph, handles tool calling and MLFlow-based agent tracing.
"""

from typing import TypedDict, Annotated
import operator

from langchain.messages import AnyMessage
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import FootnoteEnrichmentRequest
from paper_enrichment_agent.footnote_enrichment_agent.components import enrichment_context


class AgentState(TypedDict):
    """Represents the current state of the footnote enrichment agent."""

    enrichment_session_id: str
    messages: Annotated[list[AnyMessage], operator.add]


class FootnoteEnrichmentAgent:
    """The LLM agent for the `footnote_enrichment_agent` service."""

    def __init__(self, llm: ChatLiteLLM) -> None:
        self._llm = llm
        self._agent_checkpointer = InMemorySaver()

    async def invoke(
        self,
        request: FootnoteEnrichmentRequest,
        enrichment_tools: enrichment_context.EnrichmentTools,
    ) -> doc_models.Section:
        """Invokes the LLM agent to compose a footnote from the given document.

        First, the LLM invocation graph is built / restored for the enrichment session. Then, the
        agent is invoked to compose a footnote from the given document.

        Args:
            request: The footnote enrichment request data.
            enrichment_tools: The current state of the tools available for footnote enrichment.
        """

        builder = StateGraph(AgentState, input_schema=FootnoteEnrichmentRequest)
        graph = builder.compile(checkpointer=self._agent_checkpointer)

        await graph.ainvoke(request, config={'configurable': {'thread_id': request.session_id}})

        return enrichment_tools.footnote

    async def _enrich_new_footnote(self, request: FootnoteEnrichmentRequest) -> AgentState:
        """The entrypoint of the footnote enrichment pipeline.

        This is the beginning of a single agent session turn, where the initial user query is built.
        """

        return {'messages': [], 'enrichment_session_id': request.session_id}
