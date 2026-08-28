"""Contains the implementation of the LLM agent for the `footnote_enrichment_agent` service.

The module defines the agent invocation graph, handles tool calling and MLFlow-based agent tracing.
"""

import datetime
import operator
import pathlib
from typing import Annotated, TypedDict

import mcp
import pydantic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from mcp.client.sse import sse_client as mcp_client

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import FootnoteEnrichmentRequest
from paper_enrichment_agent.footnote_enrichment_agent.components import (
    doc_utils,
    enrichment_context,
)

_PROMPT_TEMPLATES_PATH = pathlib.Path(__file__).parent / 'prompt_templates'


class AgentState(TypedDict):
    """Represents the current state of the footnote enrichment agent."""

    enrichment_session_id: str
    messages: Annotated[list[BaseMessage], operator.add]


class FootnoteEnrichmentAgent:
    """The LLM agent for the `footnote_enrichment_agent` service.

    In the agentic paradigm context, this is a ReAct agent, which consists of a simple
    reason + tool execution loop.
    """

    class Configuration(pydantic.BaseModel):
        """The configuration of the footnote enrichment agent."""

        citation_context_chars: Annotated[
            int,
            pydantic.Field(
                description=(
                    'The number of characters to include in the context of a citation. '
                    'The context is extracted from the stringified form of the paragraph, '
                    'which contains the citation.'
                ),
            ),
        ]

        mcp_server_url_base: Annotated[
            str,
            pydantic.Field(
                description=(
                    'The base URL of the MCP server to connect to for tool execution. '
                    'The agent will connect to this server using the enrichment session-specific '
                    'endpoint.'
                ),
            ),
        ]

    # Number of seconds to wait for a response from the MCP server before timing out.
    _MCP_OPERATION_TIMEOUT = datetime.timedelta(seconds=5)

    def __init__(self, llm: ChatLiteLLM, cfg: 'Configuration') -> None:

        self._llm = llm
        self._cfg = cfg
        self._agent_checkpointer = InMemorySaver()

        self._enrichment_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template_file(
                    _PROMPT_TEMPLATES_PATH.joinpath('footnote_enrichment_system_prompt.md'),
                    input_variables=['survey_title', 'survey_abstract'],
                ),
                HumanMessagePromptTemplate.from_template_file(
                    _PROMPT_TEMPLATES_PATH.joinpath('footnote_enrichment_user_prompt.md'),
                    input_variables=[
                        'cited_doc_title',
                        'cited_doc_abstract',
                        'citation_reference_text',
                        'citation_context_fragment',
                        'citation_context_info',
                    ],
                ),
            ]
        )

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

        builder.add_node('enrich_new_footnote', self._enrich_new_footnote)  # type: ignore[call-overload]
        builder.add_node('perform_agent_reasoning_step', self._perform_agent_reasoning_step)
        builder.add_node('perform_agent_action_step', self._perform_agent_action_step)

        builder.add_edge(START, 'enrich_new_footnote')
        builder.add_edge('enrich_new_footnote', 'perform_agent_reasoning_step')
        builder.add_conditional_edges(
            'perform_agent_reasoning_step',
            self._call_tool_or_finish_loop,
            {
                'call_tool': 'perform_agent_action_step',
                'finish': END,
            },
        )
        builder.add_edge('perform_agent_action_step', 'perform_agent_reasoning_step')

        graph = builder.compile(checkpointer=self._agent_checkpointer)

        await graph.ainvoke(request, config={'configurable': {'thread_id': request.session_id}})

        return enrichment_tools.footnote

    async def _enrich_new_footnote(self, request: FootnoteEnrichmentRequest) -> AgentState:
        """The entrypoint of the footnote enrichment pipeline.

        This is the beginning of a single agent session turn, where the initial user query is built.
        """

        citation_reference = next(
            element
            for element in request.referencing_paragraph.elements
            if isinstance(element, doc_models.Reference)
            and element.component_id == request.reference_id
        )

        citation_context = doc_utils.get_context_of_citation_in_paragraph(
            request.referencing_paragraph,
            citation_reference,
            n_context_chars=self._cfg.citation_context_chars,
        )

        return {
            'messages': self._enrichment_prompt_template.format_messages(
                survey_title=request.survey_title,
                survey_abstract=request.survey_abstract,
                cited_doc_title=request.reference_document_title,
                cited_doc_abstract=request.reference_document.abstract,
                citation_reference_text=citation_reference.content_text,
                citation_context_fragment=citation_context,
                citation_context_info=request.citation_context_info,
            ),
            'enrichment_session_id': request.session_id,
        }

    async def _perform_agent_reasoning_step(self, state: AgentState) -> AgentState:
        """Performs a single step of the agent's reasoning and tool execution loop.

        First, the appropriate MCP endpoint is connected to get available tools. Then, the agent is
        invoked and its returned messages are appended to the conversation history.
        """

        mcp_url = f'{self._cfg.mcp_server_url_base}/{state["enrichment_session_id"]}'

        async with mcp_client(mcp_url) as connection_streams:  # noqa: F841, SIM117
            async with mcp.ClientSession(
                **connection_streams, read_timeout_seconds=self._MCP_OPERATION_TIMEOUT
            ) as mcp_session:
                await mcp_session.initialize()

                mcp_tools = (await mcp_session.list_tools()).tools

                llm_with_tools = self._llm.bind_tools(
                    [
                        {
                            'name': tool.name,
                            'description': tool.description,
                            'inputSchema': tool.inputSchema,
                            'outputSchema': tool.outputSchema,
                        }
                        for tool in mcp_tools
                    ],
                    tool_invoker='auto',
                )

                model_response = await llm_with_tools.ainvoke(state['messages'])

                return {'messages': state['messages'] + [model_response]}  # type: ignore[typeddict-item]

    async def _perform_agent_action_step(self, state: AgentState) -> AgentState:
        """Performs a single step of the agent's action execution loop.

        The agent's messages are parsed to extract the tool invocation action. Then, the appropriate
        MCP endpoint is connected to execute the tool and get the result. Finally, the result is
        appended to the conversation history.
        """

        mcp_url = f'{self._cfg.mcp_server_url_base}/{state["enrichment_session_id"]}'

        last_message = state['messages'][-1]

        if not isinstance(last_message, AIMessage):
            raise RuntimeError(
                f'Expected last message to be of type AIMessage, but got {type(last_message)}'
            )

        tool_responses: list[ToolMessage] = []

        async with mcp_client(mcp_url) as connection_streams:  # noqa: F841, SIM117
            async with mcp.ClientSession(
                **connection_streams, read_timeout_seconds=self._MCP_OPERATION_TIMEOUT
            ) as mcp_session:
                await mcp_session.initialize()

                for tool_call in last_message.tool_calls:
                    tool_response = await mcp_session.call_tool(
                        name=tool_call['name'],
                        arguments=tool_call['args'],
                    )

                    tool_responses.append(
                        ToolMessage(
                            content=tool_response.model_dump_json(), tool_call_id=tool_call['id']
                        )
                    )

        return {'messages': state['messages'] + tool_responses}  # type: ignore[typeddict-item]

    async def _call_tool_or_finish_loop(self, state: AgentState) -> str:
        """Decides whether to call a tool or finish the agent's reasoning loop."""

        last_message = state['messages'][-1]

        if not isinstance(last_message, AIMessage):
            raise RuntimeError(
                f'Expected last message to be of type AIMessage, but got {type(last_message)}'
            )

        if last_message.tool_calls:
            return 'call_tool'
        else:
            return 'finish'
