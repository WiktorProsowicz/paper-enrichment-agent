"""Contains the LLM agent finding publicly available versions of papers from citation signatures.

The agent is a ReAct agent, which queries external bibliographic services through the
:class:`~paper_enrichment_agent.paper_scraping_service.components.paper_access_tools.PaperAccessTools`.
The tools are exposed through an MCP server created by the agent and connected to in-memory, so no
separate server process or endpoint is involved.

Exported classes:
    PaperAccessType: Enumerates the kinds of public sources of papers.
    PaperAccessInfo: Describes how to access a paper and its bibliographic metadata.
    PaperAccessAgent: The LLM agent resolving citation signatures into paper access information.
"""

import dataclasses
import enum
import operator
import pathlib
import re
from typing import Annotated, Any, TypedDict

import fastmcp
import httpx
import mcp
import mlflow
import pydantic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from langchain_litellm import ChatLiteLLM
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime as LangGraphRuntime
from mlflow.entities import SpanType
from pydantic import Field

from paper_enrichment_agent.common import mcp_agent_utils
from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.paper_scraping_service.components.paper_access_tools import (
    PaperAccessTools,
)


class PaperAccessType(enum.StrEnum):
    """Enumerates the kinds of public sources of papers."""

    # The paper has an HTML version on arXiv, the source is its arXiv ID
    ARXIV = 'arxiv'

    # The paper is available as an open access PDF, the source is the URL of the PDF
    PDF = 'pdf'


class PaperAccessInfo(pydantic.BaseModel):
    """Describes how to access a publicly available version of a paper."""

    type: Annotated[PaperAccessType, Field(description='The kind of the public source.')]
    source: Annotated[
        str,
        Field(
            description=(
                'The arXiv ID of the paper (e.g. `1706.03762`) for the `arxiv` type, or the URL of '
                'the PDF file for the `pdf` type.'
            ),
            min_length=1,
        ),
    ]
    metadata: Annotated[
        misc_models.PaperMetadata,
        Field(description='The bibliographic metadata of the paper.'),
    ]


class _AgentState(TypedDict):
    """Represents the current state of the paper access agent.

    Attributes:
        messages: The conversation history of the agent, which consists of the initial task
            description and the messages appended at each reasoning / action step.
    """

    messages: Annotated[list[BaseMessage], operator.add]


@dataclasses.dataclass
class _AgentContext:
    """Represents immutable context data of a single invocation of the paper access agent."""

    mcp_session: mcp.ClientSession
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage]


class PaperAccessAgent:
    """The LLM agent resolving citation signatures into paper access information.

    In the agentic paradigm context, this is a ReAct agent, which consists of a simple
    reason + tool execution loop. For each invocation, fresh tools are created from the
    configuration, with their own HTTP client, and exposed through an MCP server connected to
    in-memory, before the agent graph is run. The tools and their HTTP client are torn down when
    the invocation finishes, so concurrent invocations share no state.

    The agent raises `PaperAccessAgentError` if it does not finish within the recursion limit or
    its final answer cannot be decoded.
    """

    class PaperAccessAgentError(Exception):
        """Raised when the agent fails to resolve the access information of a paper."""

    class Configuration(pydantic.BaseModel):
        """The configuration of the paper access agent."""

        agent_recursion_limit: Annotated[
            int,
            Field(
                description=(
                    'The maximum number of graph steps of a single agent invocation. '
                    "This prevents infinite loops in the agent's reasoning and tool usage."
                ),
                ge=1,
            ),
        ]
        tools: Annotated[
            PaperAccessTools.Configuration,
            Field(description='The configuration of the tools created for each invocation.'),
        ]

    _PROMPT_TEMPLATES_PATH = pathlib.Path(__file__).parent / 'prompt_templates'

    # Matches a final answer wrapped in a Markdown code block
    _CODE_BLOCK_PATTERN = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.DOTALL)

    _ANSWER_ADAPTER: pydantic.TypeAdapter[PaperAccessInfo | None] = pydantic.TypeAdapter(
        PaperAccessInfo | None
    )

    def __init__(self, llm: ChatLiteLLM, cfg: Configuration) -> None:

        self._llm = llm
        self._cfg = cfg

        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template_file(
                    self._PROMPT_TEMPLATES_PATH.joinpath('paper_access_system_prompt.md'),
                    input_variables=['output_schema'],
                ),
                HumanMessagePromptTemplate.from_template_file(
                    self._PROMPT_TEMPLATES_PATH.joinpath('paper_access_user_prompt.md'),
                    input_variables=['citation_signature'],
                ),
            ]
        )

    @mlflow.trace(name='PaperAccessAgent::invoke', span_type=SpanType.AGENT)
    async def invoke(self, citation_signature: str) -> PaperAccessInfo | None:
        """Invokes the LLM agent to find a publicly available version of a cited paper.

        The tools of the agent are set up before the agent graph is run and torn down afterwards.

        Args:
            citation_signature: The citation signature of the paper, as found in the bibliography
                of the citing paper.

        Returns:
            The access information of the paper, or None if no public version of it was found.

        Raises:
            PaperAccessAgentError: If the agent does not finish within the recursion limit or its
                final answer is not a valid access information.
        """

        builder = StateGraph(_AgentState, context_schema=_AgentContext)

        builder.add_node('perform_agent_reasoning_step', self._perform_agent_reasoning_step)
        builder.add_node('perform_agent_action_step', self._perform_agent_action_step)

        builder.add_edge(START, 'perform_agent_reasoning_step')
        builder.add_conditional_edges(
            'perform_agent_reasoning_step',
            self._call_tool_or_finish_loop,
            {
                'call_tool': 'perform_agent_action_step',
                'finish': END,
            },
        )
        builder.add_edge('perform_agent_action_step', 'perform_agent_reasoning_step')

        graph = builder.compile()

        initial_messages = self._prompt_template.format_messages(
            output_schema=PaperAccessInfo.model_json_schema(),
            citation_signature=citation_signature,
        )

        async with (
            httpx.AsyncClient() as http_client,
            fastmcp.Client(
                PaperAccessTools.create(self._cfg.tools, http_client).as_mcp_server()
            ) as mcp_client,
        ):
            mcp_session = mcp_client.session
            mcp_tools = (await mcp_session.list_tools()).tools
            llm_with_tools = self._llm.bind_tools(
                mcp_agent_utils.mcp_tools_to_llm_functions(mcp_tools), tool_choice='auto'
            )

            try:
                final_state = await graph.ainvoke(
                    {'messages': initial_messages},
                    context=_AgentContext(mcp_session=mcp_session, llm_with_tools=llm_with_tools),
                    config={'recursion_limit': self._cfg.agent_recursion_limit},
                )

            except GraphRecursionError as e:
                raise self.PaperAccessAgentError(
                    'The agent did not finish within the recursion limit '
                    f'({self._cfg.agent_recursion_limit}).'
                ) from e

        return self._decode_final_answer(str(final_state['messages'][-1].text))

    @mlflow.trace(name='agent_reasoning', span_type=SpanType.CHAIN)
    async def _perform_agent_reasoning_step(
        self, state: _AgentState, runtime: LangGraphRuntime[_AgentContext]
    ) -> dict[str, Any]:
        """Invokes the LLM and appends its response to the conversation history."""

        if curr_span := mlflow.get_current_active_span():
            curr_span.set_inputs(state)

        model_response = await runtime.context.llm_with_tools.ainvoke(state['messages'])

        return {'messages': [model_response]}

    @mlflow.trace(name='agent_action', span_type=SpanType.CHAIN)
    async def _perform_agent_action_step(
        self, state: _AgentState, runtime: LangGraphRuntime[_AgentContext]
    ) -> dict[str, Any]:
        """Executes the tool calls of the last LLM message and appends their results."""

        if curr_span := mlflow.get_current_active_span():
            curr_span.set_inputs(state)

        last_message = state['messages'][-1]

        if not isinstance(last_message, AIMessage):
            raise RuntimeError(
                f'Expected last message to be of type AIMessage, but got {type(last_message)}'
            )

        tool_responses = await mcp_agent_utils.execute_tool_calls(
            runtime.context.mcp_session, last_message
        )

        return {'messages': tool_responses}

    async def _call_tool_or_finish_loop(self, state: _AgentState) -> str:
        """Decides whether to call a tool or finish the agent's reasoning loop."""

        last_message = state['messages'][-1]

        if not isinstance(last_message, AIMessage):
            raise RuntimeError(
                f'Expected last message to be of type AIMessage, but got {type(last_message)}'
            )

        return 'call_tool' if last_message.tool_calls else 'finish'

    def _decode_final_answer(self, answer: str) -> PaperAccessInfo | None:
        """Decodes the final answer of the agent into the paper access information."""

        answer = answer.strip()

        if match := self._CODE_BLOCK_PATTERN.match(answer):
            answer = match.group(1)

        try:
            return self._ANSWER_ADAPTER.validate_json(answer)

        except pydantic.ValidationError as e:
            raise self.PaperAccessAgentError(
                f'The final answer of the agent is not a valid paper access information: {e}'
            ) from e
