"""Contains helpers shared by the LLM agents calling their tools through MCP.

The helpers translate between the MCP tool interface and the LangChain chat model interface, and
trace the tool calls as MLflow spans. The span names (`call_tool <name>`) are relied upon by the
LLM evaluation scorers, so they must be kept in sync with them.
"""

from typing import Any

import mcp
import mlflow
from langchain_core.messages import AIMessage, ToolMessage
from mlflow.entities import SpanType


def mcp_tools_to_llm_functions(mcp_tools: list[mcp.types.Tool]) -> list[dict[str, Any]]:
    """Converts MCP tool definitions into OpenAI-style function definitions.

    The result can be passed directly to `bind_tools` of a LangChain chat model.

    Args:
        mcp_tools: The tools listed by an MCP session.

    Returns:
        The function definitions, one per tool.
    """

    return [
        {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


async def execute_tool_calls(
    mcp_session: mcp.ClientSession, message: AIMessage
) -> list[ToolMessage]:
    """Executes the tool calls requested by an LLM message within an MCP session.

    Each tool call is traced as a separate `call_tool <name>` MLflow span. Tool errors are not
    raised, but returned to the LLM as tool messages with the `error` status.

    Args:
        mcp_session: The initialized MCP session exposing the tools.
        message: The LLM message holding the tool calls.

    Returns:
        The tool responses, in the order of the tool calls.
    """

    tool_responses: list[ToolMessage] = []

    for tool_call in message.tool_calls:
        with mlflow.start_span(
            f'call_tool {tool_call["name"]}', span_type=SpanType.TOOL
        ) as tool_call_span:
            tool_call_span.set_inputs(tool_call['args'])

            tool_response = await mcp_session.call_tool(
                name=tool_call['name'],
                arguments=tool_call['args'],
            )

            tool_call_span.set_attributes(
                {
                    'tool_name': tool_call['name'],
                    'tool_is_error': tool_response.isError,
                }
            )
            tool_call_span.set_outputs(tool_response.structuredContent)

            if tool_response.isError:
                tool_call_span.record_exception(str(tool_response.content))

        tool_responses.append(
            ToolMessage(
                content='\n'.join(
                    block.text
                    if isinstance(block, mcp.types.TextContent)
                    else block.model_dump_json()
                    for block in tool_response.content
                ),
                tool_call_id=tool_call['id'],
                status='error' if tool_response.isError else 'success',
            )
        )

    return tool_responses
