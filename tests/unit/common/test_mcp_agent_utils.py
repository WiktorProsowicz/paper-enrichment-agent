from unittest.mock import MagicMock

import fastmcp
import pytest
from langchain_core.messages import AIMessage

from paper_enrichment_agent.common import mcp_agent_utils


@pytest.fixture
def mcp_server() -> fastmcp.FastMCP:

    server = fastmcp.FastMCP(name='test_tools')

    @server.tool
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    @server.tool
    def fail(reason: str) -> str:
        """Always fails."""
        raise ValueError(f'Failed: {reason}')

    return server


class TestMcpToolsToLlmFunctions:
    @pytest.mark.asyncio
    async def test_converts_tool_definitions(self, mcp_server):

        async with fastmcp.Client(mcp_server) as client:
            mcp_tools = (await client.session.list_tools()).tools

        functions = mcp_agent_utils.mcp_tools_to_llm_functions(mcp_tools)

        add_function = next(f for f in functions if f['function']['name'] == 'add')
        assert add_function['type'] == 'function'
        assert add_function['function']['description'] == 'Adds two numbers.'
        assert set(add_function['function']['parameters']['properties']) == {'a', 'b'}
        assert len(functions) == 2


class TestExecuteToolCalls:
    @pytest.fixture(autouse=True)
    def mock_start_span(self, monkeypatch) -> MagicMock:

        start_span = MagicMock()
        monkeypatch.setattr(mcp_agent_utils.mlflow, 'start_span', start_span)

        return start_span

    @pytest.mark.asyncio
    async def test_returns_tool_responses_in_order(self, mcp_server):

        message = AIMessage(
            content='',
            tool_calls=[
                {'name': 'add', 'args': {'a': 1, 'b': 2}, 'id': 'call_1'},
                {'name': 'fail', 'args': {'reason': 'boom'}, 'id': 'call_2'},
            ],
        )

        async with fastmcp.Client(mcp_server) as client:
            responses = await mcp_agent_utils.execute_tool_calls(client.session, message)

        assert [response.tool_call_id for response in responses] == ['call_1', 'call_2']

        assert responses[0].status == 'success'
        assert responses[0].content == '3'

        assert responses[1].status == 'error'
        assert 'Failed: boom' in responses[1].content

    @pytest.mark.asyncio
    async def test_traces_tool_calls(self, mcp_server, mock_start_span):

        message = AIMessage(
            content='',
            tool_calls=[
                {'name': 'add', 'args': {'a': 1, 'b': 2}, 'id': 'call_1'},
                {'name': 'fail', 'args': {'reason': 'boom'}, 'id': 'call_2'},
            ],
        )

        async with fastmcp.Client(mcp_server) as client:
            await mcp_agent_utils.execute_tool_calls(client.session, message)

        assert [call.args[0] for call in mock_start_span.call_args_list] == [
            'call_tool add',
            'call_tool fail',
        ]

        span = mock_start_span.return_value.__enter__.return_value
        span.set_inputs.assert_any_call({'a': 1, 'b': 2})
        span.set_attributes.assert_any_call({'tool_name': 'fail', 'tool_is_error': True})
        span.record_exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_nothing_without_tool_calls(self, mcp_server):

        async with fastmcp.Client(mcp_server) as client:
            responses = await mcp_agent_utils.execute_tool_calls(
                client.session, AIMessage(content='Done.')
            )

        assert responses == []
