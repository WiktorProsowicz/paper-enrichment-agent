"""Orchestration of the MCP endpoints exposing the footnote enrichment tools.

The enrichment context of an agent session consists of the state of the enrichment tools and of
the MCP endpoint exposing them. The endpoint is attached to the API of the
`footnote_enrichment_agent` service for the time of the session.

Exported classes:
    EnrichmentContextManager: Manages the setup and teardown of the enrichment contexts.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import fastmcp
from starlette.applications import Starlette

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.footnote_enrichment_agent.components.enrichment_tools import (
    EnrichmentTools,
)


class EnrichmentContextManager:
    """Manages the setup and teardown of the footnote-enrichment MCP tools state.

    The enrichment context manager creates and stores the state of the footnote-enrichment tools,
    which are exposed within MCP server to the LLM-based enrichment agent. The tools state is
    initialized for each agent session and may be reused on enrichment continuation.
    """

    class EnrichmentContextManagerError(Exception):
        """Base class for exceptions raised by the `EnrichmentContextManager`."""

    def __init__(self) -> None:

        self._tools_states: dict[str, EnrichmentTools] = {}
        self._mcp_endpoints: dict[str, Starlette] = {}

    def get_mcp_app_for_agent_session(self, session_id: str) -> Starlette:
        """Returns the MCP app for a footnote-enrichment agent session.

        Args:
            session_id: The unique identifier of the footnote-enrichment agent session.
        """

        if session_id not in self._mcp_endpoints:
            raise self.EnrichmentContextManagerError(
                f'MCP endpoint for session {session_id} is not set up.'
            )

        return self._mcp_endpoints[session_id]

    def get_footnote_state(self, session_id: str) -> doc_models.Section:
        """Returns the footnote composed within the enrichment context of an agent session.

        Args:
            session_id: The unique identifier of the footnote-enrichment agent session.

        Raises:
            EnrichmentContextManagerError: If the tools state of the session is not set up.
        """

        if session_id not in self._tools_states:
            raise self.EnrichmentContextManagerError(
                f'Enrichment tools for session {session_id} are not set up.'
            )

        return self._tools_states[session_id].footnote

    @asynccontextmanager
    async def setup_mcp_for_agent_session(
        self, session_id: str, reference_document: doc_models.Document | None
    ) -> AsyncGenerator[EnrichmentTools, None]:
        """Sets up a child MCP endpoint for a footnote-enrichment agent session.

        The exposed MCP is attached to the primary MCP root endpoint and delegates the
        footnote-enrichment tool calls to the created / reused enrichment tools state.

        Args:
            session_id: The unique identifier of the footnote-enrichment agent session.
            reference_document: The document referenced by the survey, which is used to
                initialize the enrichment tools state. If None, the existing tools state is reused.
        """

        if not reference_document:
            if session_id not in self._tools_states:
                raise self.EnrichmentContextManagerError(
                    f'Cannot setup MCP for session {session_id} without a reference document.'
                )

        else:
            self._tools_states[session_id] = EnrichmentTools(reference_document)

        tools_state = self._tools_states[session_id]

        mcp_server = fastmcp.FastMCP(
            name=f'footnote_enrichment_tools_{session_id}',
            on_duplicate='error',
            strict_input_validation=True,
        )

        mcp_server.tool(tools_state.get_document_tree, name='get_document_tree')
        mcp_server.tool(tools_state.set_footnote_title, name='set_footnote_title')
        mcp_server.tool(tools_state.get_paragraph_content, name='get_paragraph_content')
        mcp_server.tool(tools_state.get_figure_details, name='get_figure_details')
        mcp_server.tool(tools_state.extract_paragraph_citation, name='extract_paragraph_citation')
        mcp_server.tool(tools_state.extract_figure, name='extract_figure')

        mcp_endpoint = mcp_server.http_app(transport='streamable-http')
        self._mcp_endpoints[session_id] = mcp_endpoint

        try:
            async with mcp_endpoint.router.lifespan_context(mcp_endpoint):
                yield tools_state
        finally:
            self._mcp_endpoints.pop(session_id)
