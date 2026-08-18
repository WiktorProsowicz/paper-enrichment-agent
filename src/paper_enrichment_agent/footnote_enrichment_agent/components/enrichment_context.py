"""Utilities for managing setup/teardown lifecycle of footnote enrichment tools."""

from collections.abc import Generator
from contextlib import contextmanager

import fastmcp

from paper_enrichment_agent.common.models import document as doc_models


class EnrichmentTools:
    """Represents the state of the footnote-enrichment tools."""

    type DocumentTree = str | list[DocumentTree] | dict[str, DocumentTree]

    def __init__(self, reference_document: doc_models.Document) -> None:

        self._document = reference_document

    def get_document_tree(self) -> 'DocumentTree':
        """Returns a tree representation of the reference document.

        The tree representation is a nested structure that captures the hierarchy of sections,
        paragraphs, and other components in the document. It makes it possible to determine the id
        of an arbitrary component and a path to it.
        """
        return ''

    def _convert_component_to_tree(self, component: doc_models.DocumentComponent) -> 'DocumentTree':
        """Converts a document component to its tree representation."""

        base_repr = {
            'type': component.__class__.__name__,
            'id': component.component_id,
        }

        if isinstance(component, doc_models.Section):
            return {
                **base_repr,
                'title': component.title,
                'components': [self._convert_component_to_tree(c) for c in component.components],
            }

        if isinstance(component, doc_models.List):
            return {
                **base_repr,
                'items': [self._convert_component_to_tree(i) for i in component.items],
            }

        if isinstance(component, doc_models.Figure):
            return {
                **base_repr,
                'caption': component.caption,
            }

        return base_repr


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
        self._mcp_endpoints: dict[str, fastmcp.FastMCP] = {}

    def get_mcp_for_agent_session(self, session_id: str) -> fastmcp.FastMCP:
        """Returns the MCP endpoint for a footnote-enrichment agent session.

        Args:
            session_id: The unique identifier of the footnote-enrichment agent session.
        """

        if session_id not in self._mcp_endpoints:
            raise self.EnrichmentContextManagerError(
                f'MCP endpoint for session {session_id} is not set up.'
            )

        return self._mcp_endpoints[session_id]

    @contextmanager
    def setup_mcp_for_agent_session(
        self, session_id: str, reference_document: doc_models.Document | None
    ) -> Generator[None, None, None]:
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

            self._tools_states[session_id] = EnrichmentTools(reference_document)

        tools_state = self._tools_states[session_id]

        mcp_endpoint = fastmcp.FastMCP(
            name=f'footnote_enrichment_tools_{session_id}',
            on_duplicate='error',
            strict_input_validation=True,
        )

        mcp_endpoint.tool()

        self._mcp_endpoints[session_id] = mcp_endpoint

        yield

        self._mcp_endpoints.pop(session_id)
