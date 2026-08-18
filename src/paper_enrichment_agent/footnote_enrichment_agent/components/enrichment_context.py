"""Utilities for managing setup/teardown lifecycle of footnote enrichment tools."""

from contextlib import contextmanager

from paper_enrichment_agent.common.models import document as doc_models

class EnrichmentTools:
    """Represents the state of the footnote-enrichment tools."""

    def __init__(self, reference_document: doc_models.Document) -> None:

        self._document = reference_document


class EnrichmentContextManager:
    """Manages the setup and teardown of the footnote-enrichment MCP tools state.

    The enrichment context manager creates and stores the state of the footnote-enrichment tools,
    which are exposed within MCP server to the LLM-based enrichment agent. The tools state is
    initialized for each agent session and may be reused on enrichment continuation.
    """

    def __init__(self) -> None:

        self._tools_states: dict[str, EnrichmentTools] = {}

    @contextmanager
    def 