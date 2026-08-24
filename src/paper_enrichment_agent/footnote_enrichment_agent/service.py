"""Contains the core logic of the `footnote_enrichment_agent` service.

The `footnote_enrichment_agent` service orchestrates the operations of a LLM-based agent, which
composes meaningful document footnotes from the given document and its role in the survey. The
primary use case of the service is to extract excerpts from the document referenced by the survey
and compose them into a footnote, which thoroughly explains the document's contribution.
"""

import time
from functools import cache

from prometheus_client import Counter, Summary

from paper_enrichment_agent.common import logging_setup
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import FootnoteEnrichmentRequest
from paper_enrichment_agent.footnote_enrichment_agent.components import enrichment_context
from paper_enrichment_agent.footnote_enrichment_agent.components.agent import (
    FootnoteEnrichmentAgent,
)


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


class FootnoteEnrichmentAgentService:
    """The main orchestrator of the `footnote_enrichment_agent` service."""

    class FootnoteEnrichmentAgentError(Exception):
        """Base class for exceptions raised by the `footnote_enrichment_agent` service."""

    def __init__(
        self,
        metrics: 'Metrics',
        enrichment_context_manager: enrichment_context.EnrichmentContextManager,
        enrichment_agent: FootnoteEnrichmentAgent,
    ) -> None:
        self._metrics = metrics
        self._enrichment_context_manager = enrichment_context_manager
        self._enrichment_agent = enrichment_agent

    async def reference_to_footnote(self, request: FootnoteEnrichmentRequest) -> doc_models.Section:
        """Calls the footnote enrichment agent to compose a footnote from the given document.

        Args:
            request: The footnote enrichment request data.
        """

        try:
            with self._enrichment_context_manager.setup_mcp_for_agent_session(
                request.session_id, request.reference_document
            ) as enrichment_tools:
                start_time = time.perf_counter()
                footnote = await self._enrichment_agent.invoke(request, enrichment_tools)
                end_time = time.perf_counter()

                self._metrics.enrichment_time.observe(end_time - start_time)
                self._metrics.enrichment_requests.labels(status='success').inc()

                _logger().info(
                    'Successfully enriched footnote for agent session',
                    session_id=request.session_id,
                    survey_title=request.survey_title,
                    reference_document_title=request.reference_document_title,
                )

                return footnote

        except enrichment_context.EnrichmentContextManager.EnrichmentContextManagerError as e:
            self._metrics.enrichment_requests.labels(status='failure').inc()
            _logger().error('Failed to set up enrichment context for agent session', error=str(e))

            raise self.FootnoteEnrichmentAgentError(
                'Failed to set up enrichment context for agent session'
            ) from e


class Metrics:
    """Metrics for the `footnote_enrichment_agent` service."""

    def __init__(self) -> None:

        self.enrichment_requests = Counter(
            'enrichment_requests',
            'Number of footnote enrichment requests received by the system.',
            namespace='footnote_enrichment_agent',
            labelnames=['status'],  # status can be 'success' or 'failure'
        )

        self.improvement_requests = Counter(
            'improvement_requests',
            'Number of footnote improvement requests received by the system.',
            namespace='footnote_enrichment_agent',
            labelnames=['status'],  # status can be 'success' or 'failure'
        )

        self.enrichment_time = Summary(
            'enrichment_time',
            'Time spent on successful footnote enrichment requests.',
            namespace='footnote_enrichment_agent',
        )
