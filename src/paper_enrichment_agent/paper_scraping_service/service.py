"""Contains the core logic of the `paper_scraping_service` service.

The `paper_scraping_service` service downloads papers from external sources and converts them into
the structured :class:`~paper_enrichment_agent.common.models.document.Document` format, together
with their bibliographic metadata. It is called by the other services of the application.
"""

import asyncio
import time
from functools import cache

from prometheus_client import Counter, Summary

from paper_enrichment_agent.common import logging_setup
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
)
from paper_enrichment_agent.paper_scraping_service.components.parsers.arxiv_parser import (
    ArxivParser,
)


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


class PaperScrapingService:
    """The main service of the `paper_scraping_service` component."""

    class PaperScrapingError(Exception):
        """Base class for exceptions raised by the `paper_scraping_service` service."""

    def __init__(self, metrics: 'Metrics', arxiv_metadata_fetcher: ArxivMetadataFetcher) -> None:

        self._metrics = metrics
        self._arxiv_metadata_fetcher = arxiv_metadata_fetcher

    async def parse_arxiv_paper(
        self, arxiv_id: str
    ) -> tuple[doc_models.Document, misc_models.PaperMetadata]:
        """Downloads an arXiv paper and parses it into the document structure.

        The paper's content and its metadata are fetched concurrently, each in a separate thread,
        since the parser and the fetcher are synchronous and the HTML parsing is CPU-bound. A new
        parser is created per paper, as it keeps the state of a single parsing.

        Args:
            arxiv_id: The arXiv ID of the paper to parse.

        Returns:
            The parsed document and the metadata of the paper.

        Raises:
            PaperScrapingError: If the paper or its metadata cannot be downloaded or parsed.
        """

        try:
            start = time.perf_counter()
            document, metadata = await asyncio.gather(
                asyncio.to_thread(ArxivParser().parse, arxiv_id),
                asyncio.to_thread(self._arxiv_metadata_fetcher.fetch, arxiv_id),
            )
            end = time.perf_counter()

            _logger().info(
                'Successfully parsed the arXiv paper.',
                arxiv_id=arxiv_id,
                title=document.title,
                metadata=metadata,
            )
            self._metrics.papers_parsed.labels(source='arxiv', status='success').inc()
            self._metrics.paper_parsing_time.labels(source='arxiv').observe(end - start)

            return document, metadata

        except (ArxivParser.ParsingError, ArxivMetadataFetcher.FetchingError) as e:
            _logger().error('Failed to parse the arXiv paper.', arxiv_id=arxiv_id, error=str(e))
            self._metrics.papers_parsed.labels(source='arxiv', status='failure').inc()

            raise self.PaperScrapingError(
                f'Failed to parse the arXiv paper with ID {arxiv_id}: {e}'
            ) from e


class Metrics:
    """Metrics for the `paper_scraping_service` service.

    The metrics are registered in the default Prometheus registry, so a single `Metrics` instance
    should be created per process. They are exposed by the service's `/metrics` endpoint.
    """

    # Sources of the papers the service can scrape
    _SOURCES = ('arxiv',)

    def __init__(self) -> None:

        self.papers_parsed = Counter(
            'papers_parsed',
            'Number of papers scraped and parsed by the service, including their metadata.',
            labelnames=['source', 'status'],  # status can be 'success' or 'failure'
            namespace='paper_scraping_service',
        )

        self.paper_parsing_time = Summary(
            'paper_parsing_time',
            'Time spent on successful scraping and parsing of papers, including the downloads.',
            labelnames=['source'],
            namespace='paper_scraping_service',
        )

        for source in self._SOURCES:
            self.paper_parsing_time.labels(source=source)

            for status in ('success', 'failure'):
                self.papers_parsed.labels(source=source, status=status)
