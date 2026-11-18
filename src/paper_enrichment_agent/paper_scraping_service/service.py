"""Contains the core logic of the `paper_scraping_service` service.

The `paper_scraping_service` service downloads papers from external sources and converts them into
the structured :class:`~paper_enrichment_agent.common.models.document.Document` format, together
with their bibliographic metadata. It is called by the other services of the application.
"""

import asyncio
import time
from collections.abc import Callable
from functools import cache

from prometheus_client import Counter, Summary

from paper_enrichment_agent.common import logging_setup
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
)
from paper_enrichment_agent.paper_scraping_service.components.paper_access_agent import (
    PaperAccessAgent,
    PaperAccessType,
)
from paper_enrichment_agent.paper_scraping_service.components.parsers.arxiv_parser import (
    ArxivParser,
)
from paper_enrichment_agent.paper_scraping_service.components.parsers.pdf_parser import PdfParser


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


class PaperScrapingService:
    """The main service of the `paper_scraping_service` component."""

    class PaperScrapingError(Exception):
        """Base class for exceptions raised by the `paper_scraping_service` service."""

    def __init__(
        self,
        metrics: 'Metrics',
        arxiv_metadata_fetcher: ArxivMetadataFetcher,
        paper_access_agent: PaperAccessAgent,
        pdf_parser_factory: Callable[[], PdfParser],
    ) -> None:
        """Initializes the service.

        Args:
            metrics: The metrics of the service.
            arxiv_metadata_fetcher: The fetcher of the arXiv papers' metadata.
            paper_access_agent: The agent finding public versions of papers by their citations.
            pdf_parser_factory: Creates a new PDF parser, since a parser handles a single paper.
        """

        self._metrics = metrics
        self._arxiv_metadata_fetcher = arxiv_metadata_fetcher
        self._paper_access_agent = paper_access_agent
        self._pdf_parser_factory = pdf_parser_factory

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

    async def parse_paper_from_citation_signature(
        self, citation_signature: str
    ) -> tuple[doc_models.Document, misc_models.PaperMetadata] | None:
        """Finds a public version of a cited paper and parses it into the document structure.

        The paper access agent identifies the paper and its bibliographic metadata, and picks its
        best public source: the HTML version on arXiv, or an open access PDF. The paper is then
        parsed by the respective parser in a separate thread, as the parsers are synchronous.

        Args:
            citation_signature: The citation signature of the paper, as found in the bibliography
                of the citing paper.

        Returns:
            The parsed document and the metadata of the paper, or None if no public version of the
            paper was found.

        Raises:
            PaperScrapingError: If the access information of the paper cannot be resolved, or the
                paper cannot be downloaded or parsed.
        """

        try:
            start = time.perf_counter()
            access_info = await self._paper_access_agent.invoke(citation_signature)
            end = time.perf_counter()

        except PaperAccessAgent.PaperAccessAgentError as e:
            _logger().error(
                'Failed to resolve the access information of the paper.',
                citation_signature=citation_signature,
                error=str(e),
            )
            self._metrics.paper_access_resolutions.labels(result='failure').inc()

            raise self.PaperScrapingError(
                f'Failed to find the paper cited as "{citation_signature}": {e}'
            ) from e

        self._metrics.paper_access_resolution_time.observe(end - start)

        if access_info is None:
            _logger().info(
                'No public version of the paper was found.', citation_signature=citation_signature
            )
            self._metrics.paper_access_resolutions.labels(result='not_found').inc()
            return None

        _logger().info(
            'Resolved the access information of the paper.',
            citation_signature=citation_signature,
            access_info=access_info,
        )
        self._metrics.paper_access_resolutions.labels(result=access_info.type.value).inc()

        try:
            start = time.perf_counter()

            match access_info.type:
                case PaperAccessType.ARXIV:
                    document = await asyncio.to_thread(ArxivParser().parse, access_info.source)
                case PaperAccessType.PDF:
                    document = await asyncio.to_thread(
                        self._pdf_parser_factory().parse, access_info.source
                    )

            end = time.perf_counter()

        except (ArxivParser.ParsingError, PdfParser.ParsingError) as e:
            _logger().error(
                'Failed to parse the cited paper.',
                citation_signature=citation_signature,
                access_info=access_info,
                error=str(e),
            )
            self._metrics.papers_parsed.labels(
                source=access_info.type.value, status='failure'
            ).inc()

            raise self.PaperScrapingError(
                f'Failed to parse the paper cited as "{citation_signature}" from '
                f'{access_info.type.value} source {access_info.source}: {e}'
            ) from e

        _logger().info(
            'Successfully parsed the cited paper.',
            citation_signature=citation_signature,
            title=document.title,
        )
        self._metrics.papers_parsed.labels(source=access_info.type.value, status='success').inc()
        self._metrics.paper_parsing_time.labels(source=access_info.type.value).observe(end - start)

        return document, access_info.metadata


class Metrics:
    """Metrics for the `paper_scraping_service` service.

    The metrics are registered in the default Prometheus registry, so a single `Metrics` instance
    should be created per process. They are exposed by the service's `/metrics` endpoint.
    """

    # Sources of the papers the service can scrape
    _SOURCES = ('arxiv', 'pdf')

    # Results of resolving the access information of cited papers
    _ACCESS_RESOLUTION_RESULTS = ('arxiv', 'pdf', 'not_found', 'failure')

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

        self.paper_access_resolutions = Counter(
            'paper_access_resolutions',
            'Number of citation signatures resolved by the paper access agent, by the result.',
            labelnames=['result'],
            namespace='paper_scraping_service',
        )

        self.paper_access_resolution_time = Summary(
            'paper_access_resolution_time',
            'Time spent by the paper access agent on resolving citation signatures.',
            namespace='paper_scraping_service',
        )

        for result in self._ACCESS_RESOLUTION_RESULTS:
            self.paper_access_resolutions.labels(result=result)

        for source in self._SOURCES:
            self.paper_parsing_time.labels(source=source)

            for status in ('success', 'failure'):
                self.papers_parsed.labels(source=source, status=status)
