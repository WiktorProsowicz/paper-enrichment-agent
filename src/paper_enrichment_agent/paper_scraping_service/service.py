"""Contains the core logic of the `paper_scraping_service` service.

The `paper_scraping_service` service downloads papers from external sources and converts them into
the structured :class:`~paper_enrichment_agent.common.models.document.Document` format, together
with their bibliographic metadata. It is called by the other services of the application.
"""

import asyncio
import time
from functools import cache

import httpx
from prometheus_client import Counter, Summary

from paper_enrichment_agent.common import logging_setup
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
    CrossrefFetcher,
)
from paper_enrichment_agent.paper_scraping_service.components.paper_access_agent import (
    PaperAccessAgent,
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
        crossref_fetcher: CrossrefFetcher,
        paper_access_agent: PaperAccessAgent,
        pdf_parser_config: PdfParser.Configuration,
    ) -> None:
        """Initializes the service.

        Args:
            metrics: The metrics of the service.
            arxiv_metadata_fetcher: The fetcher of the arXiv papers' metadata.
            crossref_fetcher: The fetcher of the papers' metadata by their DOIs.
            paper_access_agent: The agent finding public versions of papers by their citations.
            pdf_parser_config: The configuration of the PDF parsers, created per parsed paper.
        """

        self._metrics = metrics
        self._arxiv_metadata_fetcher = arxiv_metadata_fetcher
        self._crossref_fetcher = crossref_fetcher
        self._paper_access_agent = paper_access_agent
        self._pdf_parser_config = pdf_parser_config

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

    async def download_pdf_paper(
        self, pdf_content: bytes, file_name: str, doi: str
    ) -> tuple[doc_models.Document, misc_models.PaperMetadata]:
        """Parses a PDF paper supplied directly by the caller into the document structure.

        The paper access agent is not involved: the PDF file is validated and converted by a new
        PDF parser (in a separate thread, since the parser is synchronous), while the metadata of
        the paper is fetched by its DOI concurrently. The metadata of the papers with DOIs
        registered by arXiv (`10.48550/arXiv.<id>`), unknown to Crossref, is fetched from arXiv
        (in a separate thread), the metadata of the other papers from Crossref.

        Args:
            pdf_content: The contents of the PDF file of the paper.
            file_name: The name of the PDF file.
            doi: The DOI of the paper, optionally prefixed with `https://doi.org/`.

        Returns:
            The parsed document and the metadata of the paper.

        Raises:
            PaperScrapingError: If the file is not a valid PDF, the paper cannot be parsed, or its
                metadata cannot be fetched.
        """

        try:
            start = time.perf_counter()
            document, metadata = await asyncio.gather(
                asyncio.to_thread(self._parse_pdf_content, pdf_content, file_name),
                self._fetch_metadata_by_doi(doi),
            )
            end = time.perf_counter()

        except (
            PdfParser.ParsingError,
            CrossrefFetcher.FetchingError,
            ArxivMetadataFetcher.FetchingError,
        ) as e:
            _logger().error(
                'Failed to parse the uploaded PDF paper.',
                file_name=file_name,
                doi=doi,
                error=str(e),
            )
            self._metrics.papers_parsed.labels(source='uploaded_pdf', status='failure').inc()

            raise self.PaperScrapingError(
                f'Failed to parse the PDF paper {file_name} with DOI {doi}: {e}'
            ) from e

        _logger().info(
            'Successfully parsed the uploaded PDF paper.',
            file_name=file_name,
            doi=doi,
            title=document.title,
            metadata=metadata,
        )
        self._metrics.papers_parsed.labels(source='uploaded_pdf', status='success').inc()
        self._metrics.paper_parsing_time.labels(source='uploaded_pdf').observe(end - start)

        return document, metadata

    async def resolve_paper_access(
        self, citation_signature: str
    ) -> misc_models.PaperAccessInfo | None:
        """Finds the best public source of a cited paper and its bibliographic metadata.

        The paper access agent identifies the paper and picks its best public source: the HTML
        version on arXiv, or an open access PDF. The paper itself is not downloaded.

        Args:
            citation_signature: The citation signature of the paper, as found in the bibliography
                of the citing paper.

        Returns:
            The access information of the paper, or None if no public version of the paper was
            found.

        Raises:
            PaperScrapingError: If the access information of the paper cannot be resolved.
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

        return access_info

    async def parse_paper_from_citation_signature(
        self, citation_signature: str
    ) -> tuple[doc_models.Document, misc_models.PaperMetadata] | None:
        """Finds a public version of a cited paper and parses it into the document structure.

        The best public source of the paper is resolved with :meth:`resolve_paper_access`. The
        paper is then parsed by the respective parser in a separate thread, as the parsers are
        synchronous. A PDF paper is parsed by a new parser with its own HTTP client.

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

        access_info = await self.resolve_paper_access(citation_signature)

        if access_info is None:
            return None

        try:
            start = time.perf_counter()

            match access_info.type:
                case misc_models.PaperAccessType.ARXIV:
                    document = await asyncio.to_thread(ArxivParser().parse, access_info.source)
                case misc_models.PaperAccessType.PDF:
                    document = await asyncio.to_thread(self._parse_pdf, access_info.source)

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

    def _parse_pdf(self, pdf_url: str) -> doc_models.Document:
        """Parses a PDF paper with a new parser and a new HTTP client (blocking)."""

        with httpx.Client() as http_client:
            return PdfParser(self._pdf_parser_config, http_client).parse(pdf_url)

    async def _fetch_metadata_by_doi(self, doi: str) -> misc_models.PaperMetadata:
        """Fetches the metadata of a paper from arXiv for arXiv DOIs, from Crossref otherwise."""

        if ArxivMetadataFetcher.is_arxiv_doi(doi):
            return await asyncio.to_thread(
                self._arxiv_metadata_fetcher.fetch, ArxivMetadataFetcher.arxiv_id_from_doi(doi)
            )

        return await self._crossref_fetcher.fetch(doi)

    def _parse_pdf_content(self, pdf_content: bytes, file_name: str) -> doc_models.Document:
        """Parses a ready PDF file with a new parser and a new HTTP client (blocking)."""

        with httpx.Client() as http_client:
            return PdfParser(self._pdf_parser_config, http_client).parse_pdf_content(
                pdf_content, file_name
            )


class Metrics:
    """Metrics for the `paper_scraping_service` service.

    The metrics are registered in the default Prometheus registry, so a single `Metrics` instance
    should be created per process. They are exposed by the service's `/metrics` endpoint.
    """

    # Sources of the papers the service can scrape
    _SOURCES = ('arxiv', 'pdf', 'uploaded_pdf')

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
