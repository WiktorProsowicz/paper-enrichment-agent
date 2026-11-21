import datetime
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import prometheus_client
import pytest

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import (
    PaperAccessInfo,
    PaperAccessType,
    PaperMetadata,
)
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
from paper_enrichment_agent.paper_scraping_service import service as service_module
from paper_enrichment_agent.paper_scraping_service.service import PaperScrapingService

SAMPLE_ARXIV_ID = '1905.09263'
SAMPLE_PDF_URL = 'https://example.org/paper.pdf'
SAMPLE_DOI = '10.1145/3065386'
SAMPLE_PDF_CONTENT = b'%PDF-1.5 fake content'
SAMPLE_CITATION = 'A. Author and B. Author. A sample paper. In Proceedings, 2019.'
SAMPLE_PDF_PARSER_CONFIG = PdfParser.Configuration(
    marker_api_url='http://marker.test',
    marker_api_token='secret-token',
    marker_api_timeout=10.0,
    download_timeout=5.0,
    max_pdf_size_bytes=1024,
)


def create_service(
    metrics: MagicMock,
    fetcher: MagicMock,
    paper_access_agent: MagicMock | None = None,
    crossref_fetcher: MagicMock | None = None,
) -> PaperScrapingService:
    """Creates the service, mocking the dependencies not used by the test."""

    return PaperScrapingService(
        metrics=metrics,
        arxiv_metadata_fetcher=fetcher,
        crossref_fetcher=crossref_fetcher or MagicMock(spec=CrossrefFetcher),
        paper_access_agent=paper_access_agent or MagicMock(spec=PaperAccessAgent),
        pdf_parser_config=SAMPLE_PDF_PARSER_CONFIG,
    )


@pytest.fixture
def sample_document() -> doc_models.Document:

    return doc_models.Document(
        title='Sample Document',
        abstract='This is a sample abstract.',
        description=f'arXiv:{SAMPLE_ARXIV_ID}',
        sections=[
            doc_models.Section(
                title='Introduction',
                components=[doc_models.Paragraph(elements=['This is a sample paragraph.'])],
            )
        ],
        footnotes=[],
        referenced_papers=[('ref1', 'Reference Paper 1')],
    )


@pytest.fixture
def sample_metadata() -> PaperMetadata:

    return PaperMetadata(
        doi=f'10.48550/arXiv.{SAMPLE_ARXIV_ID}',
        authors=['Author One', 'Author Two'],
        publication_date=datetime.date(2019, 5, 22),
    )


@pytest.fixture
def mock_metrics() -> MagicMock:

    return MagicMock()


@pytest.fixture(autouse=True)
def patch_logger(monkeypatch) -> MagicMock:

    mock_logger = MagicMock()
    monkeypatch.setattr(
        'paper_enrichment_agent.paper_scraping_service.service._logger',
        Mock(return_value=mock_logger),
    )

    return mock_logger


@pytest.fixture
def patch_arxiv_parser(monkeypatch, sample_document) -> MagicMock:

    parser = MagicMock(parse=Mock(return_value=sample_document))

    monkeypatch.setattr(
        'paper_enrichment_agent.paper_scraping_service.service.ArxivParser',
        Mock(return_value=parser, ParsingError=ArxivParser.ParsingError),
    )

    return parser


@pytest.fixture
def mock_fetcher(sample_metadata) -> MagicMock:

    return MagicMock(spec=ArxivMetadataFetcher, fetch=Mock(return_value=sample_metadata))


class TestParseArxivPaper:
    @pytest.mark.asyncio
    async def test_returns_parsed_document_and_metadata(
        self, patch_arxiv_parser, mock_fetcher, mock_metrics, sample_document, sample_metadata
    ):

        service = create_service(mock_metrics, mock_fetcher)

        document, metadata = await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)

        assert document == sample_document
        assert metadata == sample_metadata
        patch_arxiv_parser.parse.assert_called_once_with(SAMPLE_ARXIV_ID)
        mock_fetcher.fetch.assert_called_once_with(SAMPLE_ARXIV_ID)

    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(self, patch_arxiv_parser, mock_fetcher, mock_metrics):

        service = create_service(mock_metrics, mock_fetcher)

        await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='arxiv', status='success')
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()

        mock_metrics.paper_parsing_time.labels.assert_called_once_with(source='arxiv')
        observe = mock_metrics.paper_parsing_time.labels.return_value.observe
        assert observe.call_count == 1
        assert observe.call_args.args[0] >= 0

    @pytest.mark.asyncio
    async def test_raises_when_parsing_fails(self, patch_arxiv_parser, mock_fetcher, mock_metrics):

        patch_arxiv_parser.parse.side_effect = ArxivParser.ParsingError('Parsing failed')

        service = create_service(mock_metrics, mock_fetcher)

        with pytest.raises(
            PaperScrapingService.PaperScrapingError,
            match=f'Failed to parse the arXiv paper with ID {SAMPLE_ARXIV_ID}: Parsing failed',
        ):
            await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)

    @pytest.mark.asyncio
    async def test_raises_when_metadata_fetching_fails(
        self, patch_arxiv_parser, mock_fetcher, mock_metrics
    ):

        mock_fetcher.fetch.side_effect = ArxivMetadataFetcher.FetchingError('Fetching failed')

        service = create_service(mock_metrics, mock_fetcher)

        with pytest.raises(
            PaperScrapingService.PaperScrapingError,
            match=f'Failed to parse the arXiv paper with ID {SAMPLE_ARXIV_ID}: Fetching failed',
        ):
            await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)

    @pytest.mark.asyncio
    async def test_reports_metrics_on_failure(self, patch_arxiv_parser, mock_fetcher, mock_metrics):

        mock_fetcher.fetch.side_effect = ArxivMetadataFetcher.FetchingError('Fetching failed')

        service = create_service(mock_metrics, mock_fetcher)

        with pytest.raises(PaperScrapingService.PaperScrapingError):
            await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='arxiv', status='failure')
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()
        assert mock_metrics.paper_parsing_time.labels.return_value.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(
        self, patch_arxiv_parser, mock_fetcher, mock_metrics
    ):

        mock_fetcher.fetch.side_effect = RuntimeError('Unexpected failure')

        service = create_service(mock_metrics, mock_fetcher)

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)


class TestDownloadPdfPaper:
    @pytest.fixture
    def mock_crossref_fetcher(self, sample_metadata) -> MagicMock:

        return MagicMock(spec=CrossrefFetcher, fetch=AsyncMock(return_value=sample_metadata))

    @pytest.fixture
    def mock_pdf_parser_cls(self, monkeypatch, sample_document) -> Mock:

        parser_cls = Mock(
            return_value=MagicMock(
                spec=PdfParser, parse_pdf_content=Mock(return_value=sample_document)
            ),
            ParsingError=PdfParser.ParsingError,
        )
        monkeypatch.setattr(
            'paper_enrichment_agent.paper_scraping_service.service.PdfParser', parser_cls
        )

        return parser_cls

    @pytest.fixture
    def mock_pdf_parser(self, mock_pdf_parser_cls) -> MagicMock:

        return mock_pdf_parser_cls.return_value

    @pytest.fixture
    def service(
        self, mock_metrics, mock_fetcher, mock_crossref_fetcher, mock_pdf_parser
    ) -> PaperScrapingService:

        return create_service(mock_metrics, mock_fetcher, crossref_fetcher=mock_crossref_fetcher)

    @pytest.mark.asyncio
    async def test_returns_parsed_document_and_crossref_metadata(
        self,
        service,
        mock_pdf_parser_cls,
        mock_pdf_parser,
        mock_crossref_fetcher,
        sample_document,
        sample_metadata,
    ):

        document, metadata = await service.download_pdf_paper(
            pdf_content=SAMPLE_PDF_CONTENT, file_name='paper.pdf', doi=SAMPLE_DOI
        )

        assert document == sample_document
        assert metadata == sample_metadata
        assert mock_pdf_parser_cls.call_args.args[0] == SAMPLE_PDF_PARSER_CONFIG
        assert isinstance(mock_pdf_parser_cls.call_args.args[1], httpx.Client)
        mock_pdf_parser.parse_pdf_content.assert_called_once_with(SAMPLE_PDF_CONTENT, 'paper.pdf')
        mock_crossref_fetcher.fetch.assert_awaited_once_with(SAMPLE_DOI)

    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(self, service, mock_metrics):

        await service.download_pdf_paper(
            pdf_content=SAMPLE_PDF_CONTENT, file_name='paper.pdf', doi=SAMPLE_DOI
        )

        mock_metrics.papers_parsed.labels.assert_called_once_with(
            source='uploaded_pdf', status='success'
        )
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()
        mock_metrics.paper_parsing_time.labels.assert_called_once_with(source='uploaded_pdf')
        assert mock_metrics.paper_parsing_time.labels.return_value.observe.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_when_pdf_is_invalid(self, service, mock_pdf_parser, mock_metrics):

        mock_pdf_parser.parse_pdf_content.side_effect = PdfParser.ParsingError('not a PDF')

        with pytest.raises(
            PaperScrapingService.PaperScrapingError,
            match=f'Failed to parse the PDF paper paper.pdf with DOI {SAMPLE_DOI}: not a PDF',
        ):
            await service.download_pdf_paper(
                pdf_content=b'<html>', file_name='paper.pdf', doi=SAMPLE_DOI
            )

        mock_metrics.papers_parsed.labels.assert_called_once_with(
            source='uploaded_pdf', status='failure'
        )
        assert mock_metrics.paper_parsing_time.labels.return_value.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_raises_when_metadata_fetching_fails(
        self, service, mock_crossref_fetcher, mock_metrics
    ):

        mock_crossref_fetcher.fetch.side_effect = CrossrefFetcher.FetchingError('Unknown DOI')

        with pytest.raises(PaperScrapingService.PaperScrapingError, match='Unknown DOI'):
            await service.download_pdf_paper(
                pdf_content=SAMPLE_PDF_CONTENT, file_name='paper.pdf', doi=SAMPLE_DOI
            )

        mock_metrics.papers_parsed.labels.assert_called_once_with(
            source='uploaded_pdf', status='failure'
        )

    @pytest.mark.asyncio
    async def test_fetches_metadata_of_arxiv_doi_from_arxiv(
        self, service, mock_fetcher, mock_crossref_fetcher, sample_metadata
    ):

        _, metadata = await service.download_pdf_paper(
            pdf_content=SAMPLE_PDF_CONTENT,
            file_name='paper.pdf',
            doi=f'10.48550/arxiv.{SAMPLE_ARXIV_ID}',
        )

        assert metadata == sample_metadata
        mock_fetcher.fetch.assert_called_once_with(SAMPLE_ARXIV_ID)
        mock_crossref_fetcher.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_arxiv_metadata_fetching_fails(
        self, service, mock_fetcher, mock_metrics
    ):

        mock_fetcher.fetch.side_effect = ArxivMetadataFetcher.FetchingError('arXiv down')

        with pytest.raises(PaperScrapingService.PaperScrapingError, match='arXiv down'):
            await service.download_pdf_paper(
                pdf_content=SAMPLE_PDF_CONTENT,
                file_name='paper.pdf',
                doi=f'10.48550/arXiv.{SAMPLE_ARXIV_ID}',
            )

        mock_metrics.papers_parsed.labels.assert_called_once_with(
            source='uploaded_pdf', status='failure'
        )

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(self, service, mock_pdf_parser):

        mock_pdf_parser.parse_pdf_content.side_effect = RuntimeError('Unexpected failure')

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.download_pdf_paper(
                pdf_content=SAMPLE_PDF_CONTENT, file_name='paper.pdf', doi=SAMPLE_DOI
            )


class TestResolvePaperAccess:
    @pytest.fixture
    def access_info(self, sample_metadata) -> PaperAccessInfo:

        return PaperAccessInfo(
            type=PaperAccessType.ARXIV, source=SAMPLE_ARXIV_ID, metadata=sample_metadata
        )

    @pytest.fixture
    def mock_access_agent(self, access_info) -> MagicMock:

        return MagicMock(spec=PaperAccessAgent, invoke=AsyncMock(return_value=access_info))

    @pytest.fixture
    def service(self, mock_metrics, mock_fetcher, mock_access_agent) -> PaperScrapingService:

        return create_service(mock_metrics, mock_fetcher, mock_access_agent)

    @pytest.mark.asyncio
    async def test_returns_access_info(
        self, service, patch_arxiv_parser, mock_access_agent, mock_fetcher, access_info
    ):

        assert await service.resolve_paper_access(SAMPLE_CITATION) == access_info

        mock_access_agent.invoke.assert_awaited_once_with(SAMPLE_CITATION)
        patch_arxiv_parser.parse.assert_not_called()
        mock_fetcher.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_public_version_is_found(
        self, service, mock_access_agent, mock_metrics
    ):

        mock_access_agent.invoke.return_value = None

        assert await service.resolve_paper_access(SAMPLE_CITATION) is None

        mock_metrics.paper_access_resolutions.labels.assert_called_once_with(result='not_found')
        mock_metrics.paper_access_resolutions.labels.return_value.inc.assert_called_once()

    @pytest.mark.parametrize('access_type', list(PaperAccessType))
    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(
        self, service, mock_access_agent, mock_metrics, access_info, access_type
    ):

        mock_access_agent.invoke.return_value = access_info.model_copy(update={'type': access_type})

        await service.resolve_paper_access(SAMPLE_CITATION)

        mock_metrics.paper_access_resolutions.labels.assert_called_once_with(
            result=access_type.value
        )
        mock_metrics.paper_access_resolutions.labels.return_value.inc.assert_called_once()
        observe = mock_metrics.paper_access_resolution_time.observe
        assert observe.call_count == 1
        assert observe.call_args.args[0] >= 0
        mock_metrics.papers_parsed.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_agent_fails(self, service, mock_access_agent, mock_metrics):

        mock_access_agent.invoke.side_effect = PaperAccessAgent.PaperAccessAgentError(
            'Recursion limit'
        )

        with pytest.raises(
            PaperScrapingService.PaperScrapingError,
            match=f'Failed to find the paper cited as "{SAMPLE_CITATION}": Recursion limit',
        ):
            await service.resolve_paper_access(SAMPLE_CITATION)

        mock_metrics.paper_access_resolutions.labels.assert_called_once_with(result='failure')
        mock_metrics.paper_access_resolutions.labels.return_value.inc.assert_called_once()
        mock_metrics.paper_access_resolution_time.observe.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(self, service, mock_access_agent):

        mock_access_agent.invoke.side_effect = RuntimeError('Unexpected failure')

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.resolve_paper_access(SAMPLE_CITATION)


class TestParsePaperFromCitationSignature:
    @pytest.fixture
    def mock_access_agent(self, sample_metadata) -> MagicMock:

        return MagicMock(
            spec=PaperAccessAgent,
            invoke=AsyncMock(
                return_value=PaperAccessInfo(
                    type=PaperAccessType.ARXIV, source=SAMPLE_ARXIV_ID, metadata=sample_metadata
                )
            ),
        )

    @pytest.fixture
    def mock_pdf_parser_cls(self, monkeypatch, sample_document) -> Mock:

        parser_cls = Mock(
            return_value=MagicMock(spec=PdfParser, parse=Mock(return_value=sample_document)),
            ParsingError=PdfParser.ParsingError,
        )
        monkeypatch.setattr(
            'paper_enrichment_agent.paper_scraping_service.service.PdfParser', parser_cls
        )

        return parser_cls

    @pytest.fixture
    def mock_pdf_parser(self, mock_pdf_parser_cls) -> MagicMock:

        return mock_pdf_parser_cls.return_value

    @pytest.fixture
    def service(
        self, mock_metrics, mock_fetcher, mock_access_agent, mock_pdf_parser
    ) -> PaperScrapingService:

        return create_service(mock_metrics, mock_fetcher, mock_access_agent)

    @pytest.fixture
    def pdf_access(self, mock_access_agent, sample_metadata) -> None:

        mock_access_agent.invoke.return_value = PaperAccessInfo(
            type=PaperAccessType.PDF, source=SAMPLE_PDF_URL, metadata=sample_metadata
        )

    @pytest.mark.asyncio
    async def test_parses_arxiv_paper(
        self,
        service,
        patch_arxiv_parser,
        mock_access_agent,
        mock_pdf_parser,
        mock_fetcher,
        sample_document,
        sample_metadata,
    ):

        result = await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        assert result == (sample_document, sample_metadata)
        mock_access_agent.invoke.assert_awaited_once_with(SAMPLE_CITATION)
        patch_arxiv_parser.parse.assert_called_once_with(SAMPLE_ARXIV_ID)
        mock_pdf_parser.parse.assert_not_called()
        mock_fetcher.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_parses_pdf_paper(
        self,
        service,
        pdf_access,
        patch_arxiv_parser,
        mock_pdf_parser,
        sample_document,
        sample_metadata,
    ):

        result = await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        assert result == (sample_document, sample_metadata)
        mock_pdf_parser.parse.assert_called_once_with(SAMPLE_PDF_URL)
        patch_arxiv_parser.parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_pdf_parser_with_fresh_http_client_per_paper(
        self, service, pdf_access, mock_pdf_parser_cls, mock_pdf_parser
    ):

        clients_during_parsing: list[httpx.Client] = []

        def record_client(_):
            http_client = mock_pdf_parser_cls.call_args.args[1]
            assert not http_client.is_closed
            clients_during_parsing.append(http_client)
            return mock_pdf_parser.parse.return_value

        mock_pdf_parser.parse.side_effect = record_client

        await service.parse_paper_from_citation_signature(SAMPLE_CITATION)
        await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        assert mock_pdf_parser_cls.call_count == 2
        for call in mock_pdf_parser_cls.call_args_list:
            config, http_client = call.args
            assert config == SAMPLE_PDF_PARSER_CONFIG
            assert isinstance(http_client, httpx.Client)

        first_client, second_client = clients_during_parsing
        assert first_client is not second_client
        assert first_client.is_closed
        assert second_client.is_closed

    @pytest.mark.asyncio
    async def test_closes_http_client_when_pdf_parsing_fails(
        self, service, pdf_access, mock_pdf_parser_cls, mock_pdf_parser
    ):

        mock_pdf_parser.parse.side_effect = PdfParser.ParsingError('Conversion failed')

        with pytest.raises(PaperScrapingService.PaperScrapingError):
            await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        assert mock_pdf_parser_cls.call_args.args[1].is_closed

    @pytest.mark.asyncio
    async def test_returns_none_when_no_public_version_is_found(
        self, service, patch_arxiv_parser, mock_access_agent, mock_metrics
    ):

        mock_access_agent.invoke.return_value = None

        assert await service.parse_paper_from_citation_signature(SAMPLE_CITATION) is None

        patch_arxiv_parser.parse.assert_not_called()
        mock_metrics.paper_access_resolutions.labels.assert_called_once_with(result='not_found')
        mock_metrics.paper_access_resolutions.labels.return_value.inc.assert_called_once()
        mock_metrics.papers_parsed.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(
        self, service, pdf_access, mock_pdf_parser, mock_metrics
    ):

        await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='pdf', status='success')
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()
        mock_metrics.paper_parsing_time.labels.assert_called_once_with(source='pdf')
        observe = mock_metrics.paper_parsing_time.labels.return_value.observe
        assert observe.call_count == 1
        assert observe.call_args.args[0] >= 0

    @pytest.mark.asyncio
    async def test_raises_when_agent_fails(self, service, mock_access_agent, mock_metrics):

        mock_access_agent.invoke.side_effect = PaperAccessAgent.PaperAccessAgentError(
            'Recursion limit'
        )

        with pytest.raises(
            PaperScrapingService.PaperScrapingError,
            match=f'Failed to find the paper cited as "{SAMPLE_CITATION}": Recursion limit',
        ):
            await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        mock_metrics.paper_access_resolutions.labels.assert_called_once_with(result='failure')
        mock_metrics.paper_access_resolutions.labels.return_value.inc.assert_called_once()
        mock_metrics.papers_parsed.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_arxiv_parsing_fails(self, service, patch_arxiv_parser, mock_metrics):

        patch_arxiv_parser.parse.side_effect = ArxivParser.ParsingError('Parsing failed')

        with pytest.raises(PaperScrapingService.PaperScrapingError, match='Parsing failed'):
            await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='arxiv', status='failure')
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_pdf_parsing_fails(
        self, service, pdf_access, mock_pdf_parser, mock_metrics
    ):

        mock_pdf_parser.parse.side_effect = PdfParser.ParsingError('Conversion failed')

        with pytest.raises(PaperScrapingService.PaperScrapingError, match='Conversion failed'):
            await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='pdf', status='failure')
        assert mock_metrics.paper_parsing_time.labels.return_value.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(self, service, mock_access_agent):

        mock_access_agent.invoke.side_effect = RuntimeError('Unexpected failure')

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.parse_paper_from_citation_signature(SAMPLE_CITATION)

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_parser_errors(
        self, service, pdf_access, mock_pdf_parser
    ):

        mock_pdf_parser.parse.side_effect = RuntimeError('Unexpected failure')

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.parse_paper_from_citation_signature(SAMPLE_CITATION)


class TestMetrics:
    def test_preinitializes_label_combinations(self, monkeypatch):

        # A separate registry avoids duplicated registrations in the default one.
        registry = prometheus_client.CollectorRegistry()
        for name in ('Counter', 'Summary'):
            metric_cls = getattr(prometheus_client, name)
            monkeypatch.setattr(
                service_module,
                name,
                lambda *args, _cls=metric_cls, **kwargs: _cls(*args, registry=registry, **kwargs),
            )

        service_module.Metrics()

        for status in ('success', 'failure'):
            assert (
                registry.get_sample_value(
                    'paper_scraping_service_papers_parsed_total',
                    {'source': 'arxiv', 'status': status},
                )
                == 0
            )
        assert (
            registry.get_sample_value(
                'paper_scraping_service_paper_parsing_time_count', {'source': 'arxiv'}
            )
            == 0
        )

        for source in ('arxiv', 'pdf', 'uploaded_pdf'):
            assert (
                registry.get_sample_value(
                    'paper_scraping_service_papers_parsed_total',
                    {'source': source, 'status': 'failure'},
                )
                == 0
            )

        for result in ('arxiv', 'pdf', 'not_found', 'failure'):
            assert (
                registry.get_sample_value(
                    'paper_scraping_service_paper_access_resolutions_total', {'result': result}
                )
                == 0
            )
