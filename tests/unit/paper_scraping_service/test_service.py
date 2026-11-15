import datetime
from unittest.mock import MagicMock, Mock

import prometheus_client
import pytest

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import PaperMetadata
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
)
from paper_enrichment_agent.paper_scraping_service.components.parsers.arxiv_parser import (
    ArxivParser,
)
from paper_enrichment_agent.paper_scraping_service import service as service_module
from paper_enrichment_agent.paper_scraping_service.service import PaperScrapingService

SAMPLE_ARXIV_ID = '1905.09263'


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

        service = PaperScrapingService(metrics=mock_metrics, arxiv_metadata_fetcher=mock_fetcher)

        document, metadata = await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)

        assert document == sample_document
        assert metadata == sample_metadata
        patch_arxiv_parser.parse.assert_called_once_with(SAMPLE_ARXIV_ID)
        mock_fetcher.fetch.assert_called_once_with(SAMPLE_ARXIV_ID)

    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(self, patch_arxiv_parser, mock_fetcher, mock_metrics):

        service = PaperScrapingService(metrics=mock_metrics, arxiv_metadata_fetcher=mock_fetcher)

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

        service = PaperScrapingService(metrics=mock_metrics, arxiv_metadata_fetcher=mock_fetcher)

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

        service = PaperScrapingService(metrics=mock_metrics, arxiv_metadata_fetcher=mock_fetcher)

        with pytest.raises(
            PaperScrapingService.PaperScrapingError,
            match=f'Failed to parse the arXiv paper with ID {SAMPLE_ARXIV_ID}: Fetching failed',
        ):
            await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)

    @pytest.mark.asyncio
    async def test_reports_metrics_on_failure(self, patch_arxiv_parser, mock_fetcher, mock_metrics):

        mock_fetcher.fetch.side_effect = ArxivMetadataFetcher.FetchingError('Fetching failed')

        service = PaperScrapingService(metrics=mock_metrics, arxiv_metadata_fetcher=mock_fetcher)

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

        service = PaperScrapingService(metrics=mock_metrics, arxiv_metadata_fetcher=mock_fetcher)

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.parse_arxiv_paper(SAMPLE_ARXIV_ID)


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
