from unittest.mock import AsyncMock, MagicMock, Mock

import fastmcp
import pytest

from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
    OpenAlexScraper,
    OpenAlexWork,
    UnpaywallFetcher,
    UnpaywallResult,
)
from paper_enrichment_agent.paper_scraping_service.components.paper_access_tools import (
    PaperAccessTools,
)

SAMPLE_WORK = OpenAlexWork(
    openalex_id='https://openalex.org/W1',
    title='Attention Is All You Need',
    doi='10.48550/arxiv.1706.03762',
    publication_date='2017-06-12',
    authors=['Ashish Vaswani'],
    venue='arXiv (Cornell University)',
    arxiv_id='1706.03762',
    relevance_score=300.3,
)

SAMPLE_UNPAYWALL_RESULT = UnpaywallResult(
    doi='10.1/abc', title='Sample paper', is_oa=False, best_pdf_url=None, pdf_locations=[]
)


@pytest.fixture
def mock_openalex_scraper() -> MagicMock:

    return MagicMock(
        spec=OpenAlexScraper, search_paper_by_title=AsyncMock(return_value=[SAMPLE_WORK])
    )


@pytest.fixture
def mock_arxiv_fetcher() -> MagicMock:

    return MagicMock(
        spec=ArxivMetadataFetcher,
        has_arxiv_html_version=Mock(return_value=True),
        is_arxiv_pdf_location_correct=Mock(return_value=False),
    )


@pytest.fixture
def mock_unpaywall_fetcher() -> MagicMock:

    return MagicMock(spec=UnpaywallFetcher, fetch=AsyncMock(return_value=SAMPLE_UNPAYWALL_RESULT))


@pytest.fixture
def tools(mock_openalex_scraper, mock_arxiv_fetcher, mock_unpaywall_fetcher) -> PaperAccessTools:

    return PaperAccessTools(
        openalex_scraper=mock_openalex_scraper,
        arxiv_fetcher=mock_arxiv_fetcher,
        unpaywall_fetcher=mock_unpaywall_fetcher,
    )


class TestScrapeMetadataFromOpenalex:
    @pytest.mark.asyncio
    async def test_delegates_to_scraper(self, tools, mock_openalex_scraper):

        assert await tools.scrape_metadata_from_openalex('Attention') == [SAMPLE_WORK]
        mock_openalex_scraper.search_paper_by_title.assert_awaited_once_with('Attention')

    @pytest.mark.asyncio
    async def test_reports_fetching_error_to_agent(self, tools, mock_openalex_scraper):

        mock_openalex_scraper.search_paper_by_title.side_effect = OpenAlexScraper.FetchingError(
            'OpenAlex responded with status 503.'
        )

        with pytest.raises(ValueError, match='status 503.*retry later'):
            await tools.scrape_metadata_from_openalex('Attention')


class TestHasArxivHtmlVersion:
    @pytest.mark.asyncio
    async def test_delegates_to_arxiv_fetcher(self, tools, mock_arxiv_fetcher):

        assert await tools.has_arxiv_html_version('1706.03762')
        mock_arxiv_fetcher.has_arxiv_html_version.assert_called_once_with('1706.03762')

    @pytest.mark.asyncio
    async def test_reports_fetching_error_to_agent(self, tools, mock_arxiv_fetcher):

        mock_arxiv_fetcher.has_arxiv_html_version.side_effect = ArxivMetadataFetcher.FetchingError(
            'The arXiv paper 1706.03762 does not exist.'
        )

        with pytest.raises(ValueError, match='does not exist.*Verify the arXiv ID'):
            await tools.has_arxiv_html_version('1706.03762')


class TestIsArxivPdfLocationCorrect:
    @pytest.mark.asyncio
    async def test_delegates_to_arxiv_fetcher(self, tools, mock_arxiv_fetcher):

        assert not await tools.is_arxiv_pdf_location_correct('https://arxiv.org/pdf/1706.03762')
        mock_arxiv_fetcher.is_arxiv_pdf_location_correct.assert_called_once_with(
            'https://arxiv.org/pdf/1706.03762'
        )

    @pytest.mark.asyncio
    async def test_reports_fetching_error_to_agent(self, tools, mock_arxiv_fetcher):

        mock_arxiv_fetcher.is_arxiv_pdf_location_correct.side_effect = (
            ArxivMetadataFetcher.FetchingError('"x" is not an arXiv PDF URL.')
        )

        with pytest.raises(ValueError, match='is not an arXiv PDF URL'):
            await tools.is_arxiv_pdf_location_correct('x')


class TestQueryUnpaywall:
    @pytest.mark.asyncio
    async def test_delegates_to_unpaywall_fetcher(self, tools, mock_unpaywall_fetcher):

        assert await tools.query_unpaywall('10.1/abc') == SAMPLE_UNPAYWALL_RESULT
        mock_unpaywall_fetcher.fetch.assert_awaited_once_with('10.1/abc')

    @pytest.mark.asyncio
    async def test_reports_fetching_error_to_agent(self, tools, mock_unpaywall_fetcher):

        mock_unpaywall_fetcher.fetch.side_effect = UnpaywallFetcher.FetchingError(
            'Unpaywall has no record for the request.'
        )

        with pytest.raises(ValueError, match='no record.*Verify the DOI'):
            await tools.query_unpaywall('10.1/abc')


class TestAsMcpServer:
    @pytest.mark.asyncio
    async def test_exposes_tools(self, tools, mock_arxiv_fetcher):

        async with fastmcp.Client(tools.as_mcp_server()) as client:
            mcp_tools = (await client.session.list_tools()).tools

            assert {tool.name for tool in mcp_tools} == {
                'scrape_metadata_from_openalex',
                'has_arxiv_html_version',
                'is_arxiv_pdf_location_correct',
                'query_unpaywall',
            }
            assert all(tool.description for tool in mcp_tools)

            result = await client.session.call_tool(
                'has_arxiv_html_version', {'arxiv_id': '1706.03762'}
            )
            assert not result.isError
            assert result.structuredContent == {'result': True}

            mock_arxiv_fetcher.has_arxiv_html_version.side_effect = (
                ArxivMetadataFetcher.FetchingError('Invalid ID.')
            )
            result = await client.session.call_tool('has_arxiv_html_version', {'arxiv_id': 'x'})
            assert result.isError
