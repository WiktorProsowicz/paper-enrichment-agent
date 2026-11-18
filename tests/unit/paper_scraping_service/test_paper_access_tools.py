from collections.abc import Callable

import fastmcp
import httpx
import pydantic
import pytest

from paper_enrichment_agent.paper_scraping_service.components.paper_access_tools import (
    PaperAccessTools,
    create_mcp_server,
)

type Handler = Callable[[httpx.Request], httpx.Response]

OPENALEX_URL = 'https://openalex.test/works'
UNPAYWALL_URL = 'https://unpaywall.test/v2'
AR5IV_URL = 'https://ar5iv.test/html'

SAMPLE_OPENALEX_WORKS = [
    {
        'id': 'https://openalex.org/W1',
        'doi': 'https://doi.org/10.48550/arxiv.1706.03762',
        'title': 'Attention Is All You Need',
        'publication_date': '2017-06-12',
        'authorships': [
            {'author': {'display_name': 'Ashish Vaswani'}},
            {'author': {'display_name': 'Noam Shazeer'}},
            {'author': {}},
        ],
        'primary_location': {'source': {'display_name': 'arXiv (Cornell University)'}},
        'locations': [],
        'open_access': {'is_oa': True, 'oa_url': 'https://arxiv.org/pdf/1706.03762'},
        'relevance_score': 300.3,
    },
    {
        'id': 'https://openalex.org/W2',
        'doi': 'https://doi.org/10.1109/icassp.2021.1',
        'title': 'Attention Is All You Need In Speech Separation',
        'publication_date': '2021-05-13',
        'authorships': [{'author': {'display_name': 'Cem Subakan'}}],
        'primary_location': {'source': None},
        'locations': [
            {'landing_page_url': 'https://doi.org/10.1109/icassp.2021.1', 'pdf_url': None},
            {'landing_page_url': None, 'pdf_url': 'https://arxiv.org/pdf/2010.13154v2'},
        ],
        'open_access': {'is_oa': True, 'oa_url': None},
        'relevance_score': 261.0,
    },
    {
        'id': 'https://openalex.org/W3',
        'doi': None,
        'title': 'Closed paper',
        'publication_date': None,
        'authorships': [],
        'primary_location': None,
        'locations': None,
        'open_access': None,
    },
]


@pytest.fixture
def config() -> PaperAccessTools.Configuration:

    return PaperAccessTools.Configuration(
        openalex_api_url=OPENALEX_URL,
        openalex_max_results=2,
        unpaywall_api_url=UNPAYWALL_URL,
        unpaywall_email='tools@example.org',
        ar5iv_html_url_base=AR5IV_URL,
    )


def create_tools(config: PaperAccessTools.Configuration, handler: Handler) -> PaperAccessTools:
    """Creates the tools with an HTTP client served by the given handler."""

    return PaperAccessTools(config, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class TestScrapeMetadataFromOpenalex:
    @pytest.mark.asyncio
    async def test_returns_preprocessed_works(self, config):

        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={'results': SAMPLE_OPENALEX_WORKS})

        works = await create_tools(config, handler).scrape_metadata_from_openalex(
            ' Attention is all you need '
        )

        assert len(works) == 2

        assert works[0].doi == '10.48550/arxiv.1706.03762'
        assert works[0].arxiv_id == '1706.03762'
        assert works[0].authors == ['Ashish Vaswani', 'Noam Shazeer']
        assert works[0].venue == 'arXiv (Cornell University)'
        assert works[0].is_oa
        assert works[0].oa_url == 'https://arxiv.org/pdf/1706.03762'
        assert works[0].publication_date == '2017-06-12'

        assert works[1].doi == '10.1109/icassp.2021.1'
        assert works[1].arxiv_id == '2010.13154'
        assert works[1].venue is None

        assert len(requests) == 1
        params = requests[0].url.params
        assert params['search'] == 'Attention is all you need'
        assert params['per_page'] == '2'
        assert 'api_key' not in params

    @pytest.mark.asyncio
    async def test_handles_missing_fields(self, config):

        config.openalex_max_results = 3

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'results': SAMPLE_OPENALEX_WORKS})

        works = await create_tools(config, handler).scrape_metadata_from_openalex('Closed paper')

        assert works[2].doi is None
        assert works[2].arxiv_id is None
        assert works[2].authors == []
        assert not works[2].is_oa
        assert works[2].relevance_score is None

    @pytest.mark.asyncio
    async def test_passes_api_key_if_configured(self, config):

        config = config.model_copy(update={'openalex_api_key': pydantic.SecretStr('secret-key')})
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={'results': []})

        assert await create_tools(config, handler).scrape_metadata_from_openalex('Title') == []
        assert requests[0].url.params['api_key'] == 'secret-key'

    @pytest.mark.asyncio
    async def test_raises_on_empty_title(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError('No request expected.')

        with pytest.raises(ValueError, match='must not be empty'):
            await create_tools(config, handler).scrape_metadata_from_openalex('  ')

    @pytest.mark.asyncio
    async def test_raises_on_error_status(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text='Service unavailable')

        with pytest.raises(ValueError, match='OpenAlex responded with status 503'):
            await create_tools(config, handler).scrape_metadata_from_openalex('Title')

    @pytest.mark.asyncio
    async def test_raises_on_connection_error(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('Connection refused')

        with pytest.raises(ValueError, match='Failed to reach OpenAlex'):
            await create_tools(config, handler).scrape_metadata_from_openalex('Title')

    @pytest.mark.parametrize('content', [b'not json', b'[1, 2]'])
    @pytest.mark.asyncio
    async def test_raises_on_invalid_response(self, config, content):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=content)

        with pytest.raises(ValueError, match='OpenAlex returned'):
            await create_tools(config, handler).scrape_metadata_from_openalex('Title')


class TestHasArxivHtmlVersion:
    @pytest.mark.parametrize('arxiv_id', ['1706.03762', 'arXiv:1706.03762v5', 'hep-th/9901001'])
    @pytest.mark.asyncio
    async def test_returns_true_for_article_page(self, config, arxiv_id):

        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={'content-type': 'text/html; charset=utf-8'},
                text='<html><div class="ltx_page_main"><article class="ltx_document">',
            )

        assert await create_tools(config, handler).has_arxiv_html_version(arxiv_id)
        assert str(requests[0].url) == f'{AR5IV_URL}/{arxiv_id.removeprefix("arXiv:")}'

    @pytest.mark.asyncio
    async def test_returns_false_on_redirect_to_abstract_page(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={'location': 'https://arxiv.org/abs/1706.03762'})

        assert not await create_tools(config, handler).has_arxiv_html_version('1706.03762')

    @pytest.mark.asyncio
    async def test_returns_false_for_non_article_page(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={'content-type': 'text/html'}, text='<html>Conversion failed</html>'
            )

        assert not await create_tools(config, handler).has_arxiv_html_version('1706.03762')

    @pytest.mark.parametrize('arxiv_id', ['', 'not-an-id', '1706.03762/abs', '17060.3762'])
    @pytest.mark.asyncio
    async def test_raises_on_invalid_id(self, config, arxiv_id):

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError('No request expected.')

        with pytest.raises(ValueError, match='is not a valid arXiv ID'):
            await create_tools(config, handler).has_arxiv_html_version(arxiv_id)

    @pytest.mark.parametrize(
        'response, error',
        [
            (httpx.Response(302, headers={'location': 'https://other.org'}), 'unexpected location'),
            (httpx.Response(404), 'does not exist'),
            (httpx.Response(500), 'status 500'),
        ],
    )
    @pytest.mark.asyncio
    async def test_raises_on_unexpected_response(self, config, response, error):

        def handler(request: httpx.Request) -> httpx.Response:
            return response

        with pytest.raises(ValueError, match=error):
            await create_tools(config, handler).has_arxiv_html_version('1706.03762')

    @pytest.mark.asyncio
    async def test_raises_on_connection_error(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout('Timed out')

        with pytest.raises(ValueError, match='Failed to reach ar5iv'):
            await create_tools(config, handler).has_arxiv_html_version('1706.03762')


class TestQueryUnpaywall:
    @pytest.mark.asyncio
    async def test_returns_pdf_locations(self, config):

        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    'title': 'Sample paper',
                    'is_oa': True,
                    'best_oa_location': {'url_for_pdf': None},
                    'oa_locations': [
                        {'url_for_pdf': None, 'host_type': 'publisher'},
                        {
                            'url_for_pdf': 'https://repo.org/paper.pdf',
                            'host_type': 'repository',
                            'version': 'submittedVersion',
                            'license': 'cc-by',
                        },
                    ],
                },
            )

        result = await create_tools(config, handler).query_unpaywall('https://doi.org/10.1/abc')

        assert result.doi == '10.1/abc'
        assert result.title == 'Sample paper'
        assert result.is_oa
        assert result.best_pdf_url is None
        assert [location.url_for_pdf for location in result.pdf_locations] == [
            'https://repo.org/paper.pdf'
        ]
        assert result.pdf_locations[0].license == 'cc-by'

        assert requests[0].url.path == '/v2/10.1/abc'
        assert requests[0].url.params['email'] == 'tools@example.org'

    @pytest.mark.asyncio
    async def test_returns_best_pdf_url(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            location = {'url_for_pdf': 'https://pub.org/paper.pdf', 'host_type': 'publisher'}
            return httpx.Response(
                200, json={'is_oa': True, 'best_oa_location': location, 'oa_locations': [location]}
            )

        result = await create_tools(config, handler).query_unpaywall('10.1/abc')

        assert result.best_pdf_url == 'https://pub.org/paper.pdf'

    @pytest.mark.asyncio
    async def test_reports_closed_paper(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={'is_oa': False, 'best_oa_location': None, 'oa_locations': []}
            )

        result = await create_tools(config, handler).query_unpaywall('10.1/abc')

        assert not result.is_oa
        assert result.best_pdf_url is None
        assert result.pdf_locations == []

    @pytest.mark.asyncio
    async def test_raises_on_unknown_doi(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        with pytest.raises(ValueError, match='Unpaywall has no record'):
            await create_tools(config, handler).query_unpaywall('10.1/unknown')

    @pytest.mark.asyncio
    async def test_raises_on_invalid_doi(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError('No request expected.')

        with pytest.raises(ValueError, match='is not a valid DOI'):
            await create_tools(config, handler).query_unpaywall('arXiv:1706.03762')


class TestCreateMcpServer:
    @pytest.mark.asyncio
    async def test_exposes_tools(self, config):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={'location': 'https://arxiv.org/abs/1706.03762'})

        mcp_server = create_mcp_server(create_tools(config, handler))

        async with fastmcp.Client(mcp_server) as client:
            tools = (await client.session.list_tools()).tools

            assert {tool.name for tool in tools} == {
                'scrape_metadata_from_openalex',
                'has_arxiv_html_version',
                'query_unpaywall',
            }
            assert all(tool.description for tool in tools)

            result = await client.session.call_tool(
                'has_arxiv_html_version', {'arxiv_id': '1706.03762'}
            )
            assert not result.isError
            assert result.structuredContent == {'result': False}

            result = await client.session.call_tool('has_arxiv_html_version', {'arxiv_id': 'x'})
            assert result.isError
