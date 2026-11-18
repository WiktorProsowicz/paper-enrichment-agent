import datetime
from collections.abc import Callable
from unittest.mock import Mock

import httpx
import pytest
import requests

from paper_enrichment_agent.common.models.misc import PaperMetadata
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
    OpenAlexScraper,
    UnpaywallFetcher,
)

type Handler = Callable[[httpx.Request], httpx.Response]

API_URL = 'https://export.arxiv.org/api/query'
AR5IV_URL = 'https://ar5iv.test/html'
OPENALEX_URL = 'https://openalex.test/works'
UNPAYWALL_URL = 'https://unpaywall.test/v2'
TIMEOUT = 12.5

SAMPLE_ARXIV_ID = '1706.03762'

FEED_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query: id_list=1706.03762</title>
  <opensearch:totalResults>1</opensearch:totalResults>
  {entry}
</feed>
"""

ENTRY_TEMPLATE = """<entry>
    <id>{entry_id}</id>
    <updated>2023-08-02T00:41:18Z</updated>
    <published>{published}</published>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name> Noam Shazeer </name></author>
    <author><name></name></author>
    <author><name>Niki Parmar</name></author>
    {doi}
  </entry>"""


def make_feed(
    entry_id: str = 'http://arxiv.org/abs/1706.03762v7',
    published: str = '2017-06-12T17:57:34Z',
    doi: str | None = None,
) -> bytes:

    doi_tag = f'<arxiv:doi>{doi}</arxiv:doi>' if doi is not None else ''
    entry = ENTRY_TEMPLATE.format(entry_id=entry_id, published=published, doi=doi_tag)

    return FEED_TEMPLATE.format(entry=entry).encode()


def create_arxiv_fetcher() -> ArxivMetadataFetcher:
    """Creates the arXiv fetcher with the test URLs."""

    return ArxivMetadataFetcher(api_url=API_URL, ar5iv_html_url_base=AR5IV_URL, timeout=TIMEOUT)


def make_response(
    status: int, headers: dict[str, str] | None = None, text: str = ''
) -> requests.Response:
    """Creates a `requests` response with the given status, headers and body."""

    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    response._content = text.encode()
    response.encoding = 'utf-8'

    return response


def make_async_client(handler: Handler) -> httpx.AsyncClient:
    """Creates an HTTP client served by the given handler."""

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def mock_requests_get(monkeypatch) -> Mock:

    get = Mock(return_value=Mock(content=make_feed(), raise_for_status=Mock()))

    monkeypatch.setattr(
        'paper_enrichment_agent.paper_scraping_service.components.fetchers.requests.get', get
    )

    return get


class TestArxivMetadataFetcher:
    def test_queries_api_with_configured_url_and_timeout(self, mock_requests_get):

        create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)

        mock_requests_get.assert_called_once_with(
            API_URL, params={'id_list': SAMPLE_ARXIV_ID}, timeout=TIMEOUT
        )

    def test_reads_authors_and_publication_date(self, mock_requests_get):

        metadata = create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)

        assert metadata == PaperMetadata(
            doi='10.48550/arXiv.1706.03762',
            authors=['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
            publication_date=datetime.date(2017, 6, 12),
        )

    def test_prefers_doi_declared_in_feed(self, mock_requests_get):

        mock_requests_get.return_value.content = make_feed(doi=' 10.1000/journal.123 ')

        metadata = create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)

        assert metadata.doi == '10.1000/journal.123'

    def test_strips_version_from_arxiv_doi(self, mock_requests_get):

        metadata = create_arxiv_fetcher().fetch('1706.03762v5')

        assert metadata.doi == '10.48550/arXiv.1706.03762'

    def test_raises_when_request_fails(self, mock_requests_get):

        mock_requests_get.side_effect = requests.ConnectionError('Connection refused')

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to download'):
            create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)

    def test_raises_on_http_error_status(self, mock_requests_get):

        mock_requests_get.return_value.raise_for_status.side_effect = requests.HTTPError(
            '503 Server Error'
        )

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to download'):
            create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)

    def test_raises_on_malformed_response(self, mock_requests_get):

        mock_requests_get.return_value.content = b'<feed><entry>'

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to parse'):
            create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)

    def test_raises_when_feed_has_no_entry(self, mock_requests_get):

        mock_requests_get.return_value.content = FEED_TEMPLATE.format(entry='').encode()

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='returned no paper'):
            create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)

    def test_raises_when_api_reports_error_entry(self, mock_requests_get):

        mock_requests_get.return_value.content = make_feed(
            entry_id='http://arxiv.org/api/errors#incorrect_id_format_for_foo'
        )

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='returned no paper'):
            create_arxiv_fetcher().fetch('foo')

    def test_raises_on_invalid_publication_date(self, mock_requests_get):

        mock_requests_get.return_value.content = make_feed(published='not a date')

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Invalid publication date'):
            create_arxiv_fetcher().fetch(SAMPLE_ARXIV_ID)


class TestHasArxivHtmlVersion:
    @pytest.fixture
    def mock_ar5iv_get(self, monkeypatch) -> Mock:

        get = Mock(
            return_value=make_response(
                200,
                headers={'content-type': 'text/html; charset=utf-8'},
                text='<html><div class="ltx_page_main"><article class="ltx_document">',
            )
        )

        monkeypatch.setattr(
            'paper_enrichment_agent.paper_scraping_service.components.fetchers.requests.get', get
        )

        return get

    @pytest.mark.parametrize('arxiv_id', ['1706.03762', 'arXiv:1706.03762v5', 'hep-th/9901001'])
    def test_returns_true_for_article_page(self, mock_ar5iv_get, arxiv_id):

        assert create_arxiv_fetcher().has_arxiv_html_version(arxiv_id)

        mock_ar5iv_get.assert_called_once_with(
            f'{AR5IV_URL}/{arxiv_id.removeprefix("arXiv:")}',
            allow_redirects=False,
            timeout=TIMEOUT,
        )

    def test_returns_false_on_redirect_to_abstract_page(self, mock_ar5iv_get):

        mock_ar5iv_get.return_value = make_response(
            302, headers={'location': 'https://arxiv.org/abs/1706.03762'}
        )

        assert not create_arxiv_fetcher().has_arxiv_html_version('1706.03762')

    def test_returns_false_for_non_article_page(self, mock_ar5iv_get):

        mock_ar5iv_get.return_value = make_response(
            200, headers={'content-type': 'text/html'}, text='<html>Conversion failed</html>'
        )

        assert not create_arxiv_fetcher().has_arxiv_html_version('1706.03762')

    @pytest.mark.parametrize('arxiv_id', ['', 'not-an-id', '1706.03762/abs', '17060.3762'])
    def test_raises_on_invalid_id(self, mock_ar5iv_get, arxiv_id):

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='is not a valid arXiv ID'):
            create_arxiv_fetcher().has_arxiv_html_version(arxiv_id)

        mock_ar5iv_get.assert_not_called()

    @pytest.mark.parametrize(
        'response, error',
        [
            (make_response(302, headers={'location': 'https://other.org'}), 'unexpected location'),
            (make_response(404), 'does not exist'),
            (make_response(500), 'status 500'),
        ],
    )
    def test_raises_on_unexpected_response(self, mock_ar5iv_get, response, error):

        mock_ar5iv_get.return_value = response

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match=error):
            create_arxiv_fetcher().has_arxiv_html_version('1706.03762')

    def test_raises_when_request_fails(self, mock_ar5iv_get):

        mock_ar5iv_get.side_effect = requests.Timeout('Timed out')

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to reach ar5iv'):
            create_arxiv_fetcher().has_arxiv_html_version('1706.03762')


class TestIsArxivPdfLocationCorrect:
    @pytest.fixture
    def mock_requests_head(self, monkeypatch) -> Mock:

        head = Mock(return_value=make_response(200, headers={'content-type': 'application/pdf'}))

        monkeypatch.setattr(
            'paper_enrichment_agent.paper_scraping_service.components.fetchers.requests.head', head
        )

        return head

    @pytest.mark.parametrize(
        'pdf_url',
        [
            'https://arxiv.org/pdf/1706.03762',
            'https://arxiv.org/pdf/1706.03762v5',
            'https://arxiv.org/pdf/hep-th/9901001',
        ],
    )
    def test_returns_true_for_existing_pdf(self, mock_requests_head, pdf_url):

        assert create_arxiv_fetcher().is_arxiv_pdf_location_correct(f' {pdf_url} ')

        mock_requests_head.assert_called_once_with(pdf_url, allow_redirects=True, timeout=TIMEOUT)

    @pytest.mark.parametrize(
        'response',
        [make_response(404, headers={'content-type': 'text/html'}), make_response(200)],
    )
    def test_returns_false_when_no_pdf_is_served(self, mock_requests_head, response):

        mock_requests_head.return_value = response

        assert not create_arxiv_fetcher().is_arxiv_pdf_location_correct(
            'https://arxiv.org/pdf/1706.03762'
        )

    @pytest.mark.parametrize(
        'pdf_url',
        [
            'https://arxiv.org/abs/1706.03762',
            'http://arxiv.org/pdf/1706.03762',
            'https://example.org/pdf/1706.03762',
            'https://arxiv.org/pdf/1706.03762.pdf',
        ],
    )
    def test_raises_on_non_arxiv_pdf_url(self, mock_requests_head, pdf_url):

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='is not an arXiv PDF URL'):
            create_arxiv_fetcher().is_arxiv_pdf_location_correct(pdf_url)

        mock_requests_head.assert_not_called()

    def test_raises_when_request_fails(self, mock_requests_head):

        mock_requests_head.side_effect = requests.ConnectionError('Connection refused')

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to reach arXiv'):
            create_arxiv_fetcher().is_arxiv_pdf_location_correct('https://arxiv.org/pdf/1706.03762')


class TestOpenAlexScraper:
    _SAMPLE_WORKS = [
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
        },
    ]

    def create_scraper(
        self, handler: Handler, api_key: str = '', max_results: int = 2
    ) -> OpenAlexScraper:

        return OpenAlexScraper(
            http_client=make_async_client(handler),
            api_url=OPENALEX_URL,
            api_key=api_key,
            max_results=max_results,
            timeout=TIMEOUT,
        )

    @pytest.mark.asyncio
    async def test_returns_preprocessed_works(self):

        requests_sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_sent.append(request)
            return httpx.Response(200, json={'results': self._SAMPLE_WORKS})

        works = await self.create_scraper(handler).search_paper_by_title(
            ' Attention is all you need '
        )

        assert len(works) == 2

        assert works[0].doi == '10.48550/arxiv.1706.03762'
        assert works[0].arxiv_id == '1706.03762'
        assert works[0].authors == ['Ashish Vaswani', 'Noam Shazeer']
        assert works[0].venue == 'arXiv (Cornell University)'
        assert works[0].publication_date == '2017-06-12'

        assert works[1].doi == '10.1109/icassp.2021.1'
        assert works[1].arxiv_id == '2010.13154'
        assert works[1].venue is None

        assert len(requests_sent) == 1
        params = requests_sent[0].url.params
        assert params['search'] == 'Attention is all you need'
        assert params['per_page'] == '2'
        assert 'open_access' not in params['select']
        assert 'api_key' not in params

    @pytest.mark.asyncio
    async def test_handles_missing_fields(self):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'results': self._SAMPLE_WORKS})

        works = await self.create_scraper(handler, max_results=3).search_paper_by_title('Closed')

        assert works[2].doi is None
        assert works[2].arxiv_id is None
        assert works[2].authors == []
        assert works[2].relevance_score is None

    @pytest.mark.asyncio
    async def test_passes_api_key_if_configured(self):

        requests_sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_sent.append(request)
            return httpx.Response(200, json={'results': []})

        scraper = self.create_scraper(handler, api_key='secret-key')

        assert await scraper.search_paper_by_title('Title') == []
        assert requests_sent[0].url.params['api_key'] == 'secret-key'

    @pytest.mark.asyncio
    async def test_raises_on_empty_title(self):

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError('No request expected.')

        with pytest.raises(OpenAlexScraper.FetchingError, match='must not be empty'):
            await self.create_scraper(handler).search_paper_by_title('  ')

    @pytest.mark.asyncio
    async def test_raises_on_error_status(self):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text='Service unavailable')

        with pytest.raises(OpenAlexScraper.FetchingError, match='responded with status 503'):
            await self.create_scraper(handler).search_paper_by_title('Title')

    @pytest.mark.asyncio
    async def test_raises_on_connection_error(self):

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError('Connection refused')

        with pytest.raises(OpenAlexScraper.FetchingError, match='Failed to reach OpenAlex'):
            await self.create_scraper(handler).search_paper_by_title('Title')

    @pytest.mark.parametrize('content', [b'not json', b'[1, 2]'])
    @pytest.mark.asyncio
    async def test_raises_on_invalid_response(self, content):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=content)

        with pytest.raises(OpenAlexScraper.FetchingError, match='OpenAlex returned'):
            await self.create_scraper(handler).search_paper_by_title('Title')


class TestUnpaywallFetcher:
    def create_fetcher(self, handler: Handler) -> UnpaywallFetcher:

        return UnpaywallFetcher(
            http_client=make_async_client(handler),
            api_url=UNPAYWALL_URL,
            email='tools@example.org',
            timeout=TIMEOUT,
        )

    @pytest.mark.asyncio
    async def test_returns_pdf_locations(self):

        requests_sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_sent.append(request)
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

        result = await self.create_fetcher(handler).fetch('https://doi.org/10.1/abc')

        assert result.doi == '10.1/abc'
        assert result.title == 'Sample paper'
        assert result.is_oa
        assert result.best_pdf_url is None
        assert [location.url_for_pdf for location in result.pdf_locations] == [
            'https://repo.org/paper.pdf'
        ]
        assert result.pdf_locations[0].license == 'cc-by'

        assert requests_sent[0].url.path == '/v2/10.1/abc'
        assert requests_sent[0].url.params['email'] == 'tools@example.org'

    @pytest.mark.asyncio
    async def test_returns_best_pdf_url(self):

        def handler(request: httpx.Request) -> httpx.Response:
            location = {'url_for_pdf': 'https://pub.org/paper.pdf', 'host_type': 'publisher'}
            return httpx.Response(
                200, json={'is_oa': True, 'best_oa_location': location, 'oa_locations': [location]}
            )

        result = await self.create_fetcher(handler).fetch('10.1/abc')

        assert result.best_pdf_url == 'https://pub.org/paper.pdf'

    @pytest.mark.asyncio
    async def test_reports_closed_paper(self):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={'is_oa': False, 'best_oa_location': None, 'oa_locations': []}
            )

        result = await self.create_fetcher(handler).fetch('10.1/abc')

        assert not result.is_oa
        assert result.best_pdf_url is None
        assert result.pdf_locations == []

    @pytest.mark.asyncio
    async def test_raises_on_unknown_doi(self):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        with pytest.raises(UnpaywallFetcher.FetchingError, match='Unpaywall has no record'):
            await self.create_fetcher(handler).fetch('10.1/unknown')

    @pytest.mark.asyncio
    async def test_raises_on_invalid_doi(self):

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError('No request expected.')

        with pytest.raises(UnpaywallFetcher.FetchingError, match='is not a valid DOI'):
            await self.create_fetcher(handler).fetch('arXiv:1706.03762')
