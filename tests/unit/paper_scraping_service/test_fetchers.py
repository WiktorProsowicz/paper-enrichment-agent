import datetime
from unittest.mock import Mock

import pytest
import requests

from paper_enrichment_agent.common.models.misc import PaperMetadata
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
)

API_URL = 'https://export.arxiv.org/api/query'
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


@pytest.fixture
def mock_requests_get(monkeypatch) -> Mock:

    get = Mock(return_value=Mock(content=make_feed(), raise_for_status=Mock()))

    monkeypatch.setattr(
        'paper_enrichment_agent.paper_scraping_service.components.fetchers.requests.get', get
    )

    return get


class TestArxivMetadataFetcher:
    def test_queries_api_with_configured_url_and_timeout(self, mock_requests_get):

        ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)

        mock_requests_get.assert_called_once_with(
            API_URL, params={'id_list': SAMPLE_ARXIV_ID}, timeout=TIMEOUT
        )

    def test_reads_authors_and_publication_date(self, mock_requests_get):

        metadata = ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)

        assert metadata == PaperMetadata(
            doi='10.48550/arXiv.1706.03762',
            authors=['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
            publication_date=datetime.date(2017, 6, 12),
        )

    def test_prefers_doi_declared_in_feed(self, mock_requests_get):

        mock_requests_get.return_value.content = make_feed(doi=' 10.1000/journal.123 ')

        metadata = ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)

        assert metadata.doi == '10.1000/journal.123'

    def test_strips_version_from_arxiv_doi(self, mock_requests_get):

        metadata = ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch('1706.03762v5')

        assert metadata.doi == '10.48550/arXiv.1706.03762'

    def test_raises_when_request_fails(self, mock_requests_get):

        mock_requests_get.side_effect = requests.ConnectionError('Connection refused')

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to download'):
            ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)

    def test_raises_on_http_error_status(self, mock_requests_get):

        mock_requests_get.return_value.raise_for_status.side_effect = requests.HTTPError(
            '503 Server Error'
        )

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to download'):
            ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)

    def test_raises_on_malformed_response(self, mock_requests_get):

        mock_requests_get.return_value.content = b'<feed><entry>'

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Failed to parse'):
            ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)

    def test_raises_when_feed_has_no_entry(self, mock_requests_get):

        mock_requests_get.return_value.content = FEED_TEMPLATE.format(entry='').encode()

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='returned no paper'):
            ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)

    def test_raises_when_api_reports_error_entry(self, mock_requests_get):

        mock_requests_get.return_value.content = make_feed(
            entry_id='http://arxiv.org/api/errors#incorrect_id_format_for_foo'
        )

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='returned no paper'):
            ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch('foo')

    def test_raises_on_invalid_publication_date(self, mock_requests_get):

        mock_requests_get.return_value.content = make_feed(published='not a date')

        with pytest.raises(ArxivMetadataFetcher.FetchingError, match='Invalid publication date'):
            ArxivMetadataFetcher(api_url=API_URL, timeout=TIMEOUT).fetch(SAMPLE_ARXIV_ID)
