import json
from collections.abc import Callable
from unittest.mock import MagicMock, Mock

import httpx
import pydantic
import pytest

from paper_enrichment_agent.common.http_client import HTTPClient

BASE_URL = 'http://service.test'

type Handler = Callable[[httpx.Request], httpx.Response]


class SampleRequest(pydantic.BaseModel):
    name: str


class SampleResponse(pydantic.BaseModel):
    value: int


@pytest.fixture(autouse=True)
def patch_logger(monkeypatch) -> MagicMock:

    mock_logger = MagicMock()
    monkeypatch.setattr(
        'paper_enrichment_agent.common.http_client._logger', Mock(return_value=mock_logger)
    )

    return mock_logger


def create_client(handler: Handler) -> HTTPClient:
    """Creates a client whose requests are served by the given handler."""

    client = HTTPClient(base_url=BASE_URL, timeout=5.0)
    client._base_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return client


class TestApost:
    @pytest.mark.asyncio
    async def test_sends_json_payload_and_validates_response(self):

        requests_sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_sent.append(request)
            return httpx.Response(200, json={'value': 7})

        response = await create_client(handler).apost(
            '/endpoint', SampleRequest(name='abc'), response_schema=SampleResponse
        )

        assert response == SampleResponse(value=7)
        assert str(requests_sent[0].url) == f'{BASE_URL}/endpoint'
        assert json.loads(requests_sent[0].content) == {'name': 'abc'}

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={'detail': 'Failure'})

        with pytest.raises(HTTPClient.HTTPClientError, match='HTTP request failed'):
            await create_client(handler).apost(
                '/endpoint', SampleRequest(name='abc'), response_schema=SampleResponse
            )


class TestApostMultipart:
    @pytest.mark.asyncio
    async def test_sends_form_fields_and_files(self):

        requests_sent: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests_sent.append(request)
            return httpx.Response(200, json={'value': 3})

        response = await create_client(handler).apost_multipart(
            '/upload',
            data={'doi': '10.1/abc'},
            files={'file': ('paper.pdf', b'%PDF-1.5 content', 'application/pdf')},
            response_schema=SampleResponse,
        )

        assert response == SampleResponse(value=3)

        (request,) = requests_sent
        assert str(request.url) == f'{BASE_URL}/upload'
        assert request.headers['Content-Type'].startswith('multipart/form-data')
        assert b'name="doi"' in request.content
        assert b'10.1/abc' in request.content
        assert b'filename="paper.pdf"' in request.content
        assert b'%PDF-1.5 content' in request.content

    @pytest.mark.asyncio
    async def test_raises_on_http_error_without_logging_file_contents(self, patch_logger):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={'detail': 'Failure'})

        with pytest.raises(HTTPClient.HTTPClientError, match='HTTP request failed'):
            await create_client(handler).apost_multipart(
                '/upload',
                data={'doi': '10.1/abc'},
                files={'file': ('paper.pdf', b'%PDF-secret', 'application/pdf')},
                response_schema=SampleResponse,
            )

        logged_payload = patch_logger.error.call_args.kwargs['payload']
        assert 'paper.pdf' in logged_payload
        assert '10.1/abc' in logged_payload
        assert 'PDF-secret' not in logged_payload

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout('Timed out')

        with pytest.raises(HTTPClient.HTTPClientError, match='timed out'):
            await create_client(handler).apost_multipart(
                '/upload', data={}, files={}, response_schema=SampleResponse
            )

    @pytest.mark.asyncio
    async def test_raises_on_invalid_response(self):

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={'unexpected': 'field'})

        with pytest.raises(HTTPClient.HTTPClientError, match='Failed to validate'):
            await create_client(handler).apost_multipart(
                '/upload', data={'doi': '10.1/abc'}, files={}, response_schema=SampleResponse
            )
