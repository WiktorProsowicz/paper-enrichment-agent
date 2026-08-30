"""Contains a custom HTTP client class.

The client provides an exception-based error handling mechanism and MLFlow trace context
propagation.
"""

from functools import cache

import httpx
import mlflow
import pydantic

from paper_enrichment_agent.common import logging_setup


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


class HTTPClient:
    """Calls HTTP endpoints and propagates MLFlow trace context."""

    class HTTPClientError(Exception):
        """Raised when the HTTP request fails."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url
        self._timeout = timeout

        self._base_client = httpx.AsyncClient()

    async def apost[ResponseSchema: pydantic.BaseModel](
        self, path: str, payload: pydantic.BaseModel, response_schema: type[ResponseSchema]
    ) -> ResponseSchema:
        """Calls the HTTP POST endpoint and returns the response.

        Args:
            path: The path to the endpoint.
            payload: The payload to send.
            response_schema: The schema to use for the response.

        Returns:
            The response as an instance of the specified schema.

        Raises:
            HTTPClientError: If the HTTP request fails.
        """

        try:
            headers = mlflow.tracing.get_tracing_context_headers_for_http_request()

            response = await self._base_client.post(
                f'{self._base_url}{path}',
                json=payload.model_dump(),
                headers=headers,
                timeout=self._timeout,
            )

            response.raise_for_status()

            return response_schema.model_validate(response.json())

        except httpx.TimeoutException as e:
            _logger().error('HTTP request timed out', path=path, payload=payload.model_dump())
            raise self.HTTPClientError('HTTP request timed out') from e

        except httpx.RequestError as e:
            _logger().error('HTTP request failed', path=path, payload=payload.model_dump())
            raise self.HTTPClientError('HTTP request failed') from e

        except pydantic.ValidationError as e:
            _logger().error(
                'Failed to validate HTTP response',
                path=path,
                payload=payload.model_dump(),
                output_schema=response_schema.__name__,
            )
            raise self.HTTPClientError('Failed to validate HTTP response') from e
