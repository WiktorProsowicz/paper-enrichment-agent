"""Contains a custom HTTP client class.

The client provides an exception-based error handling mechanism and MLFlow trace context
propagation.
"""

from functools import cache
from typing import Any

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
        """Calls the HTTP POST endpoint with a JSON body and returns the response.

        Args:
            path: The path to the endpoint.
            payload: The payload to send.
            response_schema: The schema to use for the response.

        Returns:
            The response as an instance of the specified schema.

        Raises:
            HTTPClientError: If the HTTP request fails.
        """

        return await self._post(
            path,
            response_schema,
            log_payload=str(payload),
            json=payload.model_dump(),
        )

    async def apost_multipart[ResponseSchema: pydantic.BaseModel](
        self,
        path: str,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        response_schema: type[ResponseSchema],
    ) -> ResponseSchema:
        """Calls the HTTP POST endpoint with a multipart form and returns the response.

        Only the form fields and the names of the files are logged on failure, never the files'
        contents.

        Args:
            path: The path to the endpoint.
            data: The form fields to send.
            files: Maps the form field names to the sent files, given as
                `(file_name, content, content_type)`.
            response_schema: The schema to use for the response.

        Returns:
            The response as an instance of the specified schema.

        Raises:
            HTTPClientError: If the HTTP request fails.
        """

        log_payload = str(
            {**data, **{field: file_name for field, (file_name, _, _) in files.items()}}
        )

        return await self._post(
            path, response_schema, log_payload=log_payload, data=data, files=files
        )

    async def _post[ResponseSchema: pydantic.BaseModel](
        self,
        path: str,
        response_schema: type[ResponseSchema],
        log_payload: str,
        **request_kwargs: Any,
    ) -> ResponseSchema:
        """Sends a POST request with the trace context headers and validates the response."""

        try:
            headers = mlflow.tracing.get_tracing_context_headers_for_http_request()

            response = await self._base_client.post(
                f'{self._base_url}{path}',
                headers=headers,
                timeout=self._timeout,
                **request_kwargs,
            )

            response.raise_for_status()

            return response_schema.model_validate(response.json())

        except httpx.TimeoutException as e:
            _logger().error('HTTP request timed out', path=path, payload=log_payload)
            raise self.HTTPClientError('HTTP request timed out') from e

        except httpx.HTTPError as e:
            _logger().error('HTTP request failed', path=path, payload=log_payload)
            raise self.HTTPClientError('HTTP request failed') from e

        except pydantic.ValidationError as e:
            _logger().error(
                'Failed to validate HTTP response',
                path=path,
                payload=log_payload,
                output_schema=response_schema.__name__,
            )
            raise self.HTTPClientError('Failed to validate HTTP response') from e
