"""Contains the main entrypoint of the `paper_scraping_service` service.

The script sets up the components of the service and runs a FastAPI server for inter-service
communication. The service downloads papers from external sources and parses them into the
structured document format.
"""

import functools
import pathlib
from typing import Annotated

import fastapi
import hydra
import omegaconf
import prometheus_client
import pydantic
import uvicorn
from pydantic import Field

from paper_enrichment_agent.common import logging_setup
from paper_enrichment_agent.common.models import service_conn as conn_models
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
)
from paper_enrichment_agent.paper_scraping_service.service import Metrics, PaperScrapingService


@functools.cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger('paper_enrichment_agent.__main__')


class AppCfg(pydantic.BaseModel):
    """Contains configuration of the service."""

    json_logs_path: Annotated[
        pathlib.Path, Field(description='Output path for storing json log events.')
    ]
    app_port: Annotated[int, Field(description='The port on which the FastAPI app will run.')]
    arxiv_api_url: Annotated[
        str,
        Field(
            description='The URL of the arXiv export API query endpoint serving paper metadata.',
            min_length=1,
        ),
    ]
    arxiv_api_timeout: Annotated[
        float,
        Field(description='The timeout (in seconds) of a single arXiv API request.', gt=0),
    ]


def create_app(service: PaperScrapingService) -> fastapi.FastAPI:
    """Creates a FastAPI application for the `paper_scraping_service` service.

    Each endpoint is named after the `PaperScrapingService` method it delegates to. Service errors
    are returned as HTTP 500 responses with the error message in the `detail` field.

    Args:
        service: The service handling the requests.

    Returns:
        The configured FastAPI application.
    """

    app = fastapi.FastAPI(title='Paper Scraping Service')

    app.mount('/metrics', prometheus_client.make_asgi_app())

    @app.exception_handler(PaperScrapingService.PaperScrapingError)
    async def handle_service_error(
        request: fastapi.Request, exc: PaperScrapingService.PaperScrapingError
    ) -> fastapi.responses.JSONResponse:
        return fastapi.responses.JSONResponse(status_code=500, content={'detail': str(exc)})

    @app.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.post('/parse_arxiv_paper')
    async def parse_arxiv_paper(
        request: conn_models.ArxivPaperParsingRequest,
    ) -> conn_models.ParsedPaperResponse:
        document, metadata = await service.parse_arxiv_paper(request.arxiv_id)
        return conn_models.ParsedPaperResponse(document=document, metadata=metadata)

    return app


@hydra.main(version_base='1.3', config_path='cfg', config_name='main')
def main(hydra_cfg: omegaconf.DictConfig) -> None:
    """Main entrypoint of the application."""

    app_cfg = AppCfg.model_validate(omegaconf.OmegaConf.to_container(hydra_cfg))

    logging_setup.setup_logging(app_cfg.json_logs_path)
    _logger().info('Running paper_scraping_service service', cfg=app_cfg)

    service = PaperScrapingService(
        metrics=Metrics(),
        arxiv_metadata_fetcher=ArxivMetadataFetcher(
            api_url=app_cfg.arxiv_api_url, timeout=app_cfg.arxiv_api_timeout
        ),
    )

    app = create_app(service=service)

    uvicorn.run(app, host='0.0.0.0', port=app_cfg.app_port)


if __name__ == '__main__':
    main()
