"""Contains the main entrypoint of the `paper_scraping_service` service.

The script sets up the components of the service and runs a FastAPI server for inter-service
communication. The service downloads papers from external sources and parses them into the
structured document format.
"""

import functools
import pathlib
from typing import Annotated, Any

import fastapi
import httpx
import hydra
import mlflow
import omegaconf
import prometheus_client
import pydantic
import uvicorn
from langchain_litellm import ChatLiteLLM
from pydantic import Field

from paper_enrichment_agent.common import logging_setup, mlflow_setup
from paper_enrichment_agent.common.models import service_conn as conn_models
from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
    CrossrefFetcher,
)
from paper_enrichment_agent.paper_scraping_service.components.paper_access_agent import (
    PaperAccessAgent,
)
from paper_enrichment_agent.paper_scraping_service.components.parsers.pdf_parser import PdfParser
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
    ar5iv_html_url_base: Annotated[
        str,
        Field(description='The base URL of the ar5iv HTML versions of arXiv papers.', min_length=1),
    ]
    arxiv_api_timeout: Annotated[
        float,
        Field(description='The timeout (in seconds) of a single arXiv / ar5iv request.', gt=0),
    ]
    crossref_api_url: Annotated[
        str,
        Field(description='The URL of the Crossref REST API works endpoint.', min_length=1),
    ]
    crossref_email: Annotated[
        str,
        Field(
            description=(
                'The contact email sent to the Crossref API, so that the requests are served by '
                'its "polite pool".'
            ),
            min_length=1,
        ),
    ]
    crossref_api_timeout: Annotated[
        float,
        Field(description='The timeout (in seconds) of a single Crossref request.', gt=0),
    ]
    pdf_parser: Annotated[
        PdfParser.Configuration,
        Field(description='Configuration of the PDF parser, including the marker API access.'),
    ]
    mlflow_cfg: Annotated[
        mlflow_setup.MLFlowConfig,
        Field(description='Configuration of the MLflow tracing.'),
    ]
    paper_access_agent: Annotated[
        PaperAccessAgent.Configuration,
        Field(description='Configuration of the paper access agent, including its tools.'),
    ]
    agent_llm_model: Annotated[
        str,
        Field(
            description=(
                'The LiteLLM-supported model signature of the LLM used by the paper access agent.'
            ),
            min_length=1,
        ),
    ]
    agent_llm_params: Annotated[
        dict[str, Any],
        Field(
            description=(
                'The model-specific parameters of the LLM used by the paper access agent. '
                'The params are passed to the LiteLLM constructor.'
            )
        ),
    ]
    llm_api_key: Annotated[
        pydantic.SecretStr,
        Field(description='The API key for the LLM used by the paper access agent.', min_length=1),
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

    @app.middleware('http')
    async def mlflow_trace_context(request: fastapi.Request, call_next):  # type: ignore

        headers = dict(request.headers)

        if 'traceparent' in headers:
            with mlflow.tracing.set_tracing_context_from_http_request_headers(headers):
                response = await call_next(request)

        else:
            response = await call_next(request)

        return response

    @app.post('/parse_arxiv_paper')
    async def parse_arxiv_paper(
        request: conn_models.ArxivPaperParsingRequest,
    ) -> conn_models.ParsedPaperResponse:
        document, metadata = await service.parse_arxiv_paper(request.arxiv_id)
        return conn_models.ParsedPaperResponse(document=document, metadata=metadata)

    @app.post('/download_pdf_paper')
    async def download_pdf_paper(
        file: fastapi.UploadFile,
        doi: Annotated[str, fastapi.Form(min_length=1)],
    ) -> conn_models.ParsedPaperResponse:
        document, metadata = await service.download_pdf_paper(
            pdf_content=await file.read(), file_name=file.filename or '', doi=doi
        )
        return conn_models.ParsedPaperResponse(document=document, metadata=metadata)

    @app.post('/resolve_paper_access')
    async def resolve_paper_access(
        request: conn_models.PaperAccessResolutionRequest,
    ) -> conn_models.PaperAccessResolutionResponse:
        access_info = await service.resolve_paper_access(request.citation_signature)
        return conn_models.PaperAccessResolutionResponse(access_info=access_info)

    @app.post('/parse_paper_from_citation_signature')
    async def parse_paper_from_citation_signature(
        request: conn_models.CitationSignatureParsingRequest,
    ) -> conn_models.ParsedPaperResponse | None:
        parsed_paper = await service.parse_paper_from_citation_signature(request.citation_signature)

        if parsed_paper is None:
            return None

        document, metadata = parsed_paper
        return conn_models.ParsedPaperResponse(document=document, metadata=metadata)

    return app


@hydra.main(version_base='1.3', config_path='cfg', config_name='main')
def main(hydra_cfg: omegaconf.DictConfig) -> None:
    """Main entrypoint of the application."""

    app_cfg = AppCfg.model_validate(omegaconf.OmegaConf.to_container(hydra_cfg))

    logging_setup.setup_logging(app_cfg.json_logs_path)
    _logger().info('Running paper_scraping_service service', cfg=app_cfg)

    mlflow_setup.setup_mlflow(app_cfg.mlflow_cfg)

    paper_access_agent = PaperAccessAgent(
        llm=ChatLiteLLM(
            model=app_cfg.agent_llm_model,
            api_key=app_cfg.llm_api_key.get_secret_value(),
            **app_cfg.agent_llm_params,
        ),
        cfg=app_cfg.paper_access_agent,
    )

    service = PaperScrapingService(
        metrics=Metrics(),
        arxiv_metadata_fetcher=ArxivMetadataFetcher(
            api_url=app_cfg.arxiv_api_url,
            ar5iv_html_url_base=app_cfg.ar5iv_html_url_base,
            timeout=app_cfg.arxiv_api_timeout,
        ),
        crossref_fetcher=CrossrefFetcher(
            http_client=httpx.AsyncClient(),
            api_url=app_cfg.crossref_api_url,
            email=app_cfg.crossref_email,
            timeout=app_cfg.crossref_api_timeout,
        ),
        paper_access_agent=paper_access_agent,
        pdf_parser_config=app_cfg.pdf_parser,
    )

    app = create_app(service=service)

    uvicorn.run(app, host='0.0.0.0', port=app_cfg.app_port)


if __name__ == '__main__':
    main()
