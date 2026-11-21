"""Contains the main entrypoint of the `main_survey_enricher` service.

The script sets up the components of the service and runs a FastAPI server exposing the API
endpoints used by the user-facing components. The documents are stored in an S3-compatible
document database, whose connection is configured via `AppCfg`.
"""

import functools
import pathlib
import shutil
import tempfile
from typing import Annotated

import boto3
import fastapi
import hydra
import omegaconf
import prometheus_client
import pydantic
import uvicorn
from pydantic import Field
from starlette.background import BackgroundTask

from paper_enrichment_agent.common import logging_setup, mlflow_setup
from paper_enrichment_agent.common.http_client import HTTPClient
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.common.models import service_conn as conn_models
from paper_enrichment_agent.main_survey_enricher.components.db_clients import (
    rendered_doc_db_client,
)
from paper_enrichment_agent.main_survey_enricher.components.db_clients.core import S3DBClient
from paper_enrichment_agent.main_survey_enricher.components.db_clients.doc_db_client import (
    DocDBClient,
)
from paper_enrichment_agent.main_survey_enricher.service import MainSurveyEnricherService, Metrics


@functools.cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger('paper_enrichment_agent.__main__')


class AppCfg(pydantic.BaseModel):
    """Contains configuration of the service."""

    json_logs_path: Annotated[
        pathlib.Path, Field(description='Output path for storing json log events.')
    ]
    app_port: Annotated[int, Field(description='The port on which the FastAPI app will run.')]
    mlflow_cfg: Annotated[
        mlflow_setup.MLFlowConfig,
        Field(description='Configuration of the MLflow tracing.'),
    ]
    s3_endpoint_url: Annotated[
        str, Field(description='The URL of the S3-compatible document database.', min_length=1)
    ]
    s3_region: Annotated[
        str, Field(description='The region passed to the S3 client.', min_length=1)
    ]
    s3_access_key_id: Annotated[
        pydantic.SecretStr,
        Field(description='The access key ID for the document database.', min_length=1),
    ]
    s3_secret_access_key: Annotated[
        pydantic.SecretStr,
        Field(description='The secret access key for the document database.', min_length=1),
    ]
    doc_db_bucket_name: Annotated[
        str,
        Field(
            description=(
                'The name of the S3 bucket storing the document database. Must be a valid S3 '
                'bucket name, i.e. lowercase letters, digits, dots and hyphens only.'
            ),
            min_length=3,
        ),
    ]
    rendered_db_bucket_name: Annotated[
        str,
        Field(
            description=(
                'The name of the S3 bucket storing the rendered (serialized) documents. Must be a '
                'valid S3 bucket name, i.e. lowercase letters, digits, dots and hyphens only.'
            ),
            min_length=3,
        ),
    ]
    footnote_enrichment_service_url: Annotated[
        str,
        Field(description='The base URL of the `footnote_enrichment_agent` service.', min_length=1),
    ]
    footnote_enrichment_timeout: Annotated[
        float,
        Field(
            description=(
                'The timeout (in seconds) of a single footnote enrichment request sent to the '
                '`footnote_enrichment_agent` service. It has to accommodate the whole agent run.'
            ),
            gt=0,
        ),
    ]
    paper_scraping_service_url: Annotated[
        str,
        Field(description='The base URL of the `paper_scraping_service` service.', min_length=1),
    ]
    paper_scraping_timeout: Annotated[
        float,
        Field(
            description=(
                'The timeout (in seconds) of a single paper parsing request sent to the '
                '`paper_scraping_service` service. It has to accommodate the paper download and '
                'the conversion of uploaded PDFs by the marker API.'
            ),
            gt=0,
        ),
    ]


def create_app(service: MainSurveyEnricherService) -> fastapi.FastAPI:
    """Creates a FastAPI application for the `main_survey_enricher` service.

    Each endpoint is named after the `MainSurveyEnricherService` method it delegates to. Reading
    endpoints use GET with query parameters, while the modifying ones use POST with a JSON body.
    The route handlers are asynchronous, since the service offloads its blocking operations to
    separate threads. Service errors are returned as HTTP 500 responses with the error message in
    the `detail` field. Rendered documents are downloaded as a file transfer: the archive is
    returned as an `application/zip` attachment.

    Args:
        service: The service handling the requests.

    Returns:
        The configured FastAPI application.
    """

    app = fastapi.FastAPI(title='Main Survey Enricher')

    app.mount('/metrics', prometheus_client.make_asgi_app())

    @app.exception_handler(MainSurveyEnricherService.MainSurveyEnricherError)
    async def handle_service_error(
        request: fastapi.Request, exc: MainSurveyEnricherService.MainSurveyEnricherError
    ) -> fastapi.responses.JSONResponse:
        return fastapi.responses.JSONResponse(status_code=500, content={'detail': str(exc)})

    @app.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.post('/register_arxiv_survey')
    async def register_arxiv_survey(request: conn_models.ArxivSurveyRegistrationRequest) -> None:
        await service.register_arxiv_survey(
            arxiv_id=request.arxiv_id, name=request.name, description=request.description
        )

    @app.post('/parse_referenced_paper_pdf')
    async def parse_referenced_paper_pdf(
        file: fastapi.UploadFile,
        survey_id: Annotated[str, fastapi.Form(min_length=1)],
        reference_id: Annotated[str, fastapi.Form(min_length=1)],
        doi: Annotated[str, fastapi.Form(min_length=1)],
    ) -> misc_models.DocumentMetadata:
        return await service.parse_referenced_paper_pdf(
            survey_id=survey_id,
            reference_id=reference_id,
            pdf_content=await file.read(),
            file_name=file.filename or '',
            doi=doi,
        )

    @app.post('/remove_survey')
    async def remove_survey(request: conn_models.SurveyRemovalRequest) -> None:
        await service.remove_survey(request.survey_id)

    @app.get('/list_surveys')
    async def list_surveys() -> list[misc_models.SurveyMetadata]:
        return await service.list_surveys()

    @app.get('/get_survey_info')
    async def get_survey_info(survey_id: str) -> misc_models.SurveyMetadata:
        return await service.get_survey_info(survey_id)

    @app.get('/get_document_details')
    async def get_document_details(paper_id: str) -> conn_models.DocumentDetailsResponse:
        metadata, document = await service.get_document_details(paper_id)
        return conn_models.DocumentDetailsResponse(metadata=metadata, document=document)

    @app.post('/remove_document_component')
    async def remove_document_component(
        request: conn_models.DocumentComponentRemovalRequest,
    ) -> None:
        await service.remove_document_component(
            paper_id=request.paper_id, component_path=request.component_path
        )

    @app.post('/render_document')
    async def render_document(
        request: conn_models.DocumentRenderingRequest,
    ) -> misc_models.RenderedDocumentMetadata:
        return await service.render_document(
            paper_id=request.paper_id, doc_format=request.doc_format
        )

    @app.post('/enrich_footnote')
    async def enrich_footnote(
        request: conn_models.SurveyFootnoteEnrichmentRequest,
    ) -> doc_models.Section:
        return await service.enrich_footnote(
            survey_id=request.survey_id, citation_path=request.citation_path
        )

    @app.get('/list_rendered_documents')
    async def list_rendered_documents(
        paper_id: str | None = None,
    ) -> list[misc_models.RenderedDocumentMetadata]:
        return await service.list_rendered_documents(paper_id)

    @app.get('/download_rendered_document', response_class=fastapi.responses.FileResponse)
    async def download_rendered_document(
        paper_id: str, doc_format: misc_models.SerializedDocFormat
    ) -> fastapi.responses.FileResponse:
        # The archive is staged in a temporary directory, removed once the response is sent.
        download_dir = pathlib.Path(tempfile.mkdtemp())
        local_path = download_dir / f'{paper_id}_{doc_format}.zip'

        try:
            await service.download_rendered_document(
                paper_id=paper_id, doc_format=doc_format, local_path=local_path
            )

        except BaseException:
            shutil.rmtree(download_dir, ignore_errors=True)
            raise

        return fastapi.responses.FileResponse(
            local_path,
            media_type='application/zip',
            filename=local_path.name,
            background=BackgroundTask(shutil.rmtree, download_dir, ignore_errors=True),
        )

    return app


@hydra.main(version_base='1.3', config_path='cfg', config_name='main')
def main(hydra_cfg: omegaconf.DictConfig) -> None:
    """Main entrypoint of the application."""

    app_cfg = AppCfg.model_validate(omegaconf.OmegaConf.to_container(hydra_cfg))

    logging_setup.setup_logging(app_cfg.json_logs_path)
    _logger().info('Running main_survey_enricher service', cfg=app_cfg)

    mlflow_setup.setup_mlflow(app_cfg.mlflow_cfg)

    s3_client = boto3.client(
        's3',
        endpoint_url=app_cfg.s3_endpoint_url,
        region_name=app_cfg.s3_region,
        aws_access_key_id=app_cfg.s3_access_key_id.get_secret_value(),
        aws_secret_access_key=app_cfg.s3_secret_access_key.get_secret_value(),
    )

    doc_db_client = DocDBClient(
        db_client=S3DBClient.create(s3_client=s3_client, bucket_name=app_cfg.doc_db_bucket_name)
    )
    rendered_db_client = rendered_doc_db_client.RenderedDocDBClient(
        db_client=S3DBClient.create(
            s3_client=s3_client, bucket_name=app_cfg.rendered_db_bucket_name
        )
    )

    footnote_agent_client = HTTPClient(
        base_url=app_cfg.footnote_enrichment_service_url,
        timeout=app_cfg.footnote_enrichment_timeout,
    )

    paper_scraping_client = HTTPClient(
        base_url=app_cfg.paper_scraping_service_url, timeout=app_cfg.paper_scraping_timeout
    )

    service = MainSurveyEnricherService(
        metrics=Metrics(),
        doc_db_client=doc_db_client,
        rendered_doc_db_client=rendered_db_client,
        footnote_agent_client=footnote_agent_client,
        paper_scraping_client=paper_scraping_client,
    )

    app = create_app(service=service)

    uvicorn.run(app, host='0.0.0.0', port=app_cfg.app_port)


if __name__ == '__main__':
    main()
