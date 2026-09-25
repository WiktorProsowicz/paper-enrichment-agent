"""Main entrypoint for the `footnote_enrichment_agent` service.

This script sets up the components of the service and runs a FastAPI server for inter-service
communication.
"""

import functools
import pathlib
from typing import Annotated, Any

import fastapi
import hydra
import mlflow
import omegaconf
import pydantic
import uvicorn
from langchain_litellm import ChatLiteLLM
from pydantic import Field
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from paper_enrichment_agent.common import logging_setup, mlflow_setup
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.footnote_enrichment_agent.components.agent import (
    FootnoteEnrichmentAgent,
)
from paper_enrichment_agent.footnote_enrichment_agent.components.enrichment_context_manager import (
    EnrichmentContextManager,
)
from paper_enrichment_agent.footnote_enrichment_agent.service import (
    FootnoteEnrichmentAgentService,
    Metrics,
)


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
    agent_cfg: Annotated[
        FootnoteEnrichmentAgent.Configuration,
        Field(description='Configuration of the footnote enrichment agent.'),
    ]
    agent_llm_model: Annotated[
        str,
        Field(
            description=(
                'The LiteLLM-supported model signature of the LLM used by the footnote'
                'enrichment agent.'
            )
        ),
    ]
    agent_llm_params: Annotated[
        dict[str, Any],
        Field(
            description=(
                'The model-specific parameters of the LLM used by the footnote '
                'enrichment agent. The params are passed to the LiteLLM constructor.'
            )
        ),
    ]
    llm_api_key: Annotated[
        pydantic.SecretStr,
        Field(description='The API key for the LLM used by the footnote enrichment agent'),
    ]


def create_app(
    service: FootnoteEnrichmentAgentService,
    enrichment_context_manager: EnrichmentContextManager,
) -> fastapi.FastAPI:
    """Creates a FastAPI application for the `footnote_enrichment_agent` service."""

    app = fastapi.FastAPI(title='Footnote Enrichment Agent')

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

    @app.post('/reference_to_footnote')
    async def reference_to_footnote(
        request: misc_models.FootnoteEnrichmentRequest,
    ) -> doc_models.Section:
        return await service.reference_to_footnote(request)

    class MCPDispatch:
        """Delegates the requests of the agent's MCP client to the session-specific MCP app."""

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            session_id = scope['path_params']['session_id']
            mcp_endpoint = enrichment_context_manager.get_mcp_app_for_agent_session(session_id)

            mcp_scope = {
                **scope,
                'path': '/mcp',
                'raw_path': b'/mcp',
                'path_params': {},
            }
            await mcp_endpoint(mcp_scope, receive, send)

    app.router.routes.append(
        Route(
            path='/mcp/{session_id:path}',
            endpoint=MCPDispatch(),
            methods=['GET', 'POST', 'DELETE'],
        )
    )

    return app


@hydra.main(version_base='1.3', config_path='cfg', config_name='main')
def main(hydra_cfg: omegaconf.DictConfig) -> None:
    """Main entrypoint of the application."""

    app_cfg = AppCfg.model_validate(omegaconf.OmegaConf.to_container(hydra_cfg))

    logging_setup.setup_logging(app_cfg.json_logs_path)
    _logger().info('Running paper_enrichment_agent service', cfg=app_cfg)

    mlflow_setup.setup_mlflow(app_cfg.mlflow_cfg)

    enrichment_context_manager = EnrichmentContextManager()

    enrichment_agent = FootnoteEnrichmentAgent(
        llm=ChatLiteLLM(
            model=app_cfg.agent_llm_model,
            api_key=app_cfg.llm_api_key.get_secret_value(),
            **app_cfg.agent_llm_params,
        ),
        cfg=app_cfg.agent_cfg,
    )

    service = FootnoteEnrichmentAgentService(
        metrics=Metrics(),
        enrichment_context_manager=enrichment_context_manager,
        enrichment_agent=enrichment_agent,
    )

    app = create_app(service=service, enrichment_context_manager=enrichment_context_manager)

    uvicorn.run(app, host='0.0.0.0', port=app_cfg.app_port)


if __name__ == '__main__':
    main()
