"""Main entrypoint for the `footnote_enrichment_agent` service.

This script sets up the components of the service and runs a FastAPI server for inter-service
communication.
"""

import functools
import pathlib
from typing import Annotated

import fastapi
import hydra
import omegaconf
import pydantic
from pydantic import Field
from starlette.types import Receive, Scope, Send

from paper_enrichment_agent.common import logging_setup
from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.footnote_enrichment_agent.components import enrichment_context
from paper_enrichment_agent.footnote_enrichment_agent.service import FootnoteEnrichmentAgentService
from paper_enrichment_agent.footnote_enrichment_agent.service import Metrics


@functools.cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger('paper_enrichment_agent.__main__')


class AppCfg(pydantic.BaseModel):
    """Contains configuration of the service."""

    json_logs_path: Annotated[
        pathlib.Path, Field(description='Output path for storing json log events.')
    ]


@hydra.main(version_base=None, config_path='cfg', config_name='main')
def main(hydra_cfg: omegaconf.DictConfig) -> None:
    """Main entrypoint of the application."""

    app_cfg = AppCfg.model_validate(omegaconf.OmegaConf.to_container(hydra_cfg))

    logging_setup.setup_logging(app_cfg.json_logs_path)

    _logger().info('Running paper_enrichment_agent service', cfg=app_cfg)

    enrichment_context_manager = enrichment_context.EnrichmentContextManager()
    service = FootnoteEnrichmentAgentService(
        metrics=Metrics(),
        enrichment_context_manager=enrichment_context_manager,
    )

    api = fastapi.FastAPI(title='Footnote Enrichment Agent')

    @api.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    @api.post('/reference_to_footnote')
    async def reference_to_footnote(
        request: misc_models.FootnoteEnrichmentRequest,
    ) -> doc_models.Section:
        return await service.reference_to_footnote(request)

    async def mcp_dispatch(scope: Scope, receive: Receive, send: Send) -> None:

        session_id = scope['path_params']['session_id']
        mcp_endpoint = enrichment_context_manager.get_mcp_app_for_agent_session(session_id)

        await mcp_endpoint(scope, receive, send)

    api.router.routes.append(
        fastapi.routing.APIRoute(
            path='/mcp/{session_id:path}',
            endpoint=mcp_dispatch,
            methods=['GET', 'POST'],
        )
    )


if __name__ == '__main__':
    main()
