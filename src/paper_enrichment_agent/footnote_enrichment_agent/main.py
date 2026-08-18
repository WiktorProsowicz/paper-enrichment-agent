"""Main entrypoint for the `footnote_enrichment_agent` service.

This script sets up the components of the service and runs a FastAPI server for inter-service
communication.
"""

import functools
import pathlib
from typing import Annotated

import hydra
import omegaconf
import pydantic
from pydantic import Field

from paper_enrichment_agent.common import logging_setup


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


if __name__ == '__main__':
    main()
