"""Contains the setup for MLflow tracking server and experiment management.

The :func:`setup_mlflow` function should be called in the entrypoin of the service where the
MLFlow tracking is used. The idea is to connect the application either to a persistent MLFlow
tracking server for evaluation purposes or to a local docker-based MLFlow server for production
monitoring.
"""

import os
from functools import cache
from typing import Annotated

import mlflow
import pydantic
from pydantic import Field

from paper_enrichment_agent.common import logging_setup


class MLFlowConfig(pydantic.BaseModel):
    """Represents the configuration for MLflow tracking server and experiment management."""

    tracking_uri: Annotated[str, Field(description='The URI of the MLflow tracking server')]


def setup_mlflow(config: MLFlowConfig | None) -> None:
    """Sets up the MLflow tracking server and experiment management.

    Args:
        config: Configuration of MLFlow. If None, MLFlow tracking is disabled.
    """

    if not config:
        _logger().info('MLFlow tracking is disabled.')
        mlflow.tracing.disable()
        return

    _logger().info('MLFlow tracing is enabled.', server_uri=config.tracking_uri)
    mlflow.set_tracking_uri(config.tracking_uri)

    mlflow.langchain.autolog()

    os.environ['MLFLOW_TRACE_SAMPLE_RATIO'] = '1.0'
    os.environ['MLFLOW_ENABLE_ASYNC_TRACE_LOGGING'] = 'true'


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)
