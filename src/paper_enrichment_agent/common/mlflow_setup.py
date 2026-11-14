"""Contains the setup for MLflow tracking server and experiment management.

The :func:`setup_mlflow` function should be called in the entrypoin of the service where the
MLFlow tracking is used. The idea is to connect the application either to a persistent MLFlow
tracking server for evaluation purposes or to a local docker-based MLFlow server for production
monitoring.

The :func:`trace_if_enabled` decorator traces a function only if the tracing is active, so the code
using it runs untraced e.g. in unit tests.
"""

import functools
import inspect
import os
from collections.abc import Awaitable, Callable
from functools import cache
from typing import Annotated, Any, cast

import mlflow
import pydantic
from mlflow.entities import SpanType
from mlflow.tracing import provider as mlflow_provider
from pydantic import Field

from paper_enrichment_agent.common import logging_setup


class MLFlowConfig(pydantic.BaseModel):
    """Represents the configuration for MLflow tracking server and experiment management."""

    tracking_uri: Annotated[str, Field(description='The URI of the MLflow tracking server')]
    experiment_name: Annotated[
        str | None,
        Field(
            description=(
                'The name of the MLflow experiment the traces are logged to. It is created if it '
                'does not exist. If None, the active experiment is left unchanged, e.g. for '
                'services whose traces only continue the ones started by their callers.'
            ),
            min_length=1,
        ),
    ] = None


def setup_mlflow(config: MLFlowConfig | None) -> None:
    """Sets up the MLflow tracking server and experiment management.

    Setting the experiment contacts the tracking server, so the server has to be reachable when
    `experiment_name` is configured.

    Args:
        config: Configuration of MLFlow. If None, MLFlow tracking is disabled.
    """

    if not config:
        _logger().info('MLFlow tracking is disabled.')
        mlflow.tracing.disable()
        return

    _logger().info(
        'MLFlow tracing is enabled.',
        server_uri=config.tracking_uri,
        experiment_name=config.experiment_name,
    )
    mlflow.set_tracking_uri(config.tracking_uri)

    if config.experiment_name is not None:
        mlflow.set_experiment(config.experiment_name)

    mlflow.litellm.autolog()

    os.environ['MLFLOW_TRACE_SAMPLE_RATIO'] = '1.0'
    os.environ['MLFLOW_ENABLE_ASYNC_TRACE_LOGGING'] = 'false'
    os.environ['MLFLOW_USE_BATCH_SPAN_PROCESSOR'] = 'false'
    os.environ['MLFLOW_ENABLE_OTEL_GENAI_SEMCONV'] = 'true'


def is_tracing_active() -> bool:
    """Tells whether the traces are logged, i.e. the tracking server is set up and tracing enabled.

    MLflow considers the tracing enabled by default, so the tracking URI is checked as well, to
    tell whether :func:`setup_mlflow` (or the `MLFLOW_TRACKING_URI` variable) configured a server.
    """

    return mlflow.is_tracking_uri_set() and bool(mlflow_provider.is_tracing_enabled())


def trace_if_enabled[**P, R](
    name: str, span_type: str = SpanType.UNKNOWN
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Traces the decorated function as an MLflow span, if the tracing is enabled.

    Unlike `mlflow.trace`, the decorator produces no traces unless the tracing is active (see
    :func:`is_tracing_active`). The check is done on each call, so the decorator can be
    applied at import time, before the setup. Both synchronous and asynchronous functions are
    supported.

    Args:
        name: The name of the span.
        span_type: The type of the span, see `mlflow.entities.SpanType`.

    Returns:
        A decorator wrapping the function.
    """

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        traced_func = cast(Callable[P, R], mlflow.trace(func, name=name, span_type=span_type))

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                target = traced_func if is_tracing_active() else func
                return await cast(Awaitable[Any], target(*args, **kwargs))

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            target = traced_func if is_tracing_active() else func
            return target(*args, **kwargs)

        return wrapper

    return decorate


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)
