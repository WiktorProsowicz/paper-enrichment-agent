"""Contains definition of custom logging tools based on structlog and stdlib logging."""

import logging
import logging.config
import pathlib
from collections.abc import Callable
from typing import Any

import pydantic
import structlog
from structlog.typing import EventDict


def _serialize_pydantic_objects(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Custom structlog processor to serialize Pydantic models to JSON-safe dicts."""

    for key, val in event_dict.items():
        if isinstance(val, pydantic.BaseModel):
            event_dict[key] = val.model_dump(mode='json')

    return event_dict


def setup_logging(
    json_log_file_path: pathlib.Path,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Sets up project-wide logging configuration using structlog and standard logging.

    Logs are rendered to the console in a human-readable form and, in parallel, written as
    raw one-line JSON documents to a rotating file, so that they can be scraped by Promtail
    without any additional parsing.

    Args:
        json_log_file_path: Path to the rotating JSONL file the unformatted JSON logs are saved to.
            Parent directories are created if missing.
        max_bytes: Maximum size of a single log file before it gets rotated.
        backup_count: Number of rotated log files kept alongside the active one.
    """

    json_log_file_path.parent.mkdir(parents=True, exist_ok=True)

    shared_processors: list[Callable[[Any, str, EventDict], EventDict]] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt='%Y-%m-%d %H:%M:%S', utc=True),
        _serialize_pydantic_objects,
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        foreign_pre_chain=shared_processors,
    )

    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console': {
                '()': lambda: console_formatter,
            },
            'json': {
                '()': lambda: json_formatter,
            },
        },
        'handlers': {
            'console_hand': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'console',
                'stream': 'ext://sys.stdout',
            },
            'json_file_hand': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'json',
                'filename': str(json_log_file_path),
                'maxBytes': max_bytes,
                'backupCount': backup_count,
                'encoding': 'utf-8',
            },
        },
        'loggers': {
            '': {
                'level': 'WARNING',
                'handlers': ['console_hand', 'json_file_hand'],
            },
            'paper_enrichment_agent': {
                'level': 'DEBUG',
                'handlers': ['console_hand', 'json_file_hand'],
                'propagate': False,
            },
        },
    }

    logging.config.dictConfig(logging_config)
