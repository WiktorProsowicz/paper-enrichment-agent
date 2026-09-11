FROM python:3.12-slim AS base

SHELL ["/bin/bash", "-c"]

ENV APP_WORKDIR=/opt/app
ENV DEBIAN_FRONTEND=noninteractive

RUN addgroup python-group \
    && adduser --disabled-password --shell /bin/bash --ingroup python-group python

RUN rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

USER python
WORKDIR ${APP_WORKDIR}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT="${APP_WORKDIR}/.venv"
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_CERTS=1

COPY --chown=python:python-group pyproject.toml uv.lock /opt/app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project


# ---------------------------------------------------
# Evaluation environment: used to run unit tests and
# LLM evaluation pipelines
# ---------------------------------------------------

FROM base AS eval

ENV UV_PROJECT_ENVIRONMENT="${APP_WORKDIR}/.venv"
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_CERTS=1

COPY --chown=python:python-group src/ /opt/app/src/
COPY --chown=python:python-group tests/ /opt/app/tests/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group dev

# ---------------------------------------------------
# Runtime environment: used to run the application
# ---------------------------------------------------

FROM base AS runtime

ENV UV_PROJECT_ENVIRONMENT="${APP_WORKDIR}/.venv"
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_CERTS=1

COPY --chown=python:python-group src/ /opt/app/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked