FROM python:3.12-slim

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/app

COPY pyproject.toml uv.lock /opt/app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY src/ /opt/app/src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked