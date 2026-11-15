# Marker API server. The heavy VLM OCR runs in the separate `surya-vllm` service (see
# `../marker-api.yaml`), so this image only needs CPU torch for marker's small in-process models.
FROM python:3.12-slim

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV APP_WORKDIR=/opt/app

RUN addgroup python-group \
    && adduser --disabled-password --shell /bin/bash --ingroup python-group python

RUN apt update && \
    apt install -y curl && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# CPU-only torch keeps the image small; it must be installed before marker pulls the default one.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ${APP_WORKDIR}/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r ${APP_WORKDIR}/requirements.txt

# Directory for the downloaded marker models, mounted as a named volume so that they survive
# container recreation. Named volumes inherit its ownership.
ENV MODEL_CACHE_DIR=/models/datalab
RUN mkdir -p ${MODEL_CACHE_DIR} && chown -R python:python-group /models

ENV TORCH_DEVICE=cpu
ENV SURYA_INFERENCE_BACKEND=vllm

USER python
WORKDIR ${APP_WORKDIR}

COPY --chown=python:python-group server.py ${APP_WORKDIR}/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
