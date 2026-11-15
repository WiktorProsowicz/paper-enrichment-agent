"""FastAPI server converting PDF documents into marker's JSON representation.

The server wraps the `marker` package (https://github.com/datalab-to/marker) and always runs it in
the `balanced` mode: the embedded PDF text is used where clean, while tables, equations and scanned
pages are OCR-ed by an external Surya VLM inference server, whose URL is read by `surya` from the
`SURYA_INFERENCE_URL` environment variable. The small in-process models (Indra layout / OCR) are
loaded once at startup and shared by all the conversions.

Environment variables:
    MARKER_MAX_CONCURRENCY: The maximum number of conversions running at the same time.
"""

import asyncio
import logging
import os
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict, shutdown_models

CONVERSION_MODE = 'vllm'
MAX_CONCURRENCY = int(os.environ.get('MARKER_MAX_CONCURRENCY', '1'))

_PDF_MAGIC = b'%PDF'

logger = logging.getLogger('marker_api')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Loads the marker models on startup and stops the inference connection on shutdown."""

    logger.info('Loading marker models...')
    app.state.models = create_model_dict()
    app.state.conversion_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    logger.info('Marker models loaded.')

    yield

    shutdown_models(app.state.models)


app = FastAPI(title='Marker API', lifespan=lifespan)


def _convert_pdf_sync(pdf_path: str, models: dict[str, Any]) -> dict:
    config_parser = ConfigParser({'output_format': 'json', 'mode': CONVERSION_MODE})
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(pdf_path)
    return rendered.model_dump(mode='json')


@app.get('/health')
async def health() -> dict[str, str]:
    """Reports that the server is up and the models are loaded."""

    return {'status': 'ok'}


@app.post('/convert_pdf')
async def convert_pdf(request: Request, file: UploadFile) -> dict:
    """Converts the uploaded PDF into marker's JSON output.

    Args:
        request: The incoming request, used to access the shared models.
        file: The PDF document to convert, sent as the `file` multipart field.

    Returns:
        The serialized `marker.renderers.json.JSONOutput`: a tree of blocks (`children`) with
        their `block_type`, `html`, geometry and base64-encoded `images`, plus the document
        `metadata`.
    """

    content = await file.read()
    if not content.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail='The uploaded file is not a PDF document.')

    logger.info('Converting %s (%d bytes).', file.filename, len(content))

    with tempfile.NamedTemporaryFile(suffix='.pdf') as pdf_file:
        pdf_file.write(content)
        pdf_file.flush()

        async with request.app.state.conversion_semaphore:
            try:
                return await run_in_threadpool(
                    _convert_pdf_sync, pdf_file.name, request.app.state.models
                )
            except Exception as e:
                logger.exception('Conversion of %s failed.', file.filename)
                raise HTTPException(status_code=500, detail='PDF conversion failed.') from e
