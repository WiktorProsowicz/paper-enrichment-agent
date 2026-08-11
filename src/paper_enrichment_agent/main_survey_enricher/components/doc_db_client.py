"""Contains the client class that communicates with the document database."""

import contextlib
import json
import uuid
from collections.abc import Generator
from typing import Any

import requests
from botocore.exceptions import ClientError as BotocoreClientError
from mypy_boto3_s3 import S3Client

from paper_enrichment_agent.common.document_getter import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import DocumentMetadata, SurveyMetadata


class DocDBClient:
    """Handles read/write requests to the document database."""

    class DocDBClientError(Exception):
        """Base class for exceptions raised by the `DocDBClient`."""

    def __init__(self, s3_client: S3Client) -> None:

        self._s3_client = s3_client
        self._s3_bucket = 'document_database'

    def add_survey(self, document: doc_models.Document, name: str, description: str) -> None:
        """Adds a survey to the document database."""

        doc_metadata = DocumentMetadata(name=name, description=description, images={})

        self._download_and_store_images(document=document, doc_metadata=doc_metadata)

        survey_metadata = SurveyMetadata(doc_metadata=doc_metadata, referenced_docs={})

        with self._get_db_json_reference(
            f'documents/{doc_metadata.paper_id}/metadata.json', create_if_not_exists=True
        ) as metadata_json:
            metadata_json.update(survey_metadata.model_dump())

    def _download_and_store_images(
        self, document: doc_models.Document, doc_metadata: DocumentMetadata
    ) -> None:
        """Downloads and stores the images of a document in the database."""

        doc_getter = DocumentGetter(document=document)

        for image in doc_getter.iter_components_of_type(doc_models.ImgSubfigure):
            image_extension = image.image_src.split('.')[-1]

            image_db_id = uuid.uuid4().hex
            image_db_path = (
                f'documents/{doc_metadata.paper_id}/images/{image_db_id}.{image_extension}'
            )

            try:
                with requests.get(image.image_src, stream=True) as response:
                    response.raise_for_status()

                    self._s3_client.upload_fileobj(
                        response.raw,
                        Bucket=self._s3_bucket,
                        Key=image_db_path,
                        ExtraArgs={
                            'ContentType': response.headers.get(
                                'Content-Type', 'application/octet-stream'
                            )
                        },
                    )

            except requests.RequestException as e:
                raise self.DocDBClientError(
                    f'Failed to download image from {image.image_src}: {e}'
                ) from e

            except Exception as e:
                raise self.DocDBClientError(
                    f'Failed to upload image to database at {image_db_path}: {e}'
                ) from e

            doc_metadata.images[doc_getter.get_path_of_component(image)] = image_db_path

    @contextlib.contextmanager
    def _get_db_json_reference(
        self, path: str, create_if_not_exists: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        """Returns a context manager that yields a JSON reference to a document in the database.

        The context manager ensures that the JSON reference is properly synchronized with the
        database.
        """

        try:
            json_file = self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)

        except BotocoreClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey' and create_if_not_exists:
                self._s3_client.put_object(
                    Bucket=self._s3_bucket, Key=path, Body=json.dumps({}).encode('utf-8')
                )

            else:
                raise self.DocDBClientError(
                    f'Failed to retrieve JSON from database at {path}: {e}'
                ) from e

        finally:
            json_file = self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)

        try:
            json_content = json.loads(json_file['Body'].read().decode('utf-8'))

        except json.JSONDecodeError as e:
            raise self.DocDBClientError(
                f'Failed to decode JSON from database at {path}: {e}'
            ) from e

        yield json_content

        try:
            self._s3_client.put_object(
                Bucket=self._s3_bucket, Key=path, Body=json.dumps(json_content).encode('utf-8')
            )

        except BotocoreClientError as e:
            raise self.DocDBClientError(f'Failed to update JSON in database at {path}: {e}') from e
