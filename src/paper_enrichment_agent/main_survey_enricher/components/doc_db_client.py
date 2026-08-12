"""Contains the client class that communicates with the document database."""

import contextlib
import json
import uuid
from collections.abc import Generator

import pydantic
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

    def add_document(
        self, document: doc_models.Document, name: str, description: str
    ) -> DocumentMetadata:
        """Adds a document together with its metadata to the document database.

        Args:
            document: The document to be added to the database.
            name: The name of the document.
            description: The description of the document.

        Returns:
            The metadata of the freshly added document.

        Raises:
            DocDBClientError: If there is an error while adding the document to the database.
        """

        doc_metadata = DocumentMetadata(name=name, description=description, images={})

        self._download_and_store_images(document=document, doc_metadata=doc_metadata)

        self._upload_model_to_db(
            doc_metadata, path=f'documents/{doc_metadata.paper_id}/metadata.json'
        )
        self._upload_model_to_db(document, path=f'documents/{doc_metadata.paper_id}/document.json')

        return doc_metadata

    def register_as_survey(self, paper_id: str) -> SurveyMetadata:
        """Marks the document with the given id as a survey.

        Args:
            paper_id: The identifier of an already added document.

        Returns:
            The metadata of the freshly created survey.

        Raises:
            DocDBClientError: If there is no document with the given id in the database.
        """

        if not self._document_exists(paper_id):
            raise self.DocDBClientError(f'There is no document with the id {paper_id} in database.')

        survey_metadata = SurveyMetadata(paper_id=paper_id, referenced_docs={})

        self._upload_model_to_db(survey_metadata, path=f'surveys/{paper_id}/metadata.json')

        return survey_metadata

    def get_available_surveys(self) -> list[SurveyMetadata]:
        """Returns a list of all available surveys in the document database.

        Returns:
            A list of `SurveyMetadata` objects representing the available surveys.

        Raises:
            DocDBClientError: If there is an error while retrieving the surveys from the database.
        """

        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self._s3_bucket, Prefix='surveys/', Delimiter='/'
            )

            prefixes = response['CommonPrefixes']

        except (BotocoreClientError, KeyError) as e:
            raise self.DocDBClientError(f'Failed to list surveys in database: {e}') from e

        survey_metadata_list = []

        for prefix in prefixes:
            survey_prefix = prefix['Prefix']
            metadata_path = f'{survey_prefix}metadata.json'

            survey_metadata_list.append(
                self._get_model_from_db(path=metadata_path, model_type=SurveyMetadata)
            )

        return survey_metadata_list

    def _document_exists(self, paper_id: str) -> bool:
        """Tells whether a document with the given id is present in the database."""

        try:
            self._s3_client.head_object(
                Bucket=self._s3_bucket, Key=f'documents/{paper_id}/metadata.json'
            )

        except BotocoreClientError as e:
            if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey'):
                return False

            raise self.DocDBClientError(
                f'Failed to check the presence of the document {paper_id} in database: {e}'
            ) from e

        return True

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
    def _get_model_reference[T: pydantic.BaseModel](
        self, path: str, model_type: type[T]
    ) -> Generator[T, None, None]:
        """Returns a context manager that yields a Pydantic model reference from the database.

        The context manager ensures that the reference is properly synchronized with the
        database.
        """

        model = self._get_model_from_db(path=path, model_type=model_type)

        yield model

        self._upload_model_to_db(model=model, path=path)

    def _get_model_from_db[T: pydantic.BaseModel](self, path: str, model_type: type[T]) -> T:
        """Retrieves a Pydantic model from the database at the specified path."""

        try:
            json_file = self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)

        except BotocoreClientError as e:
            raise self.DocDBClientError(
                f'Failed to retrieve JSON from database at {path}: {e}'
            ) from e

        try:
            model = model_type.model_validate_json(json_file['Body'].read().decode('utf-8'))

        except pydantic.ValidationError as e:
            raise self.DocDBClientError(
                f'Failed to decode JSON from database at {path}: {e}'
            ) from e

        return model

    def _upload_model_to_db(self, model: pydantic.BaseModel, path: str) -> None:
        """Uploads a Pydantic model to the database at the specified path."""

        try:
            self._s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=path,
                Body=json.dumps(model.model_dump()).encode('utf-8'),
            )

        except BotocoreClientError as e:
            raise self.DocDBClientError(f'Failed to upload model to database at {path}: {e}') from e
