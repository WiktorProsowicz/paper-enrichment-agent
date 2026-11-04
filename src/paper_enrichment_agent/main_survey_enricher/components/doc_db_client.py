"""Contains the client class that communicates with the document database."""

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Self

import pydantic
import requests
from botocore.exceptions import ClientError as BotocoreClientError
from mypy_boto3_s3 import S3Client
from pydantic import Field

from paper_enrichment_agent.common.document_manipulators import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import DocumentMetadata, SurveyMetadata


class DocDBClient:
    """Handles read/write requests to the document database.

    The client should be created with `create`, which creates the database's bucket if it is not
    present yet, so that the operations can assume that the bucket exists. The creation and the
    operations raise `DocDBClientError` on failure.

    The operations are asynchronous. Since the underlying S3 and HTTP clients are blocking, their
    calls are run in separate threads, so that they do not block the event loop.
    """

    class DocDBClientError(Exception):
        """Base class for exceptions raised by the `DocDBClient`."""

    class Configuration(pydantic.BaseModel):
        """Contains configuration of the `DocDBClient`."""

        bucket_name: Annotated[
            str,
            Field(
                description=(
                    'The name of the S3 bucket storing the documents. Must be a valid S3 bucket '
                    'name, i.e. lowercase letters, digits, dots and hyphens only.'
                ),
                min_length=3,
            ),
        ]

    def __init__(self, s3_client: S3Client, cfg: 'Configuration') -> None:

        self._s3_client = s3_client
        self._s3_bucket = cfg.bucket_name

    @classmethod
    def create(cls, s3_client: S3Client, cfg: 'Configuration') -> Self:
        """Creates the client and ensures that the database's bucket exists.

        The creation is blocking, as it is meant to happen once on the service's startup.

        Args:
            s3_client: The client of the S3-compatible storage backing the database.
            cfg: The configuration of the client.

        Returns:
            The client ready to perform the operations.

        Raises:
            DocDBClientError: If checking the presence of the bucket or creating it fails.
        """

        try:
            s3_client.head_bucket(Bucket=cfg.bucket_name)

        except BotocoreClientError as e:
            if e.response.get('Error', {}).get('Code') not in ('404', 'NoSuchBucket'):
                raise cls.DocDBClientError(
                    f'Failed to check the presence of the bucket {cfg.bucket_name}: {e}'
                ) from e

            try:
                s3_client.create_bucket(Bucket=cfg.bucket_name)

            except BotocoreClientError as e:
                raise cls.DocDBClientError(
                    f'Failed to create the bucket {cfg.bucket_name}: {e}'
                ) from e

        return cls(s3_client=s3_client, cfg=cfg)

    async def document_exists(self, paper_id: str) -> bool:
        """Tells whether a document with the given id is present in the database.

        Args:
            paper_id: The identifier of the document to be looked up.

        Returns:
            True if the document is present in the database, False otherwise.
        """

        try:
            await asyncio.to_thread(
                self._s3_client.head_object,
                Bucket=self._s3_bucket,
                Key=f'documents/{paper_id}/metadata.json',
            )

        except BotocoreClientError as e:
            if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey'):
                return False

            raise self.DocDBClientError(
                f'Failed to check the presence of the document {paper_id} in database: {e}'
            ) from e

        return True

    async def add_document(
        self, document: doc_models.Document, name: str, description: str
    ) -> DocumentMetadata:
        """Adds a document together with its metadata to the document database.

        Args:
            document: The document to be added to the database.
            name: The name of the document.
            description: The description of the document.

        Returns:
            The metadata of the freshly added document.
        """

        doc_metadata = DocumentMetadata(name=name, description=description, images={})

        await self._download_and_store_images(document=document, doc_metadata=doc_metadata)

        await self._upload_model_to_db(
            doc_metadata, path=f'documents/{doc_metadata.paper_id}/metadata.json'
        )
        await self._upload_model_to_db(
            document, path=f'documents/{doc_metadata.paper_id}/document.json'
        )

        return doc_metadata

    async def register_as_survey(self, paper_id: str) -> SurveyMetadata:
        """Marks the document with the given id as a survey.

        The caller is responsible for ensuring that the document is present in the database,
        see `document_exists`.

        Args:
            paper_id: The identifier of an already added document.

        Returns:
            The metadata of the freshly created survey.
        """

        survey_metadata = SurveyMetadata(paper_id=paper_id, referenced_docs={})

        await self._upload_model_to_db(survey_metadata, path=f'surveys/{paper_id}/metadata.json')

        return survey_metadata

    async def get_available_surveys(self) -> list[SurveyMetadata]:
        """Returns a list of all available surveys in the document database.

        Returns:
            A list of `SurveyMetadata` objects representing the available surveys.
        """

        try:
            response = await asyncio.to_thread(
                self._s3_client.list_objects_v2,
                Bucket=self._s3_bucket,
                Prefix='surveys/',
                Delimiter='/',
            )

        except BotocoreClientError as e:
            raise self.DocDBClientError(f'Failed to list surveys in database: {e}') from e

        survey_metadata_list = []

        # S3 omits `CommonPrefixes` from the response when no survey is registered.
        for prefix in response.get('CommonPrefixes', []):
            survey_prefix = prefix['Prefix']
            metadata_path = f'{survey_prefix}metadata.json'

            survey_metadata_list.append(
                await self._get_model_from_db(path=metadata_path, model_type=SurveyMetadata)
            )

        return survey_metadata_list

    async def get_survey_info(self, survey_id: str) -> SurveyMetadata:
        """Returns the metadata of the survey with the given id.

        The caller is responsible for ensuring that the survey is present in the database,
        see `document_exists`.

        Args:
            survey_id: The identifier of the survey in the database.

        Returns:
            The metadata of the requested survey.
        """

        return await self._get_model_from_db(
            path=f'surveys/{survey_id}/metadata.json', model_type=SurveyMetadata
        )

    async def add_referenced_document(
        self, survey_id: str, reference_id: str, referenced_paper_id: str
    ) -> None:
        """Adds a referenced document to a survey in the database.

        The caller is responsible for ensuring that both the survey and the referenced document
        are present in the database, see `document_exists`.

        Args:
            survey_id: The identifier of the survey in the database.
            reference_id: The identifier of the reference in the survey.
            referenced_paper_id: The identifier of the referenced document in the database.
        """

        survey_metadata_path = f'surveys/{survey_id}/metadata.json'

        async with self._get_model_reference(
            path=survey_metadata_path, model_type=SurveyMetadata
        ) as survey:
            survey.referenced_docs[reference_id] = referenced_paper_id

    async def delete_survey(self, survey_id: str) -> None:
        """Deletes a survey and its referenced documents from the database.

        The caller is responsible for ensuring that the survey is present in the database,
        see `document_exists`.

        Args:
            survey_id: The identifier of the survey to be deleted.
        """

        try:
            metadata = await self._get_model_from_db(
                path=f'surveys/{survey_id}/metadata.json', model_type=SurveyMetadata
            )

            for referenced_doc_id in metadata.referenced_docs.values():
                await self._delete_document(paper_id=referenced_doc_id)

            await asyncio.to_thread(
                self._s3_client.delete_object,
                Bucket=self._s3_bucket,
                Key=f'surveys/{survey_id}/metadata.json',
            )

            await self._delete_document(paper_id=survey_id)

        except BotocoreClientError as e:
            raise self.DocDBClientError(
                f'Failed to delete survey {survey_id} from database: {e}'
            ) from e

    @contextlib.asynccontextmanager
    async def get_document_struct_ref(
        self, paper_id: str
    ) -> AsyncGenerator[doc_models.Document, None]:
        """Returns an async context manager that yields a reference to a document structure from
        the database.

        The context manager ensures that the reference is properly synchronized with the
        database.
        """

        async with self._get_model_reference(
            path=f'documents/{paper_id}/document.json', model_type=doc_models.Document
        ) as document:
            yield document

    async def get_document_metadata(self, paper_id: str) -> DocumentMetadata:
        """Retrieves the metadata of a document from the database.

        Args:
            paper_id: The identifier of the document in the database.
        """

        return await self._get_model_from_db(
            path=f'documents/{paper_id}/metadata.json', model_type=DocumentMetadata
        )

    async def _delete_document(self, paper_id: str) -> None:
        """Deletes a document and its images from the database."""

        try:
            await asyncio.to_thread(
                self._s3_client.delete_object,
                Bucket=self._s3_bucket,
                Key=f'documents/{paper_id}/metadata.json',
            )

            await asyncio.to_thread(
                self._s3_client.delete_object,
                Bucket=self._s3_bucket,
                Key=f'documents/{paper_id}/document.json',
            )

            list_images_response = await asyncio.to_thread(
                self._s3_client.list_objects_v2,
                Bucket=self._s3_bucket,
                Prefix=f'documents/{paper_id}/images/',
            )
            for obj in list_images_response.get('Contents', []):
                await asyncio.to_thread(
                    self._s3_client.delete_object, Bucket=self._s3_bucket, Key=obj['Key']
                )

        except BotocoreClientError as e:
            raise self.DocDBClientError(
                f'Failed to delete document {paper_id} from database: {e}'
            ) from e

    async def _download_and_store_images(
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

            await asyncio.to_thread(
                self._transfer_image, image_src=image.image_src, image_db_path=image_db_path
            )

            doc_metadata.images[doc_getter.get_path_of_component(image)] = image_db_path

    def _transfer_image(self, image_src: str, image_db_path: str) -> None:
        """Streams an image from its source URL to the database (blocking)."""

        try:
            with requests.get(image_src, stream=True) as response:
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
            raise self.DocDBClientError(f'Failed to download image from {image_src}: {e}') from e

        except Exception as e:
            raise self.DocDBClientError(
                f'Failed to upload image to database at {image_db_path}: {e}'
            ) from e

    @contextlib.asynccontextmanager
    async def _get_model_reference[T: pydantic.BaseModel](
        self, path: str, model_type: type[T]
    ) -> AsyncGenerator[T, None]:
        """Returns an async context manager that yields a Pydantic model reference from the
        database.

        The context manager ensures that the reference is properly synchronized with the
        database.
        """

        model = await self._get_model_from_db(path=path, model_type=model_type)

        yield model

        await self._upload_model_to_db(model=model, path=path)

    async def _get_model_from_db[T: pydantic.BaseModel](self, path: str, model_type: type[T]) -> T:
        """Retrieves a Pydantic model from the database at the specified path."""

        try:
            json_bytes = await asyncio.to_thread(self._read_object, path=path)

        except BotocoreClientError as e:
            raise self.DocDBClientError(
                f'Failed to retrieve JSON from database at {path}: {e}'
            ) from e

        try:
            model = model_type.model_validate_json(json_bytes.decode('utf-8'))

        except pydantic.ValidationError as e:
            raise self.DocDBClientError(
                f'Failed to decode JSON from database at {path}: {e}'
            ) from e

        return model

    def _read_object(self, path: str) -> bytes:
        """Reads the whole content of the object at the specified path (blocking)."""

        return self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)['Body'].read()

    async def _upload_model_to_db(self, model: pydantic.BaseModel, path: str) -> None:
        """Uploads a Pydantic model to the database at the specified path."""

        try:
            await asyncio.to_thread(
                self._s3_client.put_object,
                Bucket=self._s3_bucket,
                Key=path,
                Body=json.dumps(model.model_dump()).encode('utf-8'),
            )

        except BotocoreClientError as e:
            raise self.DocDBClientError(f'Failed to upload model to database at {path}: {e}') from e
