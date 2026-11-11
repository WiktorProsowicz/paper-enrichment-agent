"""Contains the client class that communicates with the document database.

The client implements the layout of the documents and surveys within the database, while the
storage itself is accessed through the injected
:class:`~paper_enrichment_agent.main_survey_enricher.components.db_clients.core.DBClient`.

Exported classes:
    DocDBClient
"""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator

import pydantic
import requests

from paper_enrichment_agent.common.document_manipulators import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import DocumentMetadata, SurveyMetadata
from paper_enrichment_agent.main_survey_enricher.components.db_clients import core


class DocDBClient:
    """Handles read/write requests to the document database.

    The objects are stored under the following paths:
        - `documents/<paper_id>/metadata.json`: the document's metadata,
        - `documents/<paper_id>/document.json`: the document's structure,
        - `documents/<paper_id>/images/<image_id>.<ext>`: the document's images,
        - `surveys/<paper_id>/metadata.json`: the survey's metadata.

    The operations raise `DocDBClientError` on failure. They are asynchronous; the image downloads,
    which use a blocking HTTP client, are run in separate threads so that they do not block the
    event loop.
    """

    class DocDBClientError(Exception):
        """Base class for exceptions raised by the `DocDBClient`."""

    def __init__(self, db_client: core.DBClient) -> None:

        self._db_client = db_client

    async def document_exists(self, paper_id: str) -> bool:
        """Tells whether a document with the given id is present in the database.

        Args:
            paper_id: The identifier of the document to be looked up.

        Returns:
            True if the document is present in the database, False otherwise.
        """

        try:
            return await self._db_client.object_exists(path=f'documents/{paper_id}/metadata.json')

        except core.DBClientException as e:
            raise self.DocDBClientError(
                f'Failed to check the presence of the document {paper_id} in database: {e}'
            ) from e

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

        try:
            await self._db_client.upload_model(
                doc_metadata, path=f'documents/{doc_metadata.paper_id}/metadata.json'
            )
            await self._db_client.upload_model(
                document, path=f'documents/{doc_metadata.paper_id}/document.json'
            )

        except core.DBClientException as e:
            raise self.DocDBClientError(
                f'Failed to add the document {doc_metadata.paper_id} to database: {e}'
            ) from e

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

        try:
            await self._db_client.upload_model(
                survey_metadata, path=f'surveys/{paper_id}/metadata.json'
            )

        except core.DBClientException as e:
            raise self.DocDBClientError(
                f'Failed to register the document {paper_id} as a survey: {e}'
            ) from e

        return survey_metadata

    async def get_available_surveys(self) -> list[SurveyMetadata]:
        """Returns a list of all available surveys in the document database.

        Returns:
            A list of `SurveyMetadata` objects representing the available surveys.
        """

        try:
            survey_dirs = await self._db_client.list_subdirectories(prefix='surveys/')

        except core.DBClientException as e:
            raise self.DocDBClientError(f'Failed to list surveys in database: {e}') from e

        return [
            await self._get_model(path=f'{survey_dir}metadata.json', model_type=SurveyMetadata)
            for survey_dir in survey_dirs
        ]

    async def get_survey_info(self, survey_id: str) -> SurveyMetadata:
        """Returns the metadata of the survey with the given id.

        The caller is responsible for ensuring that the survey is present in the database,
        see `document_exists`.

        Args:
            survey_id: The identifier of the survey in the database.

        Returns:
            The metadata of the requested survey.
        """

        return await self._get_model(
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

        try:
            async with self._db_client.model_reference(
                path=f'surveys/{survey_id}/metadata.json', model_type=SurveyMetadata
            ) as survey:
                survey.referenced_docs[reference_id] = referenced_paper_id

        except core.DBClientException as e:
            raise self.DocDBClientError(
                f'Failed to add the referenced document to survey {survey_id}: {e}'
            ) from e

    async def delete_survey(self, survey_id: str) -> None:
        """Deletes a survey and its referenced documents from the database.

        The caller is responsible for ensuring that the survey is present in the database,
        see `document_exists`.

        Args:
            survey_id: The identifier of the survey to be deleted.
        """

        metadata = await self._get_model(
            path=f'surveys/{survey_id}/metadata.json', model_type=SurveyMetadata
        )

        for referenced_doc_id in metadata.referenced_docs.values():
            await self._delete_document(paper_id=referenced_doc_id)

        try:
            await self._db_client.delete_object(path=f'surveys/{survey_id}/metadata.json')

        except core.DBClientException as e:
            raise self.DocDBClientError(
                f'Failed to delete survey {survey_id} from database: {e}'
            ) from e

        await self._delete_document(paper_id=survey_id)

    @contextlib.asynccontextmanager
    async def get_document_struct_ref(
        self, paper_id: str
    ) -> AsyncGenerator[doc_models.Document, None]:
        """Returns an async context manager that yields a reference to a document structure from
        the database.

        The context manager ensures that the reference is properly synchronized with the
        database.

        Args:
            paper_id: The identifier of the document in the database.
        """

        try:
            async with self._db_client.model_reference(
                path=f'documents/{paper_id}/document.json', model_type=doc_models.Document
            ) as document:
                yield document

        except core.DBClientException as e:
            raise self.DocDBClientError(
                f'Failed to synchronize the structure of document {paper_id}: {e}'
            ) from e

    async def get_document_metadata(self, paper_id: str) -> DocumentMetadata:
        """Retrieves the metadata of a document from the database.

        Args:
            paper_id: The identifier of the document in the database.
        """

        return await self._get_model(
            path=f'documents/{paper_id}/metadata.json', model_type=DocumentMetadata
        )

    async def _get_model[T: pydantic.BaseModel](self, path: str, model_type: type[T]) -> T:
        """Retrieves a model from the database, wrapping the errors."""

        try:
            return await self._db_client.get_model(path=path, model_type=model_type)

        except core.DBClientException as e:
            raise self.DocDBClientError(
                f'Failed to retrieve JSON from database at {path}: {e}'
            ) from e

    async def _delete_document(self, paper_id: str) -> None:
        """Deletes a document and its images from the database."""

        try:
            await self._db_client.delete_object(path=f'documents/{paper_id}/metadata.json')
            await self._db_client.delete_object(path=f'documents/{paper_id}/document.json')

            for image_path in await self._db_client.list_objects(
                prefix=f'documents/{paper_id}/images/'
            ):
                await self._db_client.delete_object(path=image_path)

        except core.DBClientException as e:
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

            await self._transfer_image(image_src=image.image_src, image_db_path=image_db_path)

            doc_metadata.images[doc_getter.get_path_of_component(image)] = image_db_path

    async def _transfer_image(self, image_src: str, image_db_path: str) -> None:
        """Streams an image from its source URL to the database."""

        try:
            response = await asyncio.to_thread(requests.get, image_src, stream=True)

        except requests.RequestException as e:
            raise self.DocDBClientError(f'Failed to download image from {image_src}: {e}') from e

        with response:
            try:
                response.raise_for_status()

            except requests.RequestException as e:
                raise self.DocDBClientError(
                    f'Failed to download image from {image_src}: {e}'
                ) from e

            try:
                await self._db_client.write_stream(
                    path=image_db_path,
                    stream=response.raw,
                    content_type=response.headers.get('Content-Type', 'application/octet-stream'),
                )

            except core.DBClientException as e:
                raise self.DocDBClientError(
                    f'Failed to upload image to database at {image_db_path}: {e}'
                ) from e
