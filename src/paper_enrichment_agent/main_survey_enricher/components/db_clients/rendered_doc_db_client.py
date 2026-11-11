"""Contains the client class that stores and retrieves the rendered (serialized) documents.

A rendered document is an archive produced by one of the document serializers (see
:mod:`~paper_enrichment_agent.main_survey_enricher.components.doc_serializers`), stored together
with its metadata. The storage itself is accessed through the injected
:class:`~paper_enrichment_agent.main_survey_enricher.components.db_clients.core.DBClient`.

Exported classes:
    RenderedDocDBClient
"""

import pathlib

from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.main_survey_enricher.components.db_clients import core


class RenderedDocDBClient:
    """Handles read/write requests concerning the rendered documents.

    The objects are stored under the following paths:
        - `rendered/<paper_id>/<format>/metadata.json`: the rendered document's metadata,
        - `rendered/<paper_id>/<format>/document.zip`: the rendered document's archive.

    A document has at most one rendered version per format, so rendering it again replaces the
    previous version. The client does not check whether the rendered document is present in the
    document database, which is the caller's responsibility.

    The operations raise `RenderedDocDBClientError` on failure. They are asynchronous and the
    transfers of the archives are streamed, so they neither block the event loop nor load the
    archives into memory.
    """

    class RenderedDocDBClientError(Exception):
        """Base class for exceptions raised by the `RenderedDocDBClient`."""

    _ROOT_DIR = 'rendered'
    _METADATA_FILE_NAME = 'metadata.json'
    _ARCHIVE_FILE_NAME = 'document.zip'
    _ARCHIVE_CONTENT_TYPE = 'application/zip'

    def __init__(self, db_client: core.DBClient) -> None:

        self._db_client = db_client

    async def add_rendered_document(
        self, paper_id: str, doc_format: misc_models.SerializedDocFormat, archive_path: pathlib.Path
    ) -> misc_models.RenderedDocumentMetadata:
        """Stores the rendered document's archive together with its metadata.

        The previously rendered version of the document in the same format, if any, is replaced.

        Args:
            paper_id: The identifier of the rendered document in the document database.
            doc_format: The format the document has been rendered into.
            archive_path: The local path of the archive produced by the serializer.

        Returns:
            The metadata of the stored rendered document.
        """

        try:
            metadata = misc_models.RenderedDocumentMetadata(
                paper_id=paper_id, doc_format=doc_format, size_bytes=archive_path.stat().st_size
            )

        except OSError as e:
            raise self.RenderedDocDBClientError(
                f'Failed to access the rendered archive {archive_path}: {e}'
            ) from e

        try:
            await self._db_client.upload_file(
                local_path=archive_path,
                path=self._archive_path(paper_id, doc_format),
                content_type=self._ARCHIVE_CONTENT_TYPE,
            )
            # The metadata is written last, so that it marks a fully stored archive.
            await self._db_client.upload_model(
                metadata, path=self._metadata_path(paper_id, doc_format)
            )

        except core.DBClientException as e:
            raise self.RenderedDocDBClientError(
                f'Failed to store the {doc_format} rendering of document {paper_id}: {e}'
            ) from e

        return metadata

    async def rendered_document_exists(
        self, paper_id: str, doc_format: misc_models.SerializedDocFormat
    ) -> bool:
        """Tells whether the document has been rendered into the given format.

        Args:
            paper_id: The identifier of the document in the document database.
            doc_format: The format of the rendering.

        Returns:
            True if the rendered document is present in the database, False otherwise.
        """

        try:
            return await self._db_client.object_exists(
                path=self._metadata_path(paper_id, doc_format)
            )

        except core.DBClientException as e:
            raise self.RenderedDocDBClientError(
                f'Failed to check the presence of the {doc_format} rendering of document '
                f'{paper_id}: {e}'
            ) from e

    async def list_rendered_documents(
        self, paper_id: str | None = None
    ) -> list[misc_models.RenderedDocumentMetadata]:
        """Lists the metadata of the rendered documents.

        Args:
            paper_id: If given, only the renderings of this document are listed.

        Returns:
            The metadata of the matching rendered documents.
        """

        prefix = f'{self._ROOT_DIR}/' if paper_id is None else f'{self._ROOT_DIR}/{paper_id}/'

        try:
            object_paths = await self._db_client.list_objects(prefix=prefix)

            return [
                await self._db_client.get_model(
                    path=path, model_type=misc_models.RenderedDocumentMetadata
                )
                for path in object_paths
                if path.endswith(f'/{self._METADATA_FILE_NAME}')
            ]

        except core.DBClientException as e:
            raise self.RenderedDocDBClientError(
                f'Failed to list the rendered documents under {prefix}: {e}'
            ) from e

    async def download_rendered_document(
        self,
        paper_id: str,
        doc_format: misc_models.SerializedDocFormat,
        local_path: pathlib.Path,
    ) -> None:
        """Downloads the rendered document's archive to a local file.

        The file is written on the host running the client. The caller is responsible for
        ensuring that the rendered document is present, see `rendered_document_exists`.

        Args:
            paper_id: The identifier of the document in the document database.
            doc_format: The format of the rendering.
            local_path: The path of the local file to be written. Its parent directory must exist.
        """

        try:
            await self._db_client.download_to_file(
                path=self._archive_path(paper_id, doc_format), local_path=local_path
            )

        except core.DBClientException as e:
            raise self.RenderedDocDBClientError(
                f'Failed to download the {doc_format} rendering of document {paper_id} to '
                f'{local_path}: {e}'
            ) from e

    def _metadata_path(self, paper_id: str, doc_format: misc_models.SerializedDocFormat) -> str:
        """Returns the database path of the rendered document's metadata."""

        return f'{self._ROOT_DIR}/{paper_id}/{doc_format}/{self._METADATA_FILE_NAME}'

    def _archive_path(self, paper_id: str, doc_format: misc_models.SerializedDocFormat) -> str:
        """Returns the database path of the rendered document's archive."""

        return f'{self._ROOT_DIR}/{paper_id}/{doc_format}/{self._ARCHIVE_FILE_NAME}'
