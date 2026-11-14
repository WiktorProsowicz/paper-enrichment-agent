import pytest
from unittest.mock import call, create_autospec

from paper_enrichment_agent.common.models.misc import (
    RenderedDocumentMetadata,
    SerializedDocFormat,
)
from paper_enrichment_agent.main_survey_enricher.components.db_clients.core import (
    DBClient,
    DBClientException,
)
from paper_enrichment_agent.main_survey_enricher.components.db_clients import (
    rendered_doc_db_client,
)

_MD = SerializedDocFormat.MARKDOWN


@pytest.fixture
def mock_db_client():

    return create_autospec(DBClient, instance=True)


@pytest.fixture
def rendered_db_client(mock_db_client) -> rendered_doc_db_client.RenderedDocDBClient:

    return rendered_doc_db_client.RenderedDocDBClient(db_client=mock_db_client)


@pytest.fixture
def archive_path(tmp_path):

    path = tmp_path / 'document.zip'
    path.write_bytes(b'archive-bytes')
    return path


class TestAddRenderedDocument:
    @pytest.mark.asyncio
    async def test_stores_archive_and_metadata(
        self, rendered_db_client, mock_db_client, archive_path
    ):

        metadata = await rendered_db_client.add_rendered_document('p1', _MD, archive_path)

        assert metadata.paper_id == 'p1'
        assert metadata.doc_format == _MD
        assert metadata.size_bytes == len(b'archive-bytes')
        mock_db_client.upload_file.assert_awaited_once_with(
            local_path=archive_path,
            path='rendered/p1/markdown/document.zip',
            content_type='application/zip',
        )
        mock_db_client.upload_model.assert_awaited_once_with(
            metadata, path='rendered/p1/markdown/metadata.json'
        )

    @pytest.mark.asyncio
    async def test_raises_for_missing_archive(self, rendered_db_client, mock_db_client, tmp_path):

        with pytest.raises(
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            match='Failed to access the rendered',
        ):
            await rendered_db_client.add_rendered_document('p1', _MD, tmp_path / 'missing.zip')

        mock_db_client.upload_file.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_store_metadata_when_upload_fails(
        self, rendered_db_client, mock_db_client, archive_path
    ):

        mock_db_client.upload_file.side_effect = DBClientException('Upload failed')

        with pytest.raises(
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            match='Failed to store the markdown',
        ):
            await rendered_db_client.add_rendered_document('p1', _MD, archive_path)

        mock_db_client.upload_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(
        self, rendered_db_client, mock_db_client, archive_path
    ):

        mock_db_client.upload_file.side_effect = RuntimeError('Unexpected')

        with pytest.raises(RuntimeError, match='Unexpected'):
            await rendered_db_client.add_rendered_document('p1', _MD, archive_path)


class TestRenderedDocumentExists:
    @pytest.mark.parametrize('exists', [True, False])
    @pytest.mark.asyncio
    async def test_returns_presence_of_metadata(self, rendered_db_client, mock_db_client, exists):

        mock_db_client.object_exists.return_value = exists

        assert await rendered_db_client.rendered_document_exists('p1', _MD) is exists

        mock_db_client.object_exists.assert_awaited_once_with(
            path='rendered/p1/markdown/metadata.json'
        )

    @pytest.mark.asyncio
    async def test_raises_when_presence_check_fails(self, rendered_db_client, mock_db_client):

        mock_db_client.object_exists.side_effect = DBClientException('Access denied')

        with pytest.raises(
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            match='Failed to check the presence',
        ):
            await rendered_db_client.rendered_document_exists('p1', _MD)


class TestListRenderedDocuments:
    @pytest.mark.parametrize(
        ('paper_id', 'expected_prefix'), [(None, 'rendered/'), ('p1', 'rendered/p1/')]
    )
    @pytest.mark.asyncio
    async def test_returns_metadata_of_listed_renderings(
        self, rendered_db_client, mock_db_client, paper_id, expected_prefix
    ):

        metadata = RenderedDocumentMetadata(paper_id='p1', doc_format=_MD, size_bytes=10)
        mock_db_client.list_objects.return_value = [
            'rendered/p1/markdown/document.zip',
            'rendered/p1/markdown/metadata.json',
        ]
        mock_db_client.get_model.return_value = metadata

        assert await rendered_db_client.list_rendered_documents(paper_id) == [metadata]

        mock_db_client.list_objects.assert_awaited_once_with(prefix=expected_prefix)
        assert mock_db_client.get_model.await_args_list == [
            call(path='rendered/p1/markdown/metadata.json', model_type=RenderedDocumentMetadata)
        ]

    @pytest.mark.asyncio
    async def test_returns_empty_list(self, rendered_db_client, mock_db_client):

        mock_db_client.list_objects.return_value = []

        assert await rendered_db_client.list_rendered_documents() == []

    @pytest.mark.asyncio
    async def test_raises_on_list_failure(self, rendered_db_client, mock_db_client):

        mock_db_client.list_objects.side_effect = DBClientException('Access denied')

        with pytest.raises(
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            match='Failed to list the rendered',
        ):
            await rendered_db_client.list_rendered_documents()

    @pytest.mark.asyncio
    async def test_raises_on_retrieval_failure(self, rendered_db_client, mock_db_client):

        mock_db_client.list_objects.return_value = ['rendered/p1/markdown/metadata.json']
        mock_db_client.get_model.side_effect = DBClientException('Invalid JSON')

        with pytest.raises(
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            match='Failed to list the rendered',
        ):
            await rendered_db_client.list_rendered_documents()


class TestDownloadRenderedDocument:
    @pytest.mark.asyncio
    async def test_downloads_archive(self, rendered_db_client, mock_db_client, tmp_path):

        local_path = tmp_path / 'doc.zip'

        await rendered_db_client.download_rendered_document('p1', _MD, local_path)

        mock_db_client.download_to_file.assert_awaited_once_with(
            path='rendered/p1/markdown/document.zip', local_path=local_path
        )

    @pytest.mark.asyncio
    async def test_raises_when_download_fails(self, rendered_db_client, mock_db_client, tmp_path):

        mock_db_client.download_to_file.side_effect = DBClientException('Missing object')

        with pytest.raises(
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            match='Failed to download the markdown',
        ):
            await rendered_db_client.download_rendered_document('p1', _MD, tmp_path / 'doc.zip')

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(
        self, rendered_db_client, mock_db_client, tmp_path
    ):

        mock_db_client.download_to_file.side_effect = RuntimeError('Unexpected')

        with pytest.raises(RuntimeError, match='Unexpected'):
            await rendered_db_client.download_rendered_document('p1', _MD, tmp_path / 'doc.zip')
