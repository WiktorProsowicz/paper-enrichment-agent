import datetime

import requests

import pytest
from unittest.mock import MagicMock, Mock, call, create_autospec

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import (
    DocumentMetadata,
    PaperMetadata,
    SurveyMetadata,
)
from paper_enrichment_agent.main_survey_enricher.components.db_clients.core import (
    DBClient,
    DBClientException,
)
from paper_enrichment_agent.main_survey_enricher.components.db_clients.doc_db_client import (
    DocDBClient,
)

SAMPLE_PAPER_METADATA = PaperMetadata(
    doi='10.48550/arXiv.1706.03762',
    authors=['Ashish Vaswani', 'Noam Shazeer'],
    publication_date=datetime.date(2017, 6, 12),
)

_REQUESTS_GET_PATH = (
    'paper_enrichment_agent.main_survey_enricher.components.db_clients.doc_db_client.requests.get'
)


@pytest.fixture
def mock_db_client() -> MagicMock:

    return create_autospec(DBClient, instance=True)


@pytest.fixture
def doc_db_client(mock_db_client) -> DocDBClient:

    return DocDBClient(db_client=mock_db_client)


@pytest.fixture
def sample_document() -> doc_models.Document:

    return doc_models.Document(
        title='Sample Document',
        abstract='This is a sample abstract.',
        description='This is a sample description.',
        sections=[
            doc_models.Section(
                title='Introduction',
                components=[
                    doc_models.Paragraph(
                        elements=['This is a sample paragraph in the introduction section.']
                    ),
                    doc_models.Figure(
                        subfigures=[
                            doc_models.ImgSubfigure(
                                caption='Sample image caption.',
                                image_src='https://example.com/sample_image.png',
                            ),
                            doc_models.ImgSubfigure(
                                caption='Another sample image caption.',
                                image_src='https://example.com/another_sample_image.png',
                            ),
                        ],
                        caption='Sample figure caption.',
                    ),
                ],
            )
        ],
        footnotes=[],
        referenced_papers=[
            ('ref1', 'This is a sample referenced paper 1.'),
        ],
    )


@pytest.fixture
def sample_survey_metadatas() -> list[SurveyMetadata]:

    return [
        SurveyMetadata(paper_id='sample_paper_id_1', referenced_docs={}),
        SurveyMetadata(paper_id='sample_paper_id_2', referenced_docs={}),
    ]


def _mock_response(raise_for_status: Mock) -> MagicMock:

    response = MagicMock(
        raise_for_status=raise_for_status, raw=Mock(), headers={'Content-Type': 'image/png'}
    )
    response.__enter__.return_value = response
    return response


@pytest.fixture
def patch_get_request(monkeypatch):

    monkeypatch.setattr(_REQUESTS_GET_PATH, lambda url, **kwargs: _mock_response(Mock()))


@pytest.fixture
def patch_get_request_fails(monkeypatch):

    def mock_get(url, **kwargs):
        return _mock_response(Mock(side_effect=requests.RequestException('Request failed')))

    monkeypatch.setattr(_REQUESTS_GET_PATH, mock_get)


@pytest.fixture
def patch_get_request_connection_fails(monkeypatch):

    def mock_get(url, **kwargs):
        raise requests.ConnectionError('Connection refused')

    monkeypatch.setattr(_REQUESTS_GET_PATH, mock_get)


class TestDocumentExists:
    @pytest.mark.parametrize('exists', [True, False])
    @pytest.mark.asyncio
    async def test_returns_presence_of_metadata(self, doc_db_client, mock_db_client, exists):

        mock_db_client.object_exists.return_value = exists

        assert await doc_db_client.document_exists(paper_id='sample_paper_id') is exists

        mock_db_client.object_exists.assert_awaited_once_with(
            path='documents/sample_paper_id/metadata.json'
        )

    @pytest.mark.asyncio
    async def test_raises_when_presence_check_fails(self, doc_db_client, mock_db_client):

        mock_db_client.object_exists.side_effect = DBClientException('Access denied')

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to check the presence of the document'
        ):
            await doc_db_client.document_exists(paper_id='sample_paper_id')


class TestAddDocument:
    @pytest.mark.asyncio
    async def test_stores_images_and_models(
        self, doc_db_client, mock_db_client, sample_document, patch_get_request
    ):

        doc_meta = await doc_db_client.add_document(
            document=sample_document,
            name='Sample Document',
            description='Sample Description',
            paper_metadata=SAMPLE_PAPER_METADATA,
        )

        assert doc_meta.name == 'Sample Document'
        assert doc_meta.description == 'Sample Description'
        assert doc_meta.paper_metadata == SAMPLE_PAPER_METADATA

        image_paths = [c.kwargs['path'] for c in mock_db_client.write_stream.await_args_list]
        assert len(image_paths) == 2
        assert all(
            path.startswith(f'documents/{doc_meta.paper_id}/images/') and path.endswith('.png')
            for path in image_paths
        )
        assert sorted(doc_meta.images.values()) == sorted(image_paths)
        assert all(
            c.kwargs['content_type'] == 'image/png'
            for c in mock_db_client.write_stream.await_args_list
        )

        assert mock_db_client.upload_model.await_args_list == [
            call(doc_meta, path=f'documents/{doc_meta.paper_id}/metadata.json'),
            call(sample_document, path=f'documents/{doc_meta.paper_id}/document.json'),
        ]

    @pytest.mark.asyncio
    async def test_raises_when_download_request_fails(
        self, doc_db_client, mock_db_client, sample_document, patch_get_request_fails
    ):

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to download image from'):
            await doc_db_client.add_document(
                document=sample_document,
                name='Sample Document',
                description='Sample Description',
                paper_metadata=SAMPLE_PAPER_METADATA,
            )

        mock_db_client.write_stream.assert_not_awaited()
        mock_db_client.upload_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_connection_fails(
        self, doc_db_client, mock_db_client, sample_document, patch_get_request_connection_fails
    ):

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to download image from'):
            await doc_db_client.add_document(
                document=sample_document,
                name='Sample Document',
                description='Sample Description',
                paper_metadata=SAMPLE_PAPER_METADATA,
            )

        mock_db_client.write_stream.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_image_upload_fails(
        self, doc_db_client, mock_db_client, sample_document, patch_get_request
    ):

        mock_db_client.write_stream.side_effect = [None, DBClientException('Upload failed')]

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to upload image to database'
        ):
            await doc_db_client.add_document(
                document=sample_document,
                name='Sample Document',
                description='Sample Description',
                paper_metadata=SAMPLE_PAPER_METADATA,
            )

        assert mock_db_client.write_stream.await_count == 2
        mock_db_client.upload_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_model_upload_fails(
        self, doc_db_client, mock_db_client, sample_document, patch_get_request
    ):

        mock_db_client.upload_model.side_effect = DBClientException('Upload failed')

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to add the document'):
            await doc_db_client.add_document(
                document=sample_document,
                name='Sample Document',
                description='Sample Description',
                paper_metadata=SAMPLE_PAPER_METADATA,
            )

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(
        self, doc_db_client, mock_db_client, sample_document, patch_get_request
    ):

        mock_db_client.write_stream.side_effect = RuntimeError('Unexpected')

        with pytest.raises(RuntimeError, match='Unexpected'):
            await doc_db_client.add_document(
                document=sample_document,
                name='Sample Document',
                description='Sample Description',
                paper_metadata=SAMPLE_PAPER_METADATA,
            )


def _document_with_image(image_src: str) -> doc_models.Document:

    return doc_models.Document(
        title='PDF Document',
        abstract='',
        description=None,
        sections=[
            doc_models.Section(
                component_id='intro',
                title='Introduction',
                components=[
                    doc_models.Figure(
                        component_id='figure',
                        subfigures=[
                            doc_models.ImgSubfigure(
                                component_id='image', caption=None, image_src=image_src
                            )
                        ],
                        caption='Figure 1.',
                    )
                ],
            )
        ],
        footnotes=[],
        referenced_papers=[],
    )


class TestAddDocumentWithDataUriImages:
    @pytest.mark.asyncio
    async def test_stores_decoded_image_and_rewrites_source(
        self, doc_db_client, mock_db_client, monkeypatch
    ):

        get_mock = Mock()
        monkeypatch.setattr(_REQUESTS_GET_PATH, get_mock)
        uploaded_contents = []

        async def record_write_stream(path, stream, content_type):
            uploaded_contents.append(stream.read())

        mock_db_client.write_stream.side_effect = record_write_stream
        document = _document_with_image('data:image/jpeg;base64,/9j/2Q==')

        doc_meta = await doc_db_client.add_document(
            document=document,
            name='PDF Document',
            description='From PDF',
            paper_metadata=SAMPLE_PAPER_METADATA,
        )

        get_mock.assert_not_called()
        (write_call,) = mock_db_client.write_stream.await_args_list
        image_path = write_call.kwargs['path']
        assert image_path.startswith(f'documents/{doc_meta.paper_id}/images/')
        assert image_path.endswith('.jpeg')
        assert write_call.kwargs['content_type'] == 'image/jpeg'
        assert uploaded_contents == [b'\xff\xd8\xff\xd9']
        assert doc_meta.images == {'/sections/intro/figure/image': image_path}

        stored_document = mock_db_client.upload_model.await_args_list[1].args[0]
        assert stored_document.sections[0].components[0].subfigures[0].image_src == image_path

    @pytest.mark.parametrize(
        'image_src',
        ['data:text/plain;base64,YQ==', 'data:image/png,raw', 'data:image/png;base64,!!!'],
    )
    @pytest.mark.asyncio
    async def test_raises_on_invalid_data_uri(self, doc_db_client, mock_db_client, image_src):

        with pytest.raises(DocDBClient.DocDBClientError, match='data URI'):
            await doc_db_client.add_document(
                document=_document_with_image(image_src),
                name='PDF',
                description='PDF',
                paper_metadata=SAMPLE_PAPER_METADATA,
            )

        mock_db_client.write_stream.assert_not_awaited()
        mock_db_client.upload_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_image_upload_fails(self, doc_db_client, mock_db_client):

        mock_db_client.write_stream.side_effect = DBClientException('Upload failed')

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to upload image to database'
        ):
            await doc_db_client.add_document(
                document=_document_with_image('data:image/png;base64,iVBORw0KGgo='),
                name='PDF',
                description='PDF',
                paper_metadata=SAMPLE_PAPER_METADATA,
            )


class TestRegisterAsSurvey:
    @pytest.mark.asyncio
    async def test_uploads_survey_metadata(self, doc_db_client, mock_db_client):

        survey_meta = await doc_db_client.register_as_survey(paper_id='sample_paper_id')

        assert survey_meta == SurveyMetadata(paper_id='sample_paper_id', referenced_docs={})
        mock_db_client.upload_model.assert_awaited_once_with(
            survey_meta, path='surveys/sample_paper_id/metadata.json'
        )

    @pytest.mark.asyncio
    async def test_raises_when_upload_fails(self, doc_db_client, mock_db_client):

        mock_db_client.upload_model.side_effect = DBClientException('Upload failed')

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to register the document sample_paper_id'
        ):
            await doc_db_client.register_as_survey(paper_id='sample_paper_id')


class TestGetAvailableSurveys:
    @pytest.mark.asyncio
    async def test_returns_empty_list(self, doc_db_client, mock_db_client):

        mock_db_client.list_subdirectories.return_value = []

        assert await doc_db_client.get_available_surveys() == []

        mock_db_client.list_subdirectories.assert_awaited_once_with(prefix='surveys/')
        mock_db_client.get_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_returns_surveys(self, doc_db_client, mock_db_client, sample_survey_metadatas):

        mock_db_client.list_subdirectories.return_value = [
            'surveys/sample_paper_id_1/',
            'surveys/sample_paper_id_2/',
        ]
        mock_db_client.get_model.side_effect = sample_survey_metadatas

        assert await doc_db_client.get_available_surveys() == sample_survey_metadatas

        assert mock_db_client.get_model.await_args_list == [
            call(path='surveys/sample_paper_id_1/metadata.json', model_type=SurveyMetadata),
            call(path='surveys/sample_paper_id_2/metadata.json', model_type=SurveyMetadata),
        ]

    @pytest.mark.asyncio
    async def test_raises_on_list_failure(self, doc_db_client, mock_db_client):

        mock_db_client.list_subdirectories.side_effect = DBClientException('Access denied')

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to list surveys in database'
        ):
            await doc_db_client.get_available_surveys()

    @pytest.mark.asyncio
    async def test_raises_on_retrieval_failure(self, doc_db_client, mock_db_client):

        mock_db_client.list_subdirectories.return_value = ['surveys/sample_paper_id_1/']
        mock_db_client.get_model.side_effect = DBClientException('Invalid JSON')

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to retrieve JSON from database at'
        ):
            await doc_db_client.get_available_surveys()


class TestGetSurveyInfo:
    @pytest.mark.asyncio
    async def test_returns_metadata(self, doc_db_client, mock_db_client, sample_survey_metadatas):

        mock_db_client.get_model.return_value = sample_survey_metadatas[0]

        assert (
            await doc_db_client.get_survey_info('sample_paper_id_1') == sample_survey_metadatas[0]
        )

        mock_db_client.get_model.assert_awaited_once_with(
            path='surveys/sample_paper_id_1/metadata.json', model_type=SurveyMetadata
        )

    @pytest.mark.asyncio
    async def test_raises_on_retrieval_failure(self, doc_db_client, mock_db_client):

        mock_db_client.get_model.side_effect = DBClientException('Missing object')

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to retrieve JSON from database at'
        ):
            await doc_db_client.get_survey_info('sample_paper_id')


class TestAddReferencedDocument:
    @pytest.mark.asyncio
    async def test_modifies_survey_reference(self, doc_db_client, mock_db_client):

        survey = SurveyMetadata(paper_id='survey_id', referenced_docs={})
        mock_db_client.model_reference.return_value.__aenter__.return_value = survey

        await doc_db_client.add_referenced_document(
            survey_id='survey_id', reference_id='ref1', referenced_paper_id='ref_id'
        )

        mock_db_client.model_reference.assert_called_once_with(
            path='surveys/survey_id/metadata.json', model_type=SurveyMetadata
        )
        assert survey.referenced_docs == {'ref1': 'ref_id'}

    @pytest.mark.asyncio
    async def test_raises_when_synchronization_fails(self, doc_db_client, mock_db_client):

        mock_db_client.model_reference.return_value.__aenter__.side_effect = DBClientException(
            'Missing object'
        )

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to add the referenced document'
        ):
            await doc_db_client.add_referenced_document(
                survey_id='survey_id', reference_id='ref1', referenced_paper_id='ref_id'
            )


class TestDeleteSurvey:
    @pytest.mark.asyncio
    async def test_deletes_survey_and_referenced_documents(self, doc_db_client, mock_db_client):

        mock_db_client.get_model.return_value = SurveyMetadata(
            paper_id='survey_id', referenced_docs={'ref1': 'ref_id'}
        )
        mock_db_client.list_objects.side_effect = [
            ['documents/ref_id/images/image1.png', 'documents/ref_id/images/image2.png'],
            ['documents/survey_id/images/image3.png'],
        ]

        await doc_db_client.delete_survey('survey_id')

        mock_db_client.get_model.assert_awaited_once_with(
            path='surveys/survey_id/metadata.json', model_type=SurveyMetadata
        )
        assert mock_db_client.list_objects.await_args_list == [
            call(prefix='documents/ref_id/images/'),
            call(prefix='documents/survey_id/images/'),
        ]
        deleted_paths = [c.kwargs['path'] for c in mock_db_client.delete_object.await_args_list]
        assert deleted_paths == [
            'documents/ref_id/metadata.json',
            'documents/ref_id/document.json',
            'documents/ref_id/images/image1.png',
            'documents/ref_id/images/image2.png',
            'surveys/survey_id/metadata.json',
            'documents/survey_id/metadata.json',
            'documents/survey_id/document.json',
            'documents/survey_id/images/image3.png',
        ]

    @pytest.mark.asyncio
    async def test_raises_when_document_deletion_fails(self, doc_db_client, mock_db_client):

        mock_db_client.get_model.return_value = SurveyMetadata(
            paper_id='survey_id', referenced_docs={'ref1': 'ref_id'}
        )
        mock_db_client.delete_object.side_effect = DBClientException('Access denied')

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to delete document ref_id'):
            await doc_db_client.delete_survey('survey_id')

    @pytest.mark.asyncio
    async def test_raises_when_survey_metadata_deletion_fails(self, doc_db_client, mock_db_client):

        mock_db_client.get_model.return_value = SurveyMetadata(
            paper_id='survey_id', referenced_docs={}
        )
        mock_db_client.delete_object.side_effect = DBClientException('Access denied')

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to delete survey survey_id'):
            await doc_db_client.delete_survey('survey_id')

    @pytest.mark.asyncio
    async def test_raises_when_survey_retrieval_fails(self, doc_db_client, mock_db_client):

        mock_db_client.get_model.side_effect = DBClientException('Missing object')

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to retrieve JSON from database at'
        ):
            await doc_db_client.delete_survey('survey_id')

        mock_db_client.delete_object.assert_not_awaited()


class TestGetDocumentStructRef:
    @pytest.mark.asyncio
    async def test_yields_document_reference(self, doc_db_client, mock_db_client, sample_document):

        mock_db_client.model_reference.return_value.__aenter__.return_value = sample_document

        async with doc_db_client.get_document_struct_ref('sample_paper_id') as doc_struct_ref:
            assert doc_struct_ref is sample_document

        mock_db_client.model_reference.assert_called_once_with(
            path='documents/sample_paper_id/document.json', model_type=doc_models.Document
        )
        mock_db_client.model_reference.return_value.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_when_synchronization_fails(self, doc_db_client, mock_db_client):

        mock_db_client.model_reference.return_value.__aenter__.side_effect = DBClientException(
            'Missing object'
        )

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to synchronize the structure of document'
        ):
            async with doc_db_client.get_document_struct_ref('sample_paper_id'):
                pass


class TestGetDocumentMetadata:
    @pytest.mark.asyncio
    async def test_returns_metadata(self, doc_db_client, mock_db_client):

        metadata = DocumentMetadata(
            paper_id='sample_paper_id',
            name='Sample Document',
            description='Sample Description',
            images={'/sections/0/1/0': 'documents/sample_paper_id/images/image1.png'},
            paper_metadata=SAMPLE_PAPER_METADATA,
        )
        mock_db_client.get_model.return_value = metadata

        assert await doc_db_client.get_document_metadata('sample_paper_id') == metadata

        mock_db_client.get_model.assert_awaited_once_with(
            path='documents/sample_paper_id/metadata.json', model_type=DocumentMetadata
        )


class TestDownloadImage:
    @pytest.mark.asyncio
    async def test_downloads_image_to_local_file(self, doc_db_client, mock_db_client, tmp_path):

        local_path = tmp_path / 'image.png'

        await doc_db_client.download_image('documents/p/images/img.png', local_path)

        mock_db_client.download_to_file.assert_awaited_once_with(
            path='documents/p/images/img.png', local_path=local_path
        )

    @pytest.mark.asyncio
    async def test_raises_when_download_fails(self, doc_db_client, mock_db_client, tmp_path):

        mock_db_client.download_to_file.side_effect = DBClientException('Missing object')

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to download the image'):
            await doc_db_client.download_image('documents/p/images/img.png', tmp_path / 'i.png')

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(
        self, doc_db_client, mock_db_client, tmp_path
    ):

        mock_db_client.download_to_file.side_effect = RuntimeError('Unexpected')

        with pytest.raises(RuntimeError, match='Unexpected'):
            await doc_db_client.download_image('documents/p/images/img.png', tmp_path / 'i.png')
