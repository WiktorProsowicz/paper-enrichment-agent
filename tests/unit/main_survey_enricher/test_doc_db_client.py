import io
import json

import requests
from botocore.exceptions import ClientError as BotocoreClientError

import pytest
from unittest.mock import Mock, MagicMock

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.main_survey_enricher.components.doc_db_client import DocDBClient
from paper_enrichment_agent.common.models.misc import SurveyMetadata


@pytest.fixture
def sample_document() -> doc_models.Document:

    return doc_models.Document(
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


@pytest.fixture
def patch_get_request(monkeypatch):

    def mock_get(url, **kwargs):
        response = MagicMock(
            raise_for_status=Mock(),
            raw=Mock(),
        )
        response.__enter__.return_value = response
        return response

    monkeypatch.setattr(
        'paper_enrichment_agent.main_survey_enricher.components.doc_db_client.requests.get',
        mock_get,
    )


@pytest.fixture
def patch_get_request_fails(monkeypatch):

    def mock_get(url, **kwargs):
        response = MagicMock(
            raise_for_status=Mock(side_effect=requests.RequestException('Request failed')),
            raw=Mock(),
        )
        response.__enter__.return_value = response
        return response

    monkeypatch.setattr(
        'paper_enrichment_agent.main_survey_enricher.components.doc_db_client.requests.get',
        mock_get,
    )


class TestDocDBClient:
    def test_document_exists_returns_true_for_present_document(self):

        mock_s3_client = MagicMock()

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        assert doc_db_client.document_exists(paper_id='sample_paper_id') is True

        mock_s3_client.head_object.assert_called_once_with(
            Bucket='document_database', Key='documents/sample_paper_id/metadata.json'
        )

    def test_document_exists_returns_false_for_missing_document(self):

        mock_s3_client = MagicMock()
        mock_s3_client.head_object.side_effect = BotocoreClientError(
            {'Error': {'Code': '404'}}, 'HeadObject'
        )

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        assert doc_db_client.document_exists(paper_id='sample_paper_id') is False

    def test_document_exists_raises_when_presence_check_fails(self):

        mock_s3_client = MagicMock()
        mock_s3_client.head_object.side_effect = BotocoreClientError(
            {'Error': {'Code': 'AccessDenied'}}, 'HeadObject'
        )

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to check the presence of the document'
        ):
            doc_db_client.document_exists(paper_id='sample_paper_id')

    def test_add_document_raises_when_download_request_fails(
        self, sample_document, patch_get_request_fails
    ):

        mock_s3_client = MagicMock()
        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to download image from'):
            doc_db_client.add_document(
                document=sample_document, name='Sample Document', description='Sample Description'
            )

        assert mock_s3_client.upload_fileobj.call_count == 0

    def test_add_document_raises_when_upload_fails(self, sample_document, patch_get_request):

        mock_s3_client = MagicMock()
        mock_s3_client.upload_fileobj.side_effect = [None, Exception('Upload failed')]

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to upload image to database'
        ):
            doc_db_client.add_document(
                document=sample_document, name='Sample Document', description='Sample Description'
            )

        assert mock_s3_client.upload_fileobj.call_count == 2

    def test_add_document_successful(self, sample_document, patch_get_request):

        mock_s3_client = MagicMock()

        doc_db_client = DocDBClient(s3_client=mock_s3_client)
        doc_meta = doc_db_client.add_document(
            document=sample_document, name='Sample Document', description='Sample Description'
        )

        assert mock_s3_client.upload_fileobj.call_count == 2
        assert len(doc_meta.images) == 2

        uploaded_paths = [call.kwargs['Key'] for call in mock_s3_client.put_object.call_args_list]
        assert uploaded_paths == [
            f'documents/{doc_meta.paper_id}/metadata.json',
            f'documents/{doc_meta.paper_id}/document.json',
        ]

    def test_register_as_survey_successful(self):

        mock_s3_client = MagicMock()

        doc_db_client = DocDBClient(s3_client=mock_s3_client)
        survey_meta = doc_db_client.register_as_survey(paper_id='sample_paper_id')

        assert survey_meta.paper_id == 'sample_paper_id'
        assert survey_meta.referenced_docs == {}

        assert (
            mock_s3_client.put_object.call_args.kwargs['Key']
            == 'surveys/sample_paper_id/metadata.json'
        )

    def test_get_available_surveys_returns_empty_list(self):

        mock_s3_client = MagicMock()
        mock_s3_client.list_objects_v2.return_value = {'CommonPrefixes': []}

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        assert doc_db_client.get_available_surveys() == []
        assert mock_s3_client.get_object.call_count == 0

    def test_get_available_surveys_returns_surveys(self, sample_survey_metadatas):

        def mock_list_objects_v2(**kwargs):
            yield {
                'CommonPrefixes': [
                    {'Prefix': 'surveys/sample_paper_id/'},
                    {'Prefix': 'surveys/another_paper_id/'},
                ]
            }

        def mock_get_object(**kwargs):

            yield {
                'Body': io.BytesIO(
                    json.dumps(sample_survey_metadatas[0].model_dump()).encode('utf-8')
                )
            }

            yield {
                'Body': io.BytesIO(
                    json.dumps(sample_survey_metadatas[1].model_dump()).encode('utf-8')
                )
            }

        mock_s3_client = MagicMock()
        mock_s3_client.list_objects_v2.side_effect = mock_list_objects_v2()
        mock_s3_client.get_object.side_effect = mock_get_object()

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        assert len(doc_db_client.get_available_surveys()) == 2

    def test_get_available_surveys_raises_on_list_failure(self):

        mock_s3_client = MagicMock()
        mock_s3_client.list_objects_v2.side_effect = BotocoreClientError(
            {'Error': {'Code': 'AccessDenied'}}, 'ListObjectsV2'
        )

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to list surveys in database'
        ):
            doc_db_client.get_available_surveys()

    def test_add_referenced_document_successful(self):

        def mock_get_object(**kwargs):
            yield {
                'Body': io.BytesIO(
                    json.dumps(
                        SurveyMetadata(paper_id='survey_id', referenced_docs={}).model_dump()
                    ).encode('utf-8')
                )
            }

        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = mock_get_object()

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        doc_db_client.add_referenced_document(
            survey_id='survey_id', reference_id='ref1', referenced_paper_id='ref_id'
        )

        mock_s3_client.get_object.assert_called_once_with(
            Bucket='document_database', Key='surveys/survey_id/metadata.json'
        )
        mock_s3_client.put_object.assert_called_once_with(
            Bucket='document_database',
            Key='surveys/survey_id/metadata.json',
            Body=json.dumps(
                SurveyMetadata(
                    paper_id='survey_id', referenced_docs={'ref1': 'ref_id'}
                ).model_dump()
            ).encode('utf-8'),
        )

    def test_delete_survey_successful(self):

        def mock_get_object(**kwargs):
            yield {
                'Body': io.BytesIO(
                    json.dumps(
                        SurveyMetadata(
                            paper_id='survey_id', referenced_docs={'ref1': 'ref_id'}
                        ).model_dump()
                    ).encode('utf-8')
                )
            }

        def mock_list_objects_v2(**kwargs):
            yield {
                'Contents': [
                    {'Key': 'documents/ref_id/images/image1.png'},
                    {'Key': 'documents/ref_id/images/image2.png'},
                ]
            }

            yield {'Contents': [{'Key': 'documents/survey_id/images/image3.png'}]}

        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = mock_get_object()
        mock_s3_client.list_objects_v2.side_effect = mock_list_objects_v2()

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        doc_db_client.delete_survey('survey_id')

        deleted_keys = [call.kwargs['Key'] for call in mock_s3_client.delete_object.call_args_list]
        assert deleted_keys == [
            'documents/ref_id/metadata.json',
            'documents/ref_id/document.json',
            'documents/ref_id/images/image1.png',
            'documents/ref_id/images/image2.png',
            'surveys/survey_id/metadata.json',
            'documents/survey_id/metadata.json',
            'documents/survey_id/document.json',
            'documents/survey_id/images/image3.png',
        ]

    def test_get_document_struct_ref_synchronizes_properly(self, sample_document):

        def mock_get_object(**kwargs):

            yield {'Body': io.BytesIO(json.dumps(sample_document.model_dump()).encode('utf-8'))}

        mock_s3_client = MagicMock()
        mock_s3_client.get_object.side_effect = mock_get_object()

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        with doc_db_client.get_document_struct_ref('sample_paper_id') as doc_struct_ref:
            doc_struct_ref.sections.clear()

        assert (
            mock_s3_client.put_object.call_args_list[0].kwargs['Key']
            == 'documents/sample_paper_id/document.json'
        )
        assert (
            json.loads(mock_s3_client.put_object.call_args_list[0].kwargs['Body'].decode('utf-8'))[
                'sections'
            ]
            == []
        )
