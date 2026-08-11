import io
import json

import requests
from botocore.exceptions import ClientError as BotocoreClientError

import pytest
from unittest.mock import Mock, MagicMock

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.main_survey_enricher.components.doc_db_client import DocDBClient


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
        referenced_papers=[],
    )


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


@pytest.fixture
def mock_s3_backend():

    stored_objects: dict[str, bytes] = {}

    def mock_get_object(**kwargs):

        if kwargs['Key'] not in stored_objects:
            raise BotocoreClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')

        return {'Body': io.BytesIO(stored_objects[kwargs['Key']])}

    def mock_put_object(**kwargs):
        stored_objects[kwargs['Key']] = kwargs['Body']
        return {}

    mock_s3_client = MagicMock()
    mock_s3_client.get_object.side_effect = mock_get_object
    mock_s3_client.put_object.side_effect = mock_put_object

    return mock_s3_client, stored_objects


class TestDocDBClient:
    def test_add_survey_raises_when_download_request_fails(
        self, sample_document, patch_get_request_fails, mock_s3_backend
    ):

        mock_s3_client, _ = mock_s3_backend
        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        with pytest.raises(DocDBClient.DocDBClientError, match='Failed to download image from'):
            doc_db_client.add_survey(
                document=sample_document, name='Sample Survey', description='Sample Description'
            )

        assert mock_s3_client.upload_fileobj.call_count == 0

    def test_add_survey_raises_when_upload_fails(
        self, sample_document, patch_get_request, mock_s3_backend
    ):

        mock_s3_client, _ = mock_s3_backend
        mock_s3_client.upload_fileobj.side_effect = [None, Exception('Upload failed')]

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        with pytest.raises(
            DocDBClient.DocDBClientError, match='Failed to upload image to database'
        ):
            doc_db_client.add_survey(
                document=sample_document, name='Sample Survey', description='Sample Description'
            )

        assert mock_s3_client.upload_fileobj.call_count == 2

    def test_add_survey_successful(self, sample_document, patch_get_request, mock_s3_backend):

        mock_s3_client, stored_objects = mock_s3_backend

        doc_db_client = DocDBClient(s3_client=mock_s3_client)

        doc_db_client.add_survey(
            document=sample_document, name='Sample Survey', description='Sample Description'
        )

        assert mock_s3_client.upload_fileobj.call_count == 2

        stored_metadata = [
            json.loads(body.decode('utf-8'))
            for key, body in stored_objects.items()
            if key.endswith('metadata.json')
        ]

        assert len(stored_metadata) == 1
        assert stored_metadata[0]['doc_metadata']['name'] == 'Sample Survey'
        assert stored_metadata[0]['doc_metadata']['description'] == 'Sample Description'
        assert len(stored_metadata[0]['doc_metadata']['images']) == 2

        assert mock_s3_client.put_object.call_count == 2

        stored_documents = [
            json.loads(body.decode('utf-8'))
            for key, body in stored_objects.items()
            if key.endswith('document.json')
        ]

        assert len(stored_documents) == 1
        assert stored_documents[0] == sample_document.model_dump()
