import pytest
from unittest.mock import Mock

from paper_enrichment_agent.common.models import document as doc_models


@pytest.fixture
def sample_document() -> doc_models.Document:

    return doc_models.Document(
        abstract='This is a sample abstract.',
        sections=[
            doc_models.Section(
                title='Introduction',
                components=[
                    doc_models.Paragraph(
                        text='This is a sample paragraph in the introduction section.'
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
    )


@pytest.fixture
def patch_dependencies(monkeypatch):

    monkeypatch.setattr(
        'paper_enrichment_agent.main_survey_enricher.components.doc_db_client.requests.get',
        lambda url, **kwargs: Mock(raise_for_status=Mock(), raw=Mock()),
    )

    return Mock(
        get_object=Mock(),
        put_object=Mock(),
        upload_fileobj=Mock(),
    )


# class TestDocDBClient:

# def test_
