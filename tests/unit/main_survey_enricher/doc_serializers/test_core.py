import pathlib
import tempfile

import pytest

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.core import (
    SerializationEnvironment,
)


@pytest.fixture(scope='session')
def sample_document() -> doc_models.Document:

    return doc_models.Document(
        title='Sample Document',
        abstract='This is a sample abstract.',
        description=None,
        sections=[
            doc_models.Section(
                component_id='S1',
                title='Introduction',
                components=[
                    doc_models.Paragraph(component_id='P1', elements=['Sample paragraph.']),
                    doc_models.Figure(
                        component_id='F1',
                        subfigures=[
                            doc_models.ImgSubfigure(
                                component_id='img',
                                image_src='https://example.com/first.png',
                                caption=None,
                            ),
                        ],
                        caption='First figure.',
                    ),
                    doc_models.Figure(
                        component_id='F2',
                        subfigures=[
                            doc_models.ImgSubfigure(
                                component_id='img',
                                image_src='https://example.com/second.jpg',
                                caption=None,
                            ),
                            doc_models.TableSubfigure(
                                component_id='tab',
                                table_contents='<table></table>',
                                caption=None,
                            ),
                        ],
                        caption='Second figure.',
                    ),
                ],
            )
        ],
        footnotes=[],
        referenced_papers=[],
    )


@pytest.fixture
def work_dir():

    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


class TestCreate:
    def test_create_prepares_directory_structure(self, sample_document, work_dir):

        environment = SerializationEnvironment.create(sample_document, work_dir)

        assert environment.document is sample_document
        assert environment.output_dir.parent == work_dir
        assert environment.archive_path.parent == work_dir
        assert (environment.output_dir / 'images').is_dir()
        assert not environment.archive_path.exists()

    def test_create_creates_missing_work_dir(self, sample_document, work_dir):

        environment = SerializationEnvironment.create(sample_document, work_dir / 'nested')

        assert (work_dir / 'nested' / 'content' / 'images').is_dir()
        assert environment.output_dir == work_dir / 'nested' / 'content'

    def test_create_raises_when_work_dir_is_a_file(self, sample_document, work_dir):

        file_path = work_dir / 'file'
        file_path.write_text('content')

        with pytest.raises(
            SerializationEnvironment.SerializationEnvironmentError,
            match='Failed to prepare the serialization environment',
        ):
            SerializationEnvironment.create(sample_document, file_path)

        assert file_path.read_text() == 'content'


class TestImageComponentPaths:
    def test_image_component_paths_lists_images_in_document_order(self, sample_document, work_dir):

        environment = SerializationEnvironment.create(sample_document, work_dir)

        assert environment.image_component_paths == [
            '/sections/S1/F1/img',
            '/sections/S1/F2/img',
        ]


class TestImagePathForComponent:
    def test_image_path_for_component_returns_unique_paths_in_images_dir(
        self, sample_document, work_dir
    ):

        environment = SerializationEnvironment.create(sample_document, work_dir)

        first_path = environment.image_path_for_component('/sections/S1/F1/img')
        second_path = environment.image_path_for_component('/sections/S1/F2/img')

        assert first_path == environment.output_dir / 'images' / 'sections-S1-F1-img.png'
        assert second_path == environment.output_dir / 'images' / 'sections-S1-F2-img.jpg'

    def test_image_path_for_component_does_not_create_files(self, sample_document, work_dir):

        environment = SerializationEnvironment.create(sample_document, work_dir)

        assert not environment.image_path_for_component('/sections/S1/F1/img').exists()

    @pytest.mark.parametrize(
        'component_path', ['/sections/S1/P1', '/sections/S1/F2', '/sections/S1/F2/tab']
    )
    def test_image_path_for_component_raises_on_non_image_path(
        self, sample_document, work_dir, component_path
    ):

        environment = SerializationEnvironment.create(sample_document, work_dir)

        with pytest.raises(ValueError, match='is not an image subfigure'):
            environment.image_path_for_component(component_path)

    def test_image_path_for_component_raises_on_unknown_path(self, sample_document, work_dir):

        environment = SerializationEnvironment.create(sample_document, work_dir)

        with pytest.raises(ValueError, match='Component not found in the document'):
            environment.image_path_for_component('/sections/S1/F3/img')
