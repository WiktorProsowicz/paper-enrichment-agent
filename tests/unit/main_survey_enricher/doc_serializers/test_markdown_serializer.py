import pathlib
import tempfile
import zipfile

import pytest

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.core import (
    SerializationEnvironment,
)
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.markdown import (
    MarkdownSerializer,
)

SAMPLE_IMAGE_BYTES = b'sample image bytes'


@pytest.fixture(scope='session')
def sample_document() -> doc_models.Document:

    return doc_models.Document(
        title='Sample Survey',
        abstract='This is a sample abstract.',
        description=None,
        sections=[
            doc_models.Section(
                component_id='S1',
                title='Introduction',
                components=[
                    doc_models.Paragraph(
                        component_id='P1',
                        elements=[
                            'See ',
                            doc_models.Reference(
                                component_id='r1',
                                ref_type='link',
                                target='https://example.com',
                                content_text='the website',
                            ),
                            ' and ',
                            doc_models.Reference(
                                component_id='r2',
                                ref_type='citation',
                                target='bib1',
                                content_text='Smith 2020',
                            ),
                            doc_models.Reference(
                                component_id='r3',
                                ref_type='citation',
                                target='bib2',
                                content_text='Doe 2021',
                            ),
                            ' with ',
                            doc_models.MathExpression(
                                component_id='m1', expression='x^2', format='LaTeX'
                            ),
                            ' as shown in ',
                            doc_models.Reference(
                                component_id='r4',
                                ref_type='element',
                                target='/sections/S1/F1',
                                content_text='Figure 1',
                            ),
                            '.',
                            doc_models.Reference(
                                component_id='r5',
                                ref_type='element',
                                target='/footnotes/fn1',
                                content_text='1',
                            ),
                        ],
                    ),
                    doc_models.Figure(
                        component_id='F1',
                        subfigures=[
                            doc_models.ImgSubfigure(
                                component_id='img1',
                                image_src='https://example.com/image.png',
                                caption='Image caption.',
                            ),
                            doc_models.TableSubfigure(
                                component_id='tab1',
                                table_contents='<table><tr><td>1</td></tr></table>',
                                caption=None,
                            ),
                        ],
                        caption=doc_models.Paragraph(
                            component_id='cap',
                            elements=[
                                'Figure with ',
                                doc_models.MathExpression(
                                    component_id='m2', expression='y', format='LaTeX'
                                ),
                                ' and ',
                                doc_models.Reference(
                                    component_id='r6',
                                    ref_type='element',
                                    target='/sections/S1/E1',
                                    content_text='Equation 1',
                                ),
                            ],
                        ),
                    ),
                    doc_models.MathExpression(
                        component_id='E1', expression='E = mc^2', format='LaTeX'
                    ),
                    doc_models.List(
                        component_id='L1',
                        ordered=True,
                        items=[
                            doc_models.Paragraph(component_id='i1', elements=['First item']),
                            doc_models.Paragraph(component_id='i2', elements=['Second item']),
                        ],
                    ),
                    doc_models.Section(
                        component_id='S1.1',
                        title='Details',
                        components=[
                            doc_models.Paragraph(component_id='P2', elements=['Nested paragraph.'])
                        ],
                    ),
                ],
            )
        ],
        footnotes=[
            doc_models.Paragraph(component_id='fn1', elements=['Footnote text.']),
            doc_models.Section(
                component_id='fn2',
                title='Enriched footnote',
                components=[
                    doc_models.Paragraph(component_id='P3', elements=['First enriched paragraph.']),
                    doc_models.Paragraph(
                        component_id='P4', elements=['Second enriched paragraph.']
                    ),
                ],
            ),
        ],
        referenced_papers=[
            ('bib1', 'Smith. Paper A. 2020.'),
            ('bib2', 'Doe. Paper B. 2021.'),
        ],
    )


@pytest.fixture
def work_dir():

    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


@pytest.fixture
def sample_environment(sample_document, work_dir) -> SerializationEnvironment:

    environment = SerializationEnvironment.create(sample_document, work_dir)

    for component_path in environment.image_component_paths:
        environment.image_path_for_component(component_path).write_bytes(SAMPLE_IMAGE_BYTES)

    return environment


@pytest.fixture
def serialized_archive(sample_environment) -> pathlib.Path:

    return MarkdownSerializer(sample_environment).serialize()


@pytest.fixture
def rendered_markdown(sample_environment, serialized_archive) -> str:

    return (sample_environment.output_dir / 'paper.md').read_text(encoding='utf-8')


class TestSerialize:
    def test_serialize_creates_archive_with_markdown_and_images(
        self, sample_environment, serialized_archive, rendered_markdown
    ):

        assert serialized_archive == sample_environment.archive_path

        with zipfile.ZipFile(serialized_archive) as archive:
            assert sorted(archive.namelist()) == [
                'images/sections-S1-F1-img1.png',
                'paper.md',
            ]
            assert archive.read('images/sections-S1-F1-img1.png') == SAMPLE_IMAGE_BYTES
            assert archive.read('paper.md').decode('utf-8') == rendered_markdown

    def test_serialize_renders_title_and_abstract(self, rendered_markdown):

        assert rendered_markdown.startswith(
            '# Sample Survey\n\n## Abstract\n\nThis is a sample abstract.\n\n'
        )

    def test_serialize_renders_nested_section_headings(self, rendered_markdown):

        assert '\n## Introduction\n' in rendered_markdown
        assert '\n### Details\n\nNested paragraph.\n' in rendered_markdown

    def test_serialize_renders_inline_elements(self, rendered_markdown):

        assert (
            'See [the website](https://example.com) and '
            '[[Smith 2020](#cite-bib1), [Doe 2021](#cite-bib2)] with $x^2$ as shown in '
            '[Figure 1](#sections-S1-F1).[^1]'
        ) in rendered_markdown

    def test_serialize_renders_figure_with_anchor(self, rendered_markdown):

        assert (
            '<a id="sections-S1-F1"></a>\n'
            '![img1](images/sections-S1-F1-img1.png)\n\n'
            '*Image caption.*\n\n'
            '<table><tr><td>1</td></tr></table>\n\n'
            '*Figure with $y$ and [Equation 1](#sections-S1-E1)*'
        ) in rendered_markdown

    def test_serialize_renders_block_math_with_anchor_from_caption_reference(
        self, rendered_markdown
    ):

        assert '<a id="sections-S1-E1"></a>\n$$\nE = mc^2\n$$' in rendered_markdown

    def test_serialize_renders_ordered_list(self, rendered_markdown):

        assert '\n1. First item\n2. Second item\n' in rendered_markdown

    def test_serialize_does_not_anchor_unreferenced_components(self, rendered_markdown):

        assert rendered_markdown.count('<a id="sections-') == 2

    def test_serialize_renders_footnotes(self, rendered_markdown):

        assert '## Footnotes\n\n[^1]: Footnote text.\n\n' in rendered_markdown
        assert (
            '[^2]: **Enriched footnote**\n\n'
            '    First enriched paragraph.\n\n'
            '    Second enriched paragraph.\n'
        ) in rendered_markdown

    def test_serialize_renders_references(self, rendered_markdown):

        assert rendered_markdown.endswith(
            '## References\n\n'
            '- <a id="cite-bib1"></a>Smith. Paper A. 2020.\n'
            '- <a id="cite-bib2"></a>Doe. Paper B. 2021.\n'
        )

    def test_serialize_skips_empty_footnotes_and_references(self, work_dir):

        document = doc_models.Document(
            title='Empty Document',
            abstract='Abstract.',
            description=None,
            sections=[],
            footnotes=[],
            referenced_papers=[],
        )
        environment = SerializationEnvironment.create(document, work_dir)

        MarkdownSerializer(environment).serialize()

        assert (environment.output_dir / 'paper.md').read_text(encoding='utf-8') == (
            '# Empty Document\n\n## Abstract\n\nAbstract.\n'
        )

    def test_serialize_raises_when_archive_cannot_be_written(self, sample_environment):

        sample_environment.archive_path.mkdir()

        with pytest.raises(
            MarkdownSerializer.SerializationError, match='Failed to serialize the document'
        ):
            MarkdownSerializer(sample_environment).serialize()

        assert sample_environment.archive_path.is_dir()

    def test_serialize_raises_when_output_dir_is_missing(self, sample_document, work_dir):

        environment = SerializationEnvironment(sample_document, work_dir)

        with pytest.raises(
            MarkdownSerializer.SerializationError, match='Failed to serialize the document'
        ):
            MarkdownSerializer(environment).serialize()

        assert not environment.archive_path.exists()
