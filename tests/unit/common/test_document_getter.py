import pytest

from paper_enrichment_agent.common.document_getter import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models


@pytest.fixture(scope='session')
def sample_document():
    return doc_models.Document(
        component_id='doc_1',
        description='Sample document for testing.',
        abstract='Document abstract.',
        sections=[
            doc_models.Section(
                component_id='section_1',
                title='Section 1',
                components=[
                    doc_models.Paragraph(
                        component_id='paragraph_1',
                        elements=[
                            'This is a paragraph.',
                            doc_models.Reference(
                                component_id='ref_1',
                                ref_type='element',
                                target='/section_2/paragraph_2',
                                content_text='See Section 2, Paragraph 2',
                            ),
                        ],
                    ),
                    doc_models.Figure(
                        component_id='figure_1',
                        description='This is a figure.',
                        subfigures=[
                            doc_models.ImgSubfigure(
                                component_id='img_subfig_1',
                                description='This is an image subfigure.',
                                image_src='/images/sample_image.png',
                                caption='This is an image subfigure.',
                            ),
                            doc_models.TableSubfigure(
                                component_id='table_subfig_1',
                                description='This is a table subfigure.',
                                table_contents='<table><tr><td>Sample Table Content</td></tr></table>',
                                caption='This is a table subfigure.',
                            ),
                        ],
                        caption='This is a figure.',
                    ),
                ],
            ),
            doc_models.Section(
                component_id='section_2',
                title='Section 2',
                components=[
                    doc_models.Paragraph(
                        component_id='paragraph_2',
                        elements=['This is another paragraph.'],
                    ),
                    doc_models.MathExpression(
                        component_id='math_expr_1', expression='E=mc^2', format='LaTeX'
                    ),
                    doc_models.MathExpression(
                        component_id='math_expr_2',
                        expression='\\frac{1}{2}',
                        format='LaTeX',
                    ),
                    doc_models.List(
                        component_id='list_1',
                        ordered=True,
                        items=[
                            doc_models.Paragraph(
                                component_id='list_item_1',
                                elements=['This is the first list item.'],
                            ),
                            doc_models.Paragraph(
                                component_id='list_item_2',
                                elements=['This is the second list item.'],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        footnotes=[
            doc_models.Paragraph(
                component_id='footnote_1',
                elements=['This is a footnote.'],
            ),
            doc_models.Section(
                component_id='footnote_2',
                title='Footnote Section',
                components=[
                    doc_models.MathExpression(
                        component_id='math_expr', expression='a^2 + b^2 = c^2', format='LaTeX'
                    ),
                ],
            ),
        ],
        referenced_papers=[],
    )


@pytest.fixture(scope='session')
def expected_paths(sample_document):

    return (
        ('/sections/section_1', sample_document.sections[0]),
        ('/sections/section_1/paragraph_1', sample_document.sections[0].components[0]),
        ('/sections/section_1/figure_1', sample_document.sections[0].components[1]),
        (
            '/sections/section_1/figure_1/img_subfig_1',
            sample_document.sections[0].components[1].subfigures[0],
        ),
        (
            '/sections/section_1/figure_1/table_subfig_1',
            sample_document.sections[0].components[1].subfigures[1],
        ),
        ('/sections/section_2', sample_document.sections[1]),
        ('/sections/section_2/paragraph_2', sample_document.sections[1].components[0]),
        ('/sections/section_2/math_expr_1', sample_document.sections[1].components[1]),
        ('/sections/section_2/math_expr_2', sample_document.sections[1].components[2]),
        (
            '/sections/section_1/paragraph_1/ref_1',
            sample_document.sections[0].components[0].elements[1],
        ),
        (
            '/footnotes/footnote_1',
            sample_document.footnotes[0],
        ),
        (
            '/footnotes/footnote_2',
            sample_document.footnotes[1],
        ),
        (
            '/footnotes/footnote_2/math_expr',
            sample_document.footnotes[1].components[0],
        ),
        (
            '/sections/section_2/list_1',
            sample_document.sections[1].components[3],
        ),
        (
            '/sections/section_2/list_1/list_item_1',
            sample_document.sections[1].components[3].items[0],
        ),
        (
            '/sections/section_2/list_1/list_item_2',
            sample_document.sections[1].components[3].items[1],
        ),
    )


class TestDocumentGetter:
    def test_get_path_of_component(self, sample_document, expected_paths):

        getter = DocumentGetter(sample_document)

        for path, component in expected_paths:
            assert getter.get_path_of_component(component) == path

    def test_get_path_of_component_not_found(self, sample_document):
        getter = DocumentGetter(sample_document)

        with pytest.raises(ValueError):
            getter.get_path_of_component(doc_models.DocumentComponent(component_id='non_existent'))

    def test_iter_components_of_type(self, sample_document):
        getter = DocumentGetter(sample_document)

        paragraphs = list(getter.iter_components_of_type(doc_models.Paragraph))
        assert len(paragraphs) == 5

        figures = list(getter.iter_components_of_type(doc_models.Figure))
        assert len(figures) == 1

        math_expressions = list(getter.iter_components_of_type(doc_models.MathExpression))
        assert len(math_expressions) == 3

        references = list(getter.iter_components_of_type(doc_models.Reference))
        assert len(references) == 1

        img_subfigures = list(getter.iter_components_of_type(doc_models.ImgSubfigure))
        assert len(img_subfigures) == 1

        table_subfigures = list(getter.iter_components_of_type(doc_models.TableSubfigure))
        assert len(table_subfigures) == 1

        sections = list(getter.iter_components_of_type(doc_models.Section))
        assert len(sections) == 3

        lists = list(getter.iter_components_of_type(doc_models.List))
        assert len(lists) == 1

    def test_get_component_by_path(self, sample_document, expected_paths):
        getter = DocumentGetter(sample_document)

        for path, component in expected_paths:
            assert getter.get_component_by_path(path) == component

    def test_get_component_by_path_not_found(self, sample_document):
        getter = DocumentGetter(sample_document)

        with pytest.raises(ValueError):
            getter.get_component_by_path('/non_existent/path')
