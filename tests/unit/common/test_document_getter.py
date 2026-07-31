import pytest

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.document_getter import DocumentGetter


@pytest.fixture(scope='session')
def sample_document():
    return doc_models.Document(
        component_id='doc_1',
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
                                target='/section_2/paragraph_2',
                                content_text='See Section 2, Paragraph 2',
                            ),
                        ],
                    ),
                    doc_models.Figure(
                        component_id='figure_1',
                        image_paths=['/path/to/image1.png', '/path/to/image2.png'],
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
                ],
            ),
        ],
        footnotes=[],
        referenced_papers=[],
    )


@pytest.fixture(scope='session')
def expected_paths(sample_document):

    return (
        ('/section_1', sample_document.sections[0]),
        ('/section_1/paragraph_1', sample_document.sections[0].components[0]),
        ('/section_1/figure_1', sample_document.sections[0].components[1]),
        ('/section_2', sample_document.sections[1]),
        ('/section_2/paragraph_2', sample_document.sections[1].components[0]),
        ('/section_2/math_expr_1', sample_document.sections[1].components[1]),
        ('/section_2/math_expr_2', sample_document.sections[1].components[2]),
        ('/section_1/paragraph_1/ref_1', sample_document.sections[0].components[0].elements[1]),
    )


@pytest.mark.dependency(name='test_document_getter', scope='session')
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
        assert len(paragraphs) == 2

        figures = list(getter.iter_components_of_type(doc_models.Figure))
        assert len(figures) == 1

        math_expressions = list(getter.iter_components_of_type(doc_models.MathExpression))
        assert len(math_expressions) == 2

        references = list(getter.iter_components_of_type(doc_models.Reference))
        assert len(references) == 1

    def test_get_component_by_path(self, sample_document, expected_paths):
        getter = DocumentGetter(sample_document)

        for path, component in expected_paths:
            assert getter.get_component_by_path(path) == component

    def test_get_component_by_path_not_found(self, sample_document):
        getter = DocumentGetter(sample_document)

        with pytest.raises(ValueError):
            getter.get_component_by_path('/non_existent/path')
