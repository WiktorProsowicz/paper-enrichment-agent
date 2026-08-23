import pytest

from paper_enrichment_agent.footnote_enrichment_agent.components import enrichment_context
from paper_enrichment_agent.common.models import document as doc_models


@pytest.fixture
def sample_enrichment_tools():
    return enrichment_context.EnrichmentTools(
        doc_models.Document(
            abstract='Sample abstract',
            description='Sample description',
            sections=[
                doc_models.Section(
                    component_id='section_1',
                    title='Sample Section',
                    components=[
                        doc_models.Paragraph(
                            component_id='paragraph_1',
                            elements=[
                                'This is a formula: ',
                                doc_models.MathExpression(expression='E=mc^2', format='LaTeX'),
                                ' and these are referenced papers: ',
                                doc_models.Reference(
                                    ref_type='citation', content_text='1', target='bib1'
                                ),
                                doc_models.Reference(
                                    ref_type='citation', content_text='2', target='bib2'
                                ),
                                doc_models.Reference(
                                    ref_type='citation', content_text='3', target='bib3'
                                ),
                                ' and this is a link: ',
                                doc_models.Reference(
                                    ref_type='link',
                                    content_text='http://example.com',
                                    target='http://example.com',
                                ),
                                ' and this is a reference to element: ',
                                doc_models.Reference(
                                    ref_type='element',
                                    content_text='Figure 3.1',
                                    target='/sections/nested_section/figure_1',
                                ),
                                ' and this is all. Thank you.',
                            ],
                        ),
                        doc_models.List(
                            component_id='list_1',
                            ordered=False,
                            items=[
                                doc_models.Paragraph(
                                    component_id='list_item_1', elements=['List item 1']
                                ),
                                doc_models.Paragraph(
                                    component_id='list_item_2', elements=['List item 2']
                                ),
                            ],
                        ),
                        doc_models.Section(
                            component_id='nested_section',
                            title='Nested Section',
                            components=[
                                doc_models.Figure(
                                    component_id='figure_1',
                                    caption='Sample figure caption.',
                                    subfigures=[
                                        doc_models.ImgSubfigure(
                                            component_id='subfigure_1',
                                            caption='Sample subfigure caption.',
                                            image_src='http://example.com/image1.png',
                                        ),
                                        doc_models.TableSubfigure(
                                            component_id='subfigure_2',
                                            caption='Sample subfigure caption.',
                                            table_contents='<table><tr><td>Sample table content</td></tr></table>',
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                )
            ],
            footnotes=[
                doc_models.Paragraph(component_id='footnote_1', elements=['Sample footnote.'])
            ],
            referenced_papers=[
                ('bib1', 'Sample Paper 1'),
                ('bib2', 'Sample Paper 2'),
                ('bib3', 'Sample Paper 3'),
            ],
        )
    )


class TestEnrichmentContext:
    def test_get_document_tree_returns_tree(self, sample_enrichment_tools):

        expected_tree = {
            'sections': [
                {
                    'type': 'Section',
                    'id': 'section_1',
                    'title': 'Sample Section',
                    'components': [
                        {'type': 'Paragraph', 'id': 'paragraph_1'},
                        {
                            'type': 'List',
                            'id': 'list_1',
                            'items': [
                                {
                                    'type': 'Paragraph',
                                    'id': 'list_item_1',
                                },
                                {
                                    'type': 'Paragraph',
                                    'id': 'list_item_2',
                                },
                            ],
                        },
                        {
                            'type': 'Section',
                            'id': 'nested_section',
                            'title': 'Nested Section',
                            'components': [
                                {
                                    'type': 'Figure',
                                    'id': 'figure_1',
                                    'caption': 'Sample figure caption.',
                                }
                            ],
                        },
                    ],
                }
            ],
            'footnotes': [
                {
                    'type': 'Paragraph',
                    'id': 'footnote_1',
                }
            ],
        }

        assert sample_enrichment_tools.get_document_tree() == expected_tree

    def test_set_footnote_title_updates_footnote(self, sample_enrichment_tools):

        assert sample_enrichment_tools.footnote.title == ''

        sample_enrichment_tools.set_footnote_title('Updated Footnote Title')

        assert sample_enrichment_tools.footnote.title == 'Updated Footnote Title'

    def test_get_paragraph_content_raises_error_for_invalid_path(self, sample_enrichment_tools):
        with pytest.raises(ValueError, match='Component not found in the document'):
            sample_enrichment_tools.get_paragraph_content('/invalid/path')

    def test_get_paragraph_content_raises_error_for_non_paragraph_path(
        self, sample_enrichment_tools
    ):
        with pytest.raises(ValueError, match='is not a paragraph'):
            sample_enrichment_tools.get_paragraph_content('/sections/section_1/list_1')

    def test_get_paragraph_content_returns_stringified_paragraph(self, sample_enrichment_tools):

        path_to_paragraph = '/sections/section_1/paragraph_1'
        expected_content = 'This is a formula: $E=mc^2$ and these are referenced papers: [1, 2, 3] and this is a link: http://example.com and this is a reference to element: Figure 3.1 and this is all. Thank you.'

        assert sample_enrichment_tools.get_paragraph_content(path_to_paragraph) == expected_content

    def test_get_figure_details_returns_stringified_figure(self, sample_enrichment_tools):

        path_to_figure = '/sections/section_1/nested_section/figure_1'

        expected_details = (
            'Caption: Sample figure caption.\n'
            'Subfigures:\n'
            '\tImgSubfigure: Sample subfigure caption.\n'
            '\tTableSubfigure: Sample subfigure caption.'
        )

        assert sample_enrichment_tools.get_figure_details(path_to_figure) == expected_details

    def test_extract_paragraph_citation_keeps_necessary_components(self, sample_enrichment_tools):

        sample_enrichment_tools.extract_paragraph_citation(
            path_to_paragraph='/sections/section_1/paragraph_1',
            begin_anchor='a formula:',
            end_anchor='this is all.',
        )

        assert len(sample_enrichment_tools.footnote.components) == 1

        extracted_paragraph = sample_enrichment_tools.footnote.components[0]

        assert isinstance(extracted_paragraph, doc_models.Paragraph)

        assert extracted_paragraph.elements[0] == 'a formula: '
        assert isinstance(extracted_paragraph.elements[1], doc_models.MathExpression)
        assert (
            extracted_paragraph.elements[2]
            == ' and these are referenced papers: [1, 2, 3] and this is a link: '
        )
        assert isinstance(extracted_paragraph.elements[3], doc_models.Reference)
        assert (
            extracted_paragraph.elements[4]
            == ' and this is a reference to element: Figure 3.1 and this is all.'
        )

    def test_extract_paragraph_citation_does_not_drop_boundary_components(
        self, sample_enrichment_tools
    ):

        sample_enrichment_tools.extract_paragraph_citation(
            path_to_paragraph='/sections/section_1/paragraph_1',
            begin_anchor='example.com',
            end_anchor='this is a reference',
        )

        assert len(sample_enrichment_tools.footnote.components) == 1

        extracted_paragraph = sample_enrichment_tools.footnote.components[0]

        assert isinstance(extracted_paragraph, doc_models.Paragraph)

        assert isinstance(extracted_paragraph.elements[0], doc_models.Reference)
        assert extracted_paragraph.elements[0].content_text == 'http://example.com'
        assert extracted_paragraph.elements[1] == ' and this is a reference'

    def test_extract_figure_raises_on_invalid_path(self, sample_enrichment_tools):
        with pytest.raises(ValueError, match='Component not found in the document'):
            sample_enrichment_tools.extract_figure('/invalid/path', subfigures_ids=None)

    def test_extract_figure_raises_on_non_figure_path(self, sample_enrichment_tools):
        with pytest.raises(ValueError, match='is not a figure'):
            sample_enrichment_tools.extract_figure(
                '/sections/section_1/paragraph_1', subfigures_ids=None
            )

    def test_extract_figure_raises_on_wrong_subfigures_ids(self, sample_enrichment_tools):

        with pytest.raises(ValueError, match='Some subfigure ids do not exist in the figure'):
            sample_enrichment_tools.extract_figure(
                '/sections/section_1/nested_section/figure_1',
                subfigures_ids=['subfigure_1', 'subfigure_3'],
            )

    def test_extract_figure_correctly_extracts_chosen_subfigures(self, sample_enrichment_tools):

        sample_enrichment_tools.extract_figure(
            '/sections/section_1/nested_section/figure_1',
            subfigures_ids=['subfigure_1'],
        )

        assert len(sample_enrichment_tools.footnote.components) == 1

        extracted_figure = sample_enrichment_tools.footnote.components[0]

        assert isinstance(extracted_figure, doc_models.Figure)

        assert extracted_figure.caption == 'Sample figure caption.'
        assert len(extracted_figure.subfigures) == 1
        assert isinstance(extracted_figure.subfigures[0], doc_models.ImgSubfigure)

    def test_extract_figure_correctly_extracts_all_subfigures(self, sample_enrichment_tools):

        sample_enrichment_tools.extract_figure(
            '/sections/section_1/nested_section/figure_1',
            subfigures_ids=None,
        )

        assert len(sample_enrichment_tools.footnote.components) == 1

        extracted_figure = sample_enrichment_tools.footnote.components[0]

        assert isinstance(extracted_figure, doc_models.Figure)

        assert extracted_figure.caption == 'Sample figure caption.'
        assert len(extracted_figure.subfigures) == 2
        assert isinstance(extracted_figure.subfigures[0], doc_models.ImgSubfigure)
        assert isinstance(extracted_figure.subfigures[1], doc_models.TableSubfigure)
