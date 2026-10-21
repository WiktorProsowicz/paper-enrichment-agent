import pytest

from paper_enrichment_agent.footnote_enrichment_agent.components import doc_utils
from paper_enrichment_agent.common.models import document as doc_models


@pytest.fixture
def sample_paragraph():
    return doc_models.Paragraph(
        component_id='paragraph_1',
        elements=[
            'This is a formula: ',
            doc_models.MathExpression(expression='E=mc^2', format='LaTeX'),
            ' and these are referenced papers: ',
            doc_models.Reference(ref_type='citation', content_text='1', target='bib1'),
            doc_models.Reference(ref_type='citation', content_text='2', target='bib2'),
            doc_models.Reference(ref_type='citation', content_text='3', target='bib3'),
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
    )


class TestDocUtils:
    def test_stringify_paragraph_elements(self, sample_paragraph):

        str_elements = doc_utils.stringify_paragraph_elements(sample_paragraph)

        assert str_elements[0].str_content == 'This is a formula: '

        assert str_elements[1].str_content == '$E=mc^2$'
        assert isinstance(str_elements[1].doc_component, doc_models.MathExpression)

        assert str_elements[2].str_content == ' and these are referenced papers: '

        assert str_elements[3].str_content == '[1, 2, 3]'

        assert str_elements[4].str_content == ' and this is a link: '

        assert str_elements[5].str_content == 'http://example.com'
        assert isinstance(str_elements[5].doc_component, doc_models.Reference)

        assert str_elements[6].str_content == ' and this is a reference to element: '

        assert str_elements[7].str_content == 'Figure 3.1'
        assert str_elements[7].doc_component is None

        assert str_elements[8].str_content == ' and this is all. Thank you.'

    def test_stringify_paragraph_elements_with_citations_keps(self, sample_paragraph):

        str_elements = doc_utils.stringify_paragraph_elements(
            sample_paragraph, keep_citation_references=True
        )

        assert str_elements[3].str_content == '[1]'
        assert isinstance(str_elements[3].doc_component, doc_models.Reference)

        assert str_elements[4].str_content == '[2]'
        assert isinstance(str_elements[4].doc_component, doc_models.Reference)

        assert str_elements[5].str_content == '[3]'
        assert isinstance(str_elements[5].doc_component, doc_models.Reference)

    def test_render_paragraph_concatenates_stringified_elements(self, sample_paragraph):

        assert doc_utils.render_paragraph(sample_paragraph) == (
            'This is a formula: $E=mc^2$ and these are referenced papers: [1, 2, 3] and this is a '
            'link: http://example.com and this is a reference to element: Figure 3.1 and this is '
            'all. Thank you.'
        )

    def test_render_figure_lists_subfigures_with_ids(self):

        figure = doc_models.Figure(
            component_id='figure_1',
            caption='Figure caption.',
            subfigures=[
                doc_models.ImgSubfigure(
                    component_id='subfigure_1',
                    caption='Image caption.',
                    image_src='http://example.com/image.png',
                ),
                doc_models.TableSubfigure(
                    component_id='subfigure_2',
                    caption='Table caption.',
                    table_contents='<table></table>',
                ),
            ],
        )

        assert doc_utils.render_figure(figure) == (
            'Caption: Figure caption.\n'
            'Subfigures:\n'
            '\tImgSubfigure (id: subfigure_1): Image caption.\n'
            '\tTableSubfigure (id: subfigure_2): Table caption.'
        )

    def test_render_figure_renders_paragraph_captions(self, sample_paragraph):

        figure = doc_models.Figure(
            component_id='figure_1',
            caption=sample_paragraph,
            subfigures=[
                doc_models.ImgSubfigure(
                    component_id='subfigure_1',
                    caption=doc_models.Paragraph(
                        elements=[
                            '(a) ',
                            doc_models.MathExpression(expression='x > 0', format='LaTeX'),
                        ]
                    ),
                    image_src='http://example.com/image.png',
                ),
                doc_models.ImgSubfigure(
                    component_id='subfigure_2',
                    caption=None,
                    image_src='http://example.com/image.png',
                ),
            ],
        )

        assert doc_utils.render_figure(figure) == (
            f'Caption: {doc_utils.render_paragraph(sample_paragraph)}\n'
            'Subfigures:\n'
            '\tImgSubfigure (id: subfigure_1): (a) $x > 0$\n'
            '\tImgSubfigure (id: subfigure_2): '
        )

    @pytest.mark.parametrize(
        ('caption', 'expected_render'),
        [
            ('Figure 1: Plain caption.', 'Figure 1: Plain caption.'),
            (None, ''),
            (
                doc_models.Paragraph(
                    elements=[
                        'Figure 1: Caption with ',
                        doc_models.MathExpression(expression='x', format='LaTeX'),
                        ' and ',
                        doc_models.Reference(ref_type='citation', content_text='1', target='b1'),
                        doc_models.Reference(ref_type='citation', content_text='2', target='b2'),
                    ]
                ),
                'Figure 1: Caption with $x$ and [1, 2]',
            ),
        ],
    )
    def test_render_caption_renders_plain_and_paragraph_captions(self, caption, expected_render):

        assert doc_utils.render_caption(caption) == expected_render

    def test_render_equation_includes_format_and_expression(self):

        equation = doc_models.MathExpression(expression='a^2 + b^2 = c^2', format='LaTeX')

        assert doc_utils.render_equation(equation) == 'Format: LaTeX\nExpression: a^2 + b^2 = c^2'
