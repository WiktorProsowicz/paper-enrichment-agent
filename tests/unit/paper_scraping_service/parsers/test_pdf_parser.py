import json
import pathlib
from unittest.mock import Mock

import httpx
import pytest

from paper_enrichment_agent.common.document_manipulators import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.paper_scraping_service.components.parsers.pdf_parser import (
    MarkerBlock,
    PdfParser,
)

MARKER_URL = 'http://marker.test'
MARKER_TOKEN = 'secret-token'
PDF_URL = 'https://papers.test/pdf/1905.09263v5'
PDF_CONTENT = b'%PDF-1.5 fake content'

FIXTURES_DIR = pathlib.Path(__file__).parent

# Neural Machine Translation by Jointly Learning to Align and Translate (author-year citations)
EXAMPLE_PAPER_1 = '1409.0473'
# FastSpeech: Fast, Robust and Controllable Text to Speech (numeric citations)
EXAMPLE_PAPER_2 = '1905.09263'


def load_marker_output(paper_id: str) -> dict:
    return json.loads((FIXTURES_DIR / f'example_marker_output_{paper_id}.json').read_text())


def make_config(**overrides) -> PdfParser.Configuration:
    return PdfParser.Configuration.model_validate(
        {
            'marker_api_url': MARKER_URL,
            'marker_api_token': MARKER_TOKEN,
            'marker_api_timeout': 10.0,
            'download_timeout': 5.0,
            'max_pdf_size_bytes': 1024,
            **overrides,
        }
    )


def make_marker_output(*page_blocks: dict) -> dict:
    return {
        'block_type': 'Document',
        'children': [{'id': '/page/0/Page/0', 'block_type': 'Page', 'children': list(page_blocks)}],
    }


def header(block_id: str, title: str, level: int = 2) -> dict:
    return {'id': block_id, 'block_type': 'SectionHeader', 'html': f'<h{level}>{title}</h{level}>'}


def text(block_id: str, html: str) -> dict:
    return {'id': block_id, 'block_type': 'Text', 'html': html}


class FakeTransport:
    """Serves the PDF download and the marker conversion, recording the requests."""

    def __init__(self, marker_response: httpx.Response | Exception, pdf_response=None) -> None:
        self.marker_response = marker_response
        self.pdf_response = pdf_response or httpx.Response(200, content=PDF_CONTENT)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        response = (
            self.marker_response if str(request.url).startswith(MARKER_URL) else self.pdf_response
        )

        if isinstance(response, Exception):
            raise response

        return response


def make_parser(transport: FakeTransport, **config_overrides) -> PdfParser:
    return PdfParser(
        config=make_config(**config_overrides),
        http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )


def parse_fixture(paper_id: str) -> doc_models.Document:
    output = MarkerBlock.model_validate(load_marker_output(paper_id))
    return PdfParser(config=make_config(), http_client=Mock()).parse_marker_output(output)


def section_outline(sections: list[doc_models.Section]) -> list:
    return [
        (
            section.title,
            section_outline([c for c in section.components if isinstance(c, doc_models.Section)]),
        )
        for section in sections
    ]


class TestParse:
    def test_downloads_pdf_and_sends_it_to_marker(self):
        transport = FakeTransport(httpx.Response(200, json=load_marker_output(EXAMPLE_PAPER_2)))

        document = make_parser(transport).parse(PDF_URL)

        assert document.title == 'FastSpeech: Fast, Robust and Controllable Text to Speech'

        download_request, marker_request = transport.requests
        assert str(download_request.url) == PDF_URL
        assert str(marker_request.url) == f'{MARKER_URL}/convert_pdf'
        assert marker_request.method == 'POST'
        assert b'filename="1905.09263v5.pdf"' in marker_request.content
        assert PDF_CONTENT in marker_request.content

    def test_sends_token_only_to_marker(self):
        transport = FakeTransport(httpx.Response(200, json=load_marker_output(EXAMPLE_PAPER_2)))

        make_parser(transport).parse(PDF_URL)

        download_request, marker_request = transport.requests
        assert 'Authorization' not in download_request.headers
        assert marker_request.headers['Authorization'] == f'Bearer {MARKER_TOKEN}'

    def test_raises_when_download_fails(self):
        transport = FakeTransport(
            httpx.Response(200, json={}), pdf_response=httpx.Response(404, content=b'')
        )

        with pytest.raises(PdfParser.ParsingError, match='Failed to download'):
            make_parser(transport).parse(PDF_URL)

    def test_raises_when_connection_fails(self):
        transport = FakeTransport(httpx.Response(200, json={}))
        transport.pdf_response = httpx.ConnectError('Connection refused')

        with pytest.raises(PdfParser.ParsingError, match='Failed to download'):
            make_parser(transport).parse(PDF_URL)

    def test_raises_when_file_is_not_pdf(self):
        transport = FakeTransport(
            httpx.Response(200, json={}), pdf_response=httpx.Response(200, content=b'<html>')
        )

        with pytest.raises(PdfParser.ParsingError, match='not a PDF'):
            make_parser(transport).parse(PDF_URL)

        assert len(transport.requests) == 1

    def test_raises_when_pdf_is_too_large(self):
        transport = FakeTransport(httpx.Response(200, json={}))

        with pytest.raises(PdfParser.ParsingError, match='exceeding the limit'):
            make_parser(transport, max_pdf_size_bytes=4).parse(PDF_URL)

    @pytest.mark.parametrize(
        'marker_response',
        [
            httpx.Response(401, text='Unauthorized'),
            httpx.Response(500, json={'detail': 'PDF conversion failed.'}),
            httpx.ReadTimeout('Timed out'),
        ],
    )
    def test_raises_when_conversion_fails(self, marker_response):
        transport = FakeTransport(marker_response)

        with pytest.raises(PdfParser.ParsingError, match='Failed to convert'):
            make_parser(transport).parse(PDF_URL)

    @pytest.mark.parametrize(
        'marker_response',
        [
            httpx.Response(200, content=b'not json'),
            httpx.Response(200, json={'children': 'invalid'}),
        ],
    )
    def test_raises_when_marker_output_is_invalid(self, marker_response):
        transport = FakeTransport(marker_response)

        with pytest.raises(PdfParser.ParsingError, match='invalid output'):
            make_parser(transport).parse(PDF_URL)

    def test_raises_when_output_has_no_headers(self):
        transport = FakeTransport(
            httpx.Response(200, json=make_marker_output(text('/page/0/Text/0', '<p>Text</p>')))
        )

        with pytest.raises(PdfParser.ParsingError, match='no section headers'):
            make_parser(transport).parse(PDF_URL)


class TestParsePdfContent:
    def test_sends_given_file_to_marker_without_downloading(self):
        transport = FakeTransport(httpx.Response(200, json=load_marker_output(EXAMPLE_PAPER_2)))

        document = make_parser(transport).parse_pdf_content(PDF_CONTENT, 'fastspeech.pdf')

        assert document.title == 'FastSpeech: Fast, Robust and Controllable Text to Speech'

        (marker_request,) = transport.requests
        assert str(marker_request.url) == f'{MARKER_URL}/convert_pdf'
        assert marker_request.headers['Authorization'] == f'Bearer {MARKER_TOKEN}'
        assert b'filename="fastspeech.pdf"' in marker_request.content
        assert PDF_CONTENT in marker_request.content

    @pytest.mark.parametrize(
        ('file_name', 'expected_name'),
        [('paper', 'paper.pdf'), ('', 'paper.pdf'), ('../dir/paper.pdf', 'paper.pdf')],
    )
    def test_normalizes_file_name(self, file_name, expected_name):
        transport = FakeTransport(httpx.Response(200, json=load_marker_output(EXAMPLE_PAPER_2)))

        make_parser(transport).parse_pdf_content(PDF_CONTENT, file_name)

        (marker_request,) = transport.requests
        assert f'filename="{expected_name}"'.encode() in marker_request.content

    @pytest.mark.parametrize(
        ('content', 'config_overrides', 'error'),
        [
            (b'', {}, 'are empty'),
            (b'<html>', {}, 'not a PDF'),
            (PDF_CONTENT, {'max_pdf_size_bytes': 4}, 'exceeding the limit'),
        ],
    )
    def test_rejects_invalid_file_without_calling_marker(self, content, config_overrides, error):
        transport = FakeTransport(httpx.Response(200, json=load_marker_output(EXAMPLE_PAPER_2)))

        with pytest.raises(PdfParser.ParsingError, match=error):
            make_parser(transport, **config_overrides).parse_pdf_content(content, 'paper.pdf')

        assert transport.requests == []

    def test_raises_when_conversion_fails(self):
        transport = FakeTransport(httpx.Response(500, json={'detail': 'PDF conversion failed.'}))

        with pytest.raises(PdfParser.ParsingError, match='Failed to convert'):
            make_parser(transport).parse_pdf_content(PDF_CONTENT, 'paper.pdf')


class TestParseMarkerOutputOnExamplePapers:
    def test_extracts_title_and_abstract(self):
        document = parse_fixture(EXAMPLE_PAPER_1)

        assert document.title == (
            'NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE'
        )
        assert document.abstract.startswith('Neural machine translation is a recently proposed')
        assert document.footnotes == []

    def test_builds_section_tree_from_numbering(self):
        document = parse_fixture(EXAMPLE_PAPER_1)

        outline = section_outline(document.sections)

        assert [title for title, _ in outline] == [
            '1 INTRODUCTION',
            '2 BACKGROUND: NEURAL MACHINE TRANSLATION',
            '3 LEARNING TO ALIGN AND TRANSLATE',
            '4 EXPERIMENT SETTINGS',
            '5 RESULTS',
            '6 RELATED WORK',
            '7 CONCLUSION',
            'ACKNOWLEDGMENTS',
            'A MODEL ARCHITECTURE',
            'B TRAINING PROCEDURE',
            'C TRANSLATIONS OF LONG SENTENCES',
        ]
        assert outline[4] == (
            '5 RESULTS',
            [
                ('5.1 QUANTITATIVE RESULTS', []),
                (
                    '5.2 QUALITATIVE ANALYSIS',
                    [('5.2.1 ALIGNMENT', []), ('5.2.2 LONG SENTENCES', [])],
                ),
            ],
        )

    def test_keeps_unnumbered_sections_at_heading_depth(self):
        document = parse_fixture(EXAMPLE_PAPER_2)

        assert [section.title for section in document.sections][-4:] == [
            'Acknowledgments',
            'Appendices',
            'A Model Hyperparameters',
            'B 50 Particularly Hard Sentences',
        ]

    @pytest.mark.parametrize(
        ('paper_id', 'n_entries', 'first_entry'),
        [
            (EXAMPLE_PAPER_1, 29, ('bib-1', 'Axelrod, A., He, X., and Gao, J. (2011).')),
            (EXAMPLE_PAPER_2, 28, ('bib-1', 'Sercan O Arik, Mike Chrzanowski')),
        ],
    )
    def test_extracts_bibliography(self, paper_id, n_entries, first_entry):
        document = parse_fixture(paper_id)

        assert len(document.referenced_papers) == n_entries
        assert len({entry_id for entry_id, _ in document.referenced_papers}) == n_entries
        assert document.referenced_papers[0][0] == first_entry[0]
        assert document.referenced_papers[0][1].startswith(first_entry[1])
        assert all(
            section.title.lower() != 'references'
            for section in DocumentGetter(document).iter_components_of_type(doc_models.Section)
        )

    def test_uses_numeric_bibliography_labels_as_ids(self):
        document = parse_fixture(EXAMPLE_PAPER_2)

        assert dict(document.referenced_papers)['bib-25'].startswith('Ashish Vaswani')

    def test_does_not_resolve_citations(self):
        document = parse_fixture(EXAMPLE_PAPER_2)

        references = list(
            DocumentGetter(document).iter_components_of_type(doc_models.Reference)
        ) + [
            element
            for paragraph in DocumentGetter(document).iter_components_of_type(doc_models.Paragraph)
            for element in paragraph.elements
            if isinstance(element, doc_models.Reference)
        ]

        assert all(reference.ref_type == 'link' for reference in references)

    @pytest.mark.parametrize(
        ('paper_id', 'n_images', 'n_tables'), [(EXAMPLE_PAPER_1, 3, 3), (EXAMPLE_PAPER_2, 4, 5)]
    )
    def test_decodes_figures_with_captions(self, paper_id, n_images, n_tables):
        document = parse_fixture(paper_id)
        getter = DocumentGetter(document)

        figures = list(getter.iter_components_of_type(doc_models.Figure))
        images = list(getter.iter_components_of_type(doc_models.ImgSubfigure))
        tables = list(getter.iter_components_of_type(doc_models.TableSubfigure))

        assert len(images) == n_images
        assert len(tables) == n_tables
        assert all(image.image_src.startswith('data:image/jpeg;base64,/9j/') for image in images)
        assert all(table.table_contents.startswith('<table>') for table in tables)
        assert all(figure.caption for figure in figures)

    def test_keeps_inline_math_in_captions(self):
        document = parse_fixture(EXAMPLE_PAPER_1)

        caption = next(DocumentGetter(document).iter_components_of_type(doc_models.Figure)).caption

        assert isinstance(caption, doc_models.Paragraph)
        assert (
            caption.elements[0]
            == 'Figure 1: The graphical illustration of the proposed model trying to generate the '
        )
        assert caption.elements[1] == doc_models.MathExpression(
            component_id=caption.elements[1].component_id, expression='t', format='LaTeX'
        )

    def test_decodes_equations_and_inline_math(self):
        document = parse_fixture(EXAMPLE_PAPER_1)
        getter = DocumentGetter(document)

        equations = [
            component
            for section in getter.iter_components_of_type(doc_models.Section)
            for component in section.components
            if isinstance(component, doc_models.MathExpression)
        ]
        inline_math = [
            element
            for paragraph in getter.iter_components_of_type(doc_models.Paragraph)
            for element in paragraph.elements
            if isinstance(element, doc_models.MathExpression)
        ]

        assert equations[0].expression == r'h_t = f(x_t, h_{t-1}) \quad (1)'
        assert all(equation.format == 'LaTeX' for equation in equations + inline_math)
        assert len(inline_math) > 50

    def test_decodes_lists_without_bullets(self):
        document = parse_fixture(EXAMPLE_PAPER_2)

        first_list = next(DocumentGetter(document).iter_components_of_type(doc_models.List))

        assert not first_list.ordered
        assert len(first_list.items) == 3
        assert first_list.items[0].elements[0].startswith('Slow inference speed')

    def test_merges_paragraphs_split_by_page_breaks(self):
        document = parse_fixture(EXAMPLE_PAPER_2)

        paragraph = document.sections[0].components[0]

        assert isinstance(paragraph, doc_models.Paragraph)
        assert 'Parallel WaveNet [16], or WaveGlow [20]. Neural network based TTS' in str(
            paragraph.elements[0]
        )

    def test_drops_page_footnotes_and_footnote_markers(self):
        document = parse_fixture(EXAMPLE_PAPER_2)

        texts = [
            element
            for paragraph in DocumentGetter(document).iter_components_of_type(doc_models.Paragraph)
            for element in paragraph.elements
            if isinstance(element, str)
        ]

        assert not any('speechresearch.github.io' in t for t in texts)
        assert not any('WaveGlow [20]4' in t for t in texts)

    @pytest.mark.parametrize('paper_id', [EXAMPLE_PAPER_1, EXAMPLE_PAPER_2])
    def test_produces_addressable_components(self, paper_id):
        document = parse_fixture(paper_id)
        getter = DocumentGetter(document)

        paths = [path for path, _ in getter.iter_components_with_paths()]

        assert len(paths) == len(set(paths))
        assert all(
            '/' not in component.component_id
            for _, component in getter.iter_components_with_paths()
        )
        for path in paths:
            getter.get_component_by_path(path)


class TestParseMarkerOutputEdgeCases:
    def parse(self, *blocks: dict) -> doc_models.Document:
        output = MarkerBlock.model_validate(make_marker_output(*blocks))
        return PdfParser(config=make_config(), http_client=Mock()).parse_marker_output(output)

    def test_drops_blocks_between_title_and_first_section(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            text('/page/0/Text/1', '<p>John Doe</p>'),
            header('/page/0/SectionHeader/2', '1 Intro'),
            text('/page/0/Text/3', '<p>Body.</p>'),
        )

        assert document.title == 'Title'
        assert document.abstract == ''
        assert [s.title for s in document.sections] == ['1 Intro']
        assert document.sections[0].components == [
            doc_models.Paragraph(component_id='page-0-Text-3', elements=['Body.'])
        ]

    def test_nests_unnumbered_sections_by_heading_rank(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'Method', level=2),
            header('/page/0/SectionHeader/2', 'Details', level=3),
            header('/page/0/SectionHeader/3', 'Results', level=2),
        )

        assert section_outline(document.sections) == [
            ('Method', [('Details', [])]),
            ('Results', []),
        ]

    def test_decodes_links(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'Intro'),
            text('/page/0/Text/2', '<p>See <a href="https://x.test">the site</a>.</p>'),
        )

        elements = document.sections[0].components[0].elements
        assert elements[0] == 'See '
        assert elements[1] == doc_models.Reference(
            component_id=elements[1].component_id,
            ref_type='link',
            target='https://x.test',
            content_text='the site',
        )
        assert elements[2] == '.'

    def test_attaches_following_caption_to_standalone_visual(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'Intro'),
            {
                'id': '/page/0/Picture/2',
                'block_type': 'Picture',
                'images': {'/page/0/Picture/2': 'iVBORw0KGgo='},
            },
            {'id': '/page/0/Caption/3', 'block_type': 'Caption', 'html': '<p>Figure 1: A.</p>'},
            {'id': '/page/0/Caption/4', 'block_type': 'Caption', 'html': '<p>Stray.</p>'},
            {'id': '/page/0/Table/5', 'block_type': 'Table', 'html': '<table></table>'},
        )

        figure_1, figure_2 = document.sections[0].components
        assert isinstance(figure_1, doc_models.Figure)
        assert figure_1.caption == 'Figure 1: A.'
        assert figure_1.subfigures[0].image_src == 'data:image/png;base64,iVBORw0KGgo='
        assert isinstance(figure_2, doc_models.Figure)
        assert figure_2.caption == ''
        assert isinstance(figure_2.subfigures[0], doc_models.TableSubfigure)

    def test_skips_figures_without_images(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'Intro'),
            {'id': '/page/0/Figure/2', 'block_type': 'Figure', 'html': ''},
        )

        assert document.sections[0].components == []

    def test_resolves_group_members_from_content_refs(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'Intro'),
            {
                'id': '/page/0/TableGroup/2',
                'block_type': 'TableGroup',
                'html': "<content-ref src='/page/0/Table/3'></content-ref>",
            },
            {'id': '/page/0/Table/3', 'block_type': 'Table', 'html': '<table>t</table>'},
        )

        group_figure = document.sections[0].components[0]
        assert isinstance(group_figure, doc_models.Figure)
        assert group_figure.component_id == 'page-0-TableGroup-2'
        assert group_figure.subfigures[0].component_id == 'page-0-Table-3'

    def test_does_not_merge_paragraphs_on_same_page(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'Intro'),
            text('/page/0/Text/2', '<p>First without a dot</p>'),
            text('/page/0/Text/3', '<p>second</p>'),
        )

        assert len(document.sections[0].components) == 2

    def test_splits_multiple_equations_in_block(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'Intro'),
            {
                'id': '/page/0/Equation/2',
                'block_type': 'Equation',
                'html': '<p><math display="block">a</math><math display="block">b</math></p>',
            },
            {'id': '/page/0/Equation/3', 'block_type': 'Equation', 'html': '<p>c = d</p>'},
        )

        assert [(c.component_id, c.expression) for c in document.sections[0].components] == [
            ('page-0-Equation-2-0', 'a'),
            ('page-0-Equation-2-1', 'b'),
            ('page-0-Equation-3', 'c = d'),
        ]

    def test_numbers_unlabeled_bibliography_entries(self):
        document = self.parse(
            header('/page/0/SectionHeader/0', 'Title', level=1),
            header('/page/0/SectionHeader/1', 'References'),
            {'id': '/page/0/ListGroup/2', 'block_type': 'ListGroup', 'html': '<p>A.</p><p>B.</p>'},
            {'id': '/page/0/Bibliography/3', 'block_type': 'Bibliography', 'html': '<p>C.</p>'},
        )

        assert document.referenced_papers == [('bib-1', 'A.'), ('bib-2', 'B.'), ('bib-3', 'C.')]
        assert document.sections == []
