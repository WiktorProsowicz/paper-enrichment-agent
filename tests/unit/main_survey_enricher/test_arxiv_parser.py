import pytest
from unittest.mock import Mock
import pathlib

import requests

from paper_enrichment_agent.main_survey_enricher.arxiv_parser import ArxivParser
from paper_enrichment_agent.common.document_getter import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models

EXAMPLE_PAPER_1 = '1905.09263'
EXAMPLE_PAPER_2 = '2006.04558'
EXAMPLE_PAPER_3 = '1706.03762'

EXAMPLE_PAPERS = [
    # FastSpeech: Fast, Robust and Controllable Text to Speech
    {
        'id': EXAMPLE_PAPER_1,
        'path': pathlib.Path(__file__).parent / f'example_arxiv_paper_{EXAMPLE_PAPER_1}.html',
        'url': f'https://ar5iv.labs.arxiv.org/html/{EXAMPLE_PAPER_1}',
    },
    # FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
    {
        'id': EXAMPLE_PAPER_2,
        'path': pathlib.Path(__file__).parent / f'example_arxiv_paper_{EXAMPLE_PAPER_2}.html',
        'url': f'https://ar5iv.labs.arxiv.org/html/{EXAMPLE_PAPER_2}',
    },
    # Attention Is All You Need
    {
        'id': EXAMPLE_PAPER_3,
        'path': pathlib.Path(__file__).parent / f'example_arxiv_paper_{EXAMPLE_PAPER_3}.html',
        'url': f'https://ar5iv.labs.arxiv.org/html/{EXAMPLE_PAPER_3}',
    },
]


@pytest.fixture
def mock_example_paper(monkeypatch):
    def fake_get(url, *args, **kwargs):

        for paper in EXAMPLE_PAPERS:
            if url == paper['url']:
                return Mock(content=paper['path'].read_bytes(), raise_for_status=Mock())

        raise_for_status = Mock()
        raise_for_status.side_effect = requests.HTTPError(
            f'404 Client Error: Not Found for url: {url}'
        )

        return Mock(content=b'', raise_for_status=raise_for_status)

    monkeypatch.setattr(
        'paper_enrichment_agent.main_survey_enricher.arxiv_parser.requests.get', fake_get
    )


class TestArxivParser:
    def test_raises_on_nonexistent_paper(self, mock_example_paper):
        parser = ArxivParser()

        with pytest.raises(ArxivParser.ParsingError):
            parser.parse('nonexistent_paper_id')

    def test_parses_correct_abstract(self, mock_example_paper):

        parser = ArxivParser()
        document = parser.parse('1905.09263')

        assert document.abstract.startswith('Neural network based end-to-end text to speech (TTS)')
        assert document.abstract.endswith('Therefore, we call our model FastSpeech.')

    def test_parses_correct_sections(self, mock_example_paper):

        parser = ArxivParser()
        document = parser.parse(EXAMPLE_PAPER_1)

        doc_getter = DocumentGetter(document)

        all_sections = doc_getter.iter_components_of_type(doc_models.Section)
        sec_titles = [section.title for section in all_sections]

        expected_titles = [
            '1 Introduction',
            '2 Background',
            'Text to Speech',
            'Sequence to Sequence Learning',
            'Non-Autoregressive Sequence Generation',
            '3 FastSpeech',
            '3.1 Feed-Forward Transformer',
            '3.2 Length Regulator',
            '3.3 Duration Predictor',
            '4 Experimental Setup',
            '4.1 Datasets',
            '4.2 Model Configuration',
            'FastSpeech model',
            'Autoregressive Transformer TTS model',
            '4.3 Training and Inference',
            '5 Results',
            'Audio Quality',
            'Inference Speedup',
            'Robustness',
            'Length Control',
            'Ablation Study',
            '6 Conclusions',
            'Acknowledgments',
            'Appendix A Model Hyperparameters',
            'Appendix B 50 Particularly Hard Sentences',
        ]

        assert sec_titles == expected_titles

    def test_parses_correct_tables(self, mock_example_paper):

        parser = ArxivParser()
        document = parser.parse(EXAMPLE_PAPER_2)

        doc_getter = DocumentGetter(document)

        all_figures = list(doc_getter.iter_components_of_type(doc_models.Figure))
        table_captions = [table.caption for table in all_figures]

        assert 'Table 1: Audio quality comparison.' in table_captions[1]
        assert 'Table 6: CMOS comparison in the ablation studies.' in table_captions[6]

        second_figure_subfigures = all_figures[1].subfigures

        assert 'GT (Mel + PWG)' in second_figure_subfigures[0].table_contents
        second_figure_subfigures[0].caption == '(a) The MOS with 95% confidence intervals.'
        assert 'FastSpeech 2' in second_figure_subfigures[1].table_contents
        second_figure_subfigures[1].caption == '(b) CMOS comparison.'

    def test_parses_correct_figures(self, mock_example_paper):

        parser = ArxivParser()
        document = parser.parse(EXAMPLE_PAPER_1)

        doc_getter = DocumentGetter(document)

        all_figures = list(doc_getter.iter_components_of_type(doc_models.Figure))
        figure_captions = [figure.caption for figure in all_figures]

        assert 'Figure 1 : The overall architecture for FastSpeech.' in figure_captions[0]
        assert 'Figure 4 : The mel-spectrograms before and after' in figure_captions[6]

        first_figure_subfigures = all_figures[0].subfigures

        assert len(first_figure_subfigures) == 4
        assert first_figure_subfigures[0].image_src == '/html/1905.09263/assets/x1.png'
        assert first_figure_subfigures[0].caption == '(a) Feed-Forward Transformer'
        assert first_figure_subfigures[3].image_src == '/html/1905.09263/assets/x4.png'
        assert first_figure_subfigures[3].caption == '(d) Duration Predictor'

    def test_parses_correct_equations(self, mock_example_paper):

        parser = ArxivParser()
        document = parser.parse(EXAMPLE_PAPER_3)

        doc_getter = DocumentGetter(document)

        sections = list(doc_getter.iter_components_of_type(doc_models.Section))
        equations = [
            equation
            for section in sections
            for equation in filter(
                lambda comp: isinstance(comp, doc_models.MathExpression), section.components
            )
        ]

        equation_contents = [equation.expression for equation in equations]

        assert (
            r'\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^{T}}{\sqrt{d_{k}}})V'
            in equation_contents[0]
        )

        assert (
            r'\displaystyle PE_{(pos,2i)}=sin(pos/10000^{2i/d_{\text{model}}})'
            in equation_contents[4]
        )
        assert (
            r'\displaystyle PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{\text{model}}})'
            in equation_contents[5]
        )

    def test_parses_correct_references(self, mock_example_paper):

        parser = ArxivParser()
        document = parser.parse(EXAMPLE_PAPER_3)

        assert len(document.referenced_papers) == 40

        assert document.referenced_papers[0] == (
            'bib.bib1',
            'Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450 , 2016.',
        )

        assert document.referenced_papers[-1] == (
            'bib.bib40',
            'Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers) , pages 434–443. ACL, August 2013.',
        )

    def test_parses_correct_paragraphs(self, mock_example_paper):

        parser = ArxivParser()
        document = parser.parse(EXAMPLE_PAPER_1)

        doc_getter = DocumentGetter(document)

        chosen_paragraph = doc_getter.get_component_by_path('/sections/S3/S3.SS2/S3.SS2.p1.8')

        assert chosen_paragraph.elements[0] == 'The length regulator (Figure '
        assert chosen_paragraph.elements[1].content_text == '1(c)'
        assert (
            chosen_paragraph.elements[7].expression == r'\mathcal{H}_{pho}=[h_{1},h_{2},...,h_{n}]'
        )

    def test_parses_correct_references(self, mock_example_paper):

        document = ArxivParser().parse(EXAMPLE_PAPER_1)
        doc_getter = DocumentGetter(document)

        ref_1 = doc_getter.get_component_by_path(
            '/sections/S2/S2.SS0.SSS0.Px3/S2.SS0.SSS0.Px3.p1.1/3.0'
        )

        assert isinstance(ref_1, doc_models.Reference)
        assert ref_1.ref_type == 'citation'
        assert ref_1.target == 'bib.bib16'

        ref_2 = doc_getter.get_component_by_path('/sections/S3/S3.SS2/S3.SS2.p1.21/11')

        assert isinstance(ref_2, doc_models.Reference)
        assert ref_2.ref_type == 'element'
        assert ref_2.target == '/sections/S3/S3.SS2/S3.E1'

        document = ArxivParser().parse(EXAMPLE_PAPER_2)
        doc_getter = DocumentGetter(document)

        ref_1 = doc_getter.get_component_by_path('/sections/S1/S1.p3.1/5')

        assert isinstance(ref_1, doc_models.Reference)
        assert ref_1.ref_type == 'link'
        assert ref_1.target == 'https://speechresearch.github.io/fastspeech2/'

    def test_parses_correct_footnotes(self, mock_example_paper):

        document = ArxivParser().parse(EXAMPLE_PAPER_1)

        assert len(document.footnotes) == 4

        assert isinstance(document.footnotes[0], doc_models.Paragraph)
        assert document.footnotes[0].elements[0] == 'Although ClariNet '
        assert document.footnotes[0].elements[1].content_text == '18'
        assert (
            document.footnotes[0].elements[2]
            == ' is fully end-to-end, it still first generates mel-spectrogram autoregressively and then synthesizes speech in one model.'
        )

        assert isinstance(document.footnotes[-1], doc_models.Paragraph)
        assert (
            document.footnotes[-1].elements[0]
            == 'These cases include single letters, spellings, repeated numbers, and long sentences. We list the cases in the supplementary materials.'
        )
