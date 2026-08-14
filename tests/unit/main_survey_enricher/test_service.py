from unittest.mock import MagicMock, Mock

import pytest

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import DocumentMetadata, SurveyMetadata
from paper_enrichment_agent.main_survey_enricher.components.arxiv_parser import ArxivParser
from paper_enrichment_agent.main_survey_enricher.components.doc_db_client import DocDBClient
from paper_enrichment_agent.main_survey_enricher.service import MainSurveyEnricherService

SAMPLE_ARXIV_ID = '1905.09263'
SAMPLE_NAME = 'Sample Survey'
SAMPLE_DESCRIPTION = 'Sample Description'
SAMPLE_SURVEY_ID = 'sample_survey_id'


@pytest.fixture
def sample_survey_document() -> tuple[doc_models.Document, DocumentMetadata]:

    document = doc_models.Document(
        abstract='This is a sample abstract.',
        description='This is a sample description.',
        sections=[
            doc_models.Section(
                title='Introduction',
                components=[
                    doc_models.Paragraph(
                        elements=['This is a sample paragraph in the introduction section.']
                    )
                ],
            )
        ],
        footnotes=[],
        referenced_papers=[('ref1', 'Reference Paper 1'), ('ref2', 'Reference Paper 2')],
    )
    metadata = DocumentMetadata(
        paper_id='sample_paper_id',
        name=SAMPLE_NAME,
        description=SAMPLE_DESCRIPTION,
        images={},
    )
    return document, metadata


@pytest.fixture
def ref1_document() -> tuple[doc_models.Document, DocumentMetadata]:

    document = doc_models.Document(
        abstract='This is a sample abstract for Reference Paper 1.',
        description='This is a sample description for Reference Paper 1.',
        sections=[],
        footnotes=[],
        referenced_papers=[],
    )
    metadata = DocumentMetadata(
        paper_id='ref1_paper_id',
        name='Reference Paper 1',
        description='This is a sample description for Reference Paper 1.',
        images={},
    )
    return document, metadata


@pytest.fixture
def ref2_document() -> tuple[doc_models.Document, DocumentMetadata]:

    document = doc_models.Document(
        abstract='This is a sample abstract for Reference Paper 2.',
        description='This is a sample description for Reference Paper 2.',
        sections=[],
        footnotes=[],
        referenced_papers=[],
    )
    metadata = DocumentMetadata(
        paper_id='ref2_paper_id',
        name='Reference Paper 2',
        description='This is a sample description for Reference Paper 2.',
        images={},
    )
    return document, metadata


@pytest.fixture
def patch_arxiv_parser_fails(monkeypatch):

    parser = MagicMock(parse=Mock(side_effect=ArxivParser.ParsingError('Parsing failed')))

    monkeypatch.setattr(
        'paper_enrichment_agent.main_survey_enricher.service.ArxivParser',
        Mock(return_value=parser, ParsingError=ArxivParser.ParsingError),
    )

    return parser


@pytest.fixture
def mock_metrics() -> MagicMock:

    return MagicMock()


@pytest.fixture(autouse=True)
def patch_logger(monkeypatch) -> MagicMock:

    mock_logger = MagicMock()
    monkeypatch.setattr(
        'paper_enrichment_agent.main_survey_enricher.service._logger',
        Mock(return_value=mock_logger),
    )

    return mock_logger


class TestRegisterArxivSurvey:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:

        db_client = MagicMock()
        db_client.add_document.return_value = DocumentMetadata(
            paper_id='sample_paper_id', name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION, images={}
        )
        db_client.register_as_survey.return_value = SurveyMetadata(
            paper_id='sample_paper_id', referenced_docs={}
        )
        db_client.get_survey_info.return_value = SurveyMetadata(
            paper_id=SAMPLE_SURVEY_ID,
            referenced_docs={'ref1': 'ref_paper_id_1', 'ref2': 'ref_paper_id_2'},
        )

        return db_client

    @pytest.fixture
    def patch_arxiv_parser(self, monkeypatch, sample_survey_document):

        parser = MagicMock(parse=Mock(return_value=sample_survey_document))

        monkeypatch.setattr(
            'paper_enrichment_agent.main_survey_enricher.service.ArxivParser',
            Mock(return_value=parser, ParsingError=ArxivParser.ParsingError),
        )

        return parser

    def test_adds_document_and_registers_survey(
        self, patch_arxiv_parser, mock_metrics, mock_db_client, sample_survey_document
    ):

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)
        service.register_arxiv_survey(
            arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
        )

        patch_arxiv_parser.parse.assert_called_once_with(SAMPLE_ARXIV_ID)
        mock_db_client.add_document.assert_called_once_with(
            document=sample_survey_document, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
        )
        mock_db_client.register_as_survey.assert_called_once_with(paper_id='sample_paper_id')

    def test_reports_metrics_on_success(self, patch_arxiv_parser, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)
        service.register_arxiv_survey(
            arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
        )

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='arxiv', status='success')
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()

        mock_metrics.papers_added.labels.assert_called_once_with(source='arxiv')
        mock_metrics.papers_added.labels.return_value.inc.assert_called_once()

        assert mock_metrics.paper_parsing_time.observe.call_count == 1
        assert mock_metrics.paper_parsing_time.observe.call_args.args[0] >= 0

        assert mock_metrics.doc_db_operations_time.observe.call_count == 1
        assert mock_metrics.doc_db_operations_time.observe.call_args.args[0] >= 0

    def test_raises_when_parsing_fails(
        self, patch_arxiv_parser_fails, mock_metrics, mock_db_client
    ):

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to parse the arXiv paper with ID {SAMPLE_ARXIV_ID}',
        ):
            service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        assert mock_db_client.add_document.call_count == 0
        assert mock_db_client.register_as_survey.call_count == 0

    def test_reports_metrics_when_parsing_fails(
        self, patch_arxiv_parser_fails, mock_metrics, mock_db_client
    ):

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(MainSurveyEnricherService.MainSurveyEnricherError):
            service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='arxiv', status='failed')
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()

        assert mock_metrics.paper_parsing_time.observe.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    def test_raises_when_document_adding_fails(
        self, patch_arxiv_parser, mock_metrics, mock_db_client
    ):

        mock_db_client.add_document.side_effect = DocDBClient.DocDBClientError('Adding failed')

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to register the survey for the arXiv paper with ID {SAMPLE_ARXIV_ID}',
        ):
            service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        assert mock_db_client.register_as_survey.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    def test_raises_when_survey_registration_fails(
        self, patch_arxiv_parser, mock_metrics, mock_db_client
    ):

        mock_db_client.register_as_survey.side_effect = DocDBClient.DocDBClientError(
            'Registration failed'
        )

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to register the survey for the arXiv paper with ID {SAMPLE_ARXIV_ID}',
        ):
            service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        mock_db_client.add_document.assert_called_once()
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    def test_does_not_swallow_unexpected_errors(
        self, patch_arxiv_parser, mock_metrics, mock_db_client
    ):

        mock_db_client.add_document.side_effect = RuntimeError('Unexpected failure')

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )


class TestRemoveSurvey:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:

        db_client = MagicMock()
        db_client.get_survey_info.return_value = SurveyMetadata(
            paper_id=SAMPLE_SURVEY_ID,
            referenced_docs={'ref1': 'ref_paper_id_1', 'ref2': 'ref_paper_id_2'},
        )
        return db_client

    def test_deletes_survey(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)
        service.remove_survey(SAMPLE_SURVEY_ID)

        mock_db_client.get_survey_info.assert_called_once_with(SAMPLE_SURVEY_ID)
        mock_db_client.delete_survey.assert_called_once_with(SAMPLE_SURVEY_ID)

    def test_reports_metrics_on_success(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)
        service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        mock_metrics.papers_removed.inc.assert_called_once_with(3)

        assert mock_metrics.doc_db_operations_time.observe.call_count == 1
        assert mock_metrics.doc_db_operations_time.observe.call_args.args[0] >= 0

    def test_counts_survey_without_references(self, mock_metrics, mock_db_client):

        mock_db_client.get_survey_info.return_value = SurveyMetadata(
            paper_id=SAMPLE_SURVEY_ID,
            referenced_docs={},
        )

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)
        service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        mock_metrics.papers_removed.inc.assert_called_once_with(1)

    def test_raises_when_survey_info_retrieval_fails(self, mock_metrics, mock_db_client):

        mock_db_client.get_survey_info.side_effect = DocDBClient.DocDBClientError(
            'Retrieval failed'
        )

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to remove the survey with ID {SAMPLE_SURVEY_ID}',
        ):
            service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        assert mock_db_client.delete_survey.call_count == 0
        assert mock_metrics.papers_removed.inc.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    def test_raises_when_deletion_fails(self, mock_metrics, mock_db_client):

        mock_db_client.delete_survey.side_effect = DocDBClient.DocDBClientError('Deletion failed')

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to remove the survey with ID {SAMPLE_SURVEY_ID}',
        ):
            service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        assert mock_metrics.papers_removed.inc.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    def test_does_not_swallow_unexpected_errors(self, mock_metrics, mock_db_client):

        mock_db_client.delete_survey.side_effect = RuntimeError('Unexpected failure')

        service = MainSurveyEnricherService(metrics=mock_metrics, db_client=mock_db_client)

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            service.remove_survey(survey_id=SAMPLE_SURVEY_ID)
