import zipfile
from unittest.mock import MagicMock, Mock

import pytest

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import (
    DocumentMetadata,
    RenderedDocumentMetadata,
    SerializedDocFormat,
    SurveyMetadata,
)
from paper_enrichment_agent.main_survey_enricher.components.arxiv_parser import ArxivParser
from paper_enrichment_agent.main_survey_enricher.components.db_clients.doc_db_client import (
    DocDBClient,
)
from paper_enrichment_agent.main_survey_enricher.components.db_clients import (
    rendered_doc_db_client,
)
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.core import (
    SerializationEnvironment,
)
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.markdown import (
    MarkdownSerializer,
)
from paper_enrichment_agent.main_survey_enricher.service import MainSurveyEnricherService

SAMPLE_ARXIV_ID = '1905.09263'
SAMPLE_NAME = 'Sample Survey'
SAMPLE_DESCRIPTION = 'Sample Description'
SAMPLE_SURVEY_ID = 'sample_survey_id'


@pytest.fixture
def sample_survey_document() -> tuple[doc_models.Document, DocumentMetadata]:

    document = doc_models.Document(
        title='Sample Document',
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
        title='Reference Paper 1',
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
        title='Reference Paper 2',
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

        db_client = MagicMock(spec=DocDBClient)
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

    @pytest.mark.asyncio
    async def test_adds_document_and_registers_survey(
        self, patch_arxiv_parser, mock_metrics, mock_db_client, sample_survey_document
    ):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        await service.register_arxiv_survey(
            arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
        )

        patch_arxiv_parser.parse.assert_called_once_with(SAMPLE_ARXIV_ID)
        mock_db_client.add_document.assert_called_once_with(
            document=sample_survey_document, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
        )
        mock_db_client.register_as_survey.assert_called_once_with(paper_id='sample_paper_id')

    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(
        self, patch_arxiv_parser, mock_metrics, mock_db_client
    ):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        await service.register_arxiv_survey(
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

    @pytest.mark.asyncio
    async def test_raises_when_parsing_fails(
        self, patch_arxiv_parser_fails, mock_metrics, mock_db_client
    ):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to parse the arXiv paper with ID {SAMPLE_ARXIV_ID}',
        ):
            await service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        assert mock_db_client.add_document.call_count == 0
        assert mock_db_client.register_as_survey.call_count == 0

    @pytest.mark.asyncio
    async def test_reports_metrics_when_parsing_fails(
        self, patch_arxiv_parser_fails, mock_metrics, mock_db_client
    ):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(MainSurveyEnricherService.MainSurveyEnricherError):
            await service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        mock_metrics.papers_parsed.labels.assert_called_once_with(source='arxiv', status='failure')
        mock_metrics.papers_parsed.labels.return_value.inc.assert_called_once()

        assert mock_metrics.paper_parsing_time.observe.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_raises_when_document_adding_fails(
        self, patch_arxiv_parser, mock_metrics, mock_db_client
    ):

        mock_db_client.add_document.side_effect = DocDBClient.DocDBClientError('Adding failed')

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to register the survey for the arXiv paper with ID {SAMPLE_ARXIV_ID}',
        ):
            await service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        assert mock_db_client.register_as_survey.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_raises_when_survey_registration_fails(
        self, patch_arxiv_parser, mock_metrics, mock_db_client
    ):

        mock_db_client.register_as_survey.side_effect = DocDBClient.DocDBClientError(
            'Registration failed'
        )

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to register the survey for the arXiv paper with ID {SAMPLE_ARXIV_ID}',
        ):
            await service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )

        mock_db_client.add_document.assert_called_once()
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(
        self, patch_arxiv_parser, mock_metrics, mock_db_client
    ):

        mock_db_client.add_document.side_effect = RuntimeError('Unexpected failure')

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.register_arxiv_survey(
                arxiv_id=SAMPLE_ARXIV_ID, name=SAMPLE_NAME, description=SAMPLE_DESCRIPTION
            )


class TestRemoveSurvey:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:

        db_client = MagicMock(spec=DocDBClient)
        db_client.get_survey_info.return_value = SurveyMetadata(
            paper_id=SAMPLE_SURVEY_ID,
            referenced_docs={'ref1': 'ref_paper_id_1', 'ref2': 'ref_paper_id_2'},
        )
        return db_client

    @pytest.mark.asyncio
    async def test_deletes_survey(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        await service.remove_survey(SAMPLE_SURVEY_ID)

        mock_db_client.get_survey_info.assert_called_once_with(SAMPLE_SURVEY_ID)
        mock_db_client.delete_survey.assert_called_once_with(SAMPLE_SURVEY_ID)

    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        await service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        mock_metrics.papers_removed.inc.assert_called_once_with(3)

        assert mock_metrics.doc_db_operations_time.observe.call_count == 1
        assert mock_metrics.doc_db_operations_time.observe.call_args.args[0] >= 0

    @pytest.mark.asyncio
    async def test_counts_survey_without_references(self, mock_metrics, mock_db_client):

        mock_db_client.get_survey_info.return_value = SurveyMetadata(
            paper_id=SAMPLE_SURVEY_ID,
            referenced_docs={},
        )

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        await service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        mock_metrics.papers_removed.inc.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_raises_when_survey_info_retrieval_fails(self, mock_metrics, mock_db_client):

        mock_db_client.get_survey_info.side_effect = DocDBClient.DocDBClientError(
            'Retrieval failed'
        )

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to remove the survey with ID {SAMPLE_SURVEY_ID}',
        ):
            await service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        assert mock_db_client.delete_survey.call_count == 0
        assert mock_metrics.papers_removed.inc.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_raises_when_deletion_fails(self, mock_metrics, mock_db_client):

        mock_db_client.delete_survey.side_effect = DocDBClient.DocDBClientError('Deletion failed')

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match=f'Failed to remove the survey with ID {SAMPLE_SURVEY_ID}',
        ):
            await service.remove_survey(survey_id=SAMPLE_SURVEY_ID)

        assert mock_metrics.papers_removed.inc.call_count == 0
        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(self, mock_metrics, mock_db_client):

        mock_db_client.delete_survey.side_effect = RuntimeError('Unexpected failure')

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.remove_survey(survey_id=SAMPLE_SURVEY_ID)


class TestListSurveys:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:

        db_client = MagicMock(spec=DocDBClient)
        db_client.get_available_surveys.return_value = [
            SurveyMetadata(
                paper_id='survey1',
                referenced_docs={'ref1': 'ref_paper_id_1', 'ref2': 'ref_paper_id_2'},
            ),
            SurveyMetadata(
                paper_id='survey2',
                referenced_docs={'ref3': 'ref_paper_id_3'},
            ),
        ]
        return db_client

    @pytest.mark.asyncio
    async def test_lists_surveys(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        surveys = await service.list_surveys()

        mock_db_client.get_available_surveys.assert_called_once()
        assert len(surveys) == 2
        assert surveys[0].paper_id == 'survey1'
        assert surveys[1].paper_id == 'survey2'

        mock_metrics.doc_db_operations_time.observe.assert_called_once()
        assert mock_metrics.doc_db_operations_time.observe.call_args.args[0] >= 0

    @pytest.mark.asyncio
    async def test_raises_when_listing_surveys_fails(self, mock_metrics, mock_db_client):

        mock_db_client.get_available_surveys.side_effect = DocDBClient.DocDBClientError(
            'Listing failed'
        )

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match='Failed to list the registered surveys',
        ):
            await service.list_surveys()

        assert mock_metrics.doc_db_operations_time.observe.call_count == 0

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(self, mock_metrics, mock_db_client):

        mock_db_client.get_available_surveys.side_effect = RuntimeError('Unexpected failure')

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(RuntimeError, match='Unexpected failure'):
            await service.list_surveys()


class TestGetSurveyInfo:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:

        db_client = MagicMock(spec=DocDBClient)
        db_client.get_survey_info.return_value = SurveyMetadata(
            paper_id=SAMPLE_SURVEY_ID,
            referenced_docs={'ref1': 'ref_paper_id_1', 'ref2': 'ref_paper_id_2'},
        )
        return db_client

    @pytest.mark.asyncio
    async def test_gets_survey_info(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        survey_info = await service.get_survey_info(survey_id=SAMPLE_SURVEY_ID)

        mock_db_client.get_survey_info.assert_called_once_with(SAMPLE_SURVEY_ID)
        assert survey_info.paper_id == SAMPLE_SURVEY_ID
        assert len(survey_info.referenced_docs) == 2

    @pytest.mark.asyncio
    async def test_raises_when_survey_info_retrieval_fails(self, mock_metrics, mock_db_client):

        mock_db_client.get_survey_info.side_effect = DocDBClient.DocDBClientError(
            'Retrieval failed'
        )

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match='Failed to retrieve information about the survey with ID',
        ):
            await service.get_survey_info(survey_id=SAMPLE_SURVEY_ID)


class TestGetDocumentDetails:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:

        db_client = MagicMock(spec=DocDBClient)
        db_client.get_document_metadata.return_value = DocumentMetadata(
            paper_id='sample_paper_id',
            name=SAMPLE_NAME,
            description=SAMPLE_DESCRIPTION,
            images={},
        )
        db_client.get_document_struct_ref.return_value.__aenter__.return_value = (
            doc_models.Document(
                title='Sample Document',
                abstract='This is a sample abstract.',
                description='This is a sample description.',
                sections=[],
                footnotes=[],
                referenced_papers=[],
            )
        )
        return db_client

    @pytest.mark.asyncio
    async def test_gets_document_details(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )
        metadata, document = await service.get_document_details(paper_id='sample_paper_id')

        mock_db_client.get_document_metadata.assert_called_once_with('sample_paper_id')
        mock_db_client.get_document_struct_ref.assert_called_once_with('sample_paper_id')

        assert metadata.paper_id == 'sample_paper_id'
        assert document.abstract == 'This is a sample abstract.'


class TestRemoveDocumentComponent:
    @pytest.fixture
    def mock_db_client(self) -> MagicMock:

        db_client = MagicMock(spec=DocDBClient)
        db_client.get_document_struct_ref.return_value.__aenter__.return_value = (
            doc_models.Document(
                title='Sample Document',
                abstract='This is a sample abstract.',
                description='This is a sample description.',
                sections=[
                    doc_models.Section(
                        component_id='section1',
                        title='Introduction',
                        components=[
                            doc_models.Paragraph(
                                component_id='paragraph1',
                                elements=[
                                    'This is a sample paragraph in the introduction section.'
                                ],
                            )
                        ],
                    )
                ],
                footnotes=[],
                referenced_papers=[],
            )
        )
        return db_client

    @pytest.mark.asyncio
    async def test_raises_when_invalid_path(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match='Failed to remove component',
        ):
            await service.remove_document_component(
                paper_id='sample_paper_id', component_path='/non/existent/path'
            )

    @pytest.mark.asyncio
    async def test_removes_component_successfully(self, mock_metrics, mock_db_client):

        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient),
        )

        await service.remove_document_component(
            paper_id='sample_paper_id', component_path='/sections/section1/paragraph1'
        )

        mock_db_client.get_document_struct_ref.assert_called_once_with('sample_paper_id')
        mock_metrics.doc_db_operations_time.observe.assert_called_once()


SAMPLE_IMAGE_DB_PATH = 'documents/sample_paper_id/images/img.png'


@pytest.fixture
def document_with_image() -> doc_models.Document:

    return doc_models.Document(
        title='Document With Image',
        abstract='Abstract.',
        description=None,
        sections=[
            doc_models.Section(
                component_id='S1',
                title='Introduction',
                components=[
                    doc_models.Paragraph(component_id='P1', elements=['Some text.']),
                    doc_models.Figure(
                        component_id='F1',
                        subfigures=[
                            doc_models.ImgSubfigure(
                                component_id='I1',
                                image_src='https://example.com/img.png',
                                caption=None,
                            )
                        ],
                        caption='Figure caption.',
                    ),
                ],
            )
        ],
        footnotes=[],
        referenced_papers=[],
    )


class TestRenderDocument:
    @pytest.fixture
    def mock_db_client(self, document_with_image) -> MagicMock:

        db_client = MagicMock(spec=DocDBClient)
        db_client.get_document_metadata.return_value = DocumentMetadata(
            paper_id='sample_paper_id',
            name=SAMPLE_NAME,
            description=SAMPLE_DESCRIPTION,
            images={'/sections/S1/F1/I1': SAMPLE_IMAGE_DB_PATH},
        )
        db_client.get_document_struct_ref.return_value.__aenter__.return_value = document_with_image

        async def download_image(image_db_path, local_path):
            local_path.write_bytes(b'image bytes')

        db_client.download_image.side_effect = download_image

        return db_client

    @pytest.fixture
    def mock_rendered_db_client(self) -> MagicMock:

        rendered_db_client = MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient)
        # Captures the archive's content, since the archive is removed after the rendering.
        rendered_db_client.archive_contents = {}

        async def add_rendered_document(paper_id, doc_format, archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                rendered_db_client.archive_contents.update(
                    {name: archive.read(name) for name in archive.namelist()}
                )
            return RenderedDocumentMetadata(
                paper_id=paper_id, doc_format=doc_format, size_bytes=archive_path.stat().st_size
            )

        rendered_db_client.add_rendered_document.side_effect = add_rendered_document

        return rendered_db_client

    @pytest.fixture
    def service(self, mock_metrics, mock_db_client, mock_rendered_db_client):

        return MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=mock_db_client,
            rendered_doc_db_client=mock_rendered_db_client,
        )

    @pytest.mark.asyncio
    async def test_renders_and_stores_archive(
        self, service, mock_db_client, mock_rendered_db_client
    ):

        metadata = await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

        assert metadata.paper_id == 'sample_paper_id'
        assert metadata.doc_format == SerializedDocFormat.MARKDOWN
        assert metadata.size_bytes > 0

        mock_db_client.download_image.assert_awaited_once()
        assert (
            mock_db_client.download_image.await_args.kwargs['image_db_path'] == SAMPLE_IMAGE_DB_PATH
        )

        contents = mock_rendered_db_client.archive_contents
        assert contents['images/sections-S1-F1-I1.png'] == b'image bytes'
        assert b'# Document With Image' in contents['paper.md']
        assert b'images/sections-S1-F1-I1.png' in contents['paper.md']

    @pytest.mark.asyncio
    async def test_removes_work_dir_after_rendering(self, service, mock_rendered_db_client):

        await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

        archive_path = mock_rendered_db_client.add_rendered_document.await_args.kwargs[
            'archive_path'
        ]
        assert not archive_path.parent.exists()

    @pytest.mark.asyncio
    async def test_reports_metrics_on_success(self, service, mock_metrics):

        await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

        mock_metrics.documents_rendered.labels.assert_called_once_with(
            format=SerializedDocFormat.MARKDOWN, status='success'
        )
        mock_metrics.documents_rendered.labels.return_value.inc.assert_called_once()
        assert mock_metrics.document_rendering_time.observe.call_args.args[0] >= 0

    @pytest.mark.asyncio
    async def test_raises_when_image_is_missing_in_metadata(
        self, service, mock_metrics, mock_db_client
    ):

        mock_db_client.get_document_metadata.return_value.images.clear()

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match='image of component /sections/S1/F1/I1 is missing',
        ):
            await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

        mock_metrics.documents_rendered.labels.assert_called_once_with(
            format=SerializedDocFormat.MARKDOWN, status='failure'
        )

    @pytest.mark.asyncio
    async def test_raises_when_image_download_fails(self, service, mock_db_client):

        mock_db_client.download_image.side_effect = DocDBClient.DocDBClientError('No image')

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError, match='Failed to render'
        ):
            await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

    @pytest.mark.asyncio
    async def test_raises_when_storing_fails(self, service, mock_rendered_db_client):

        mock_rendered_db_client.add_rendered_document.side_effect = (
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError('Upload failed')
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError, match='Upload failed'
        ):
            await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

    @pytest.mark.asyncio
    async def test_raises_when_serialization_fails(self, service, monkeypatch):

        serializer = MagicMock(
            serialize=Mock(side_effect=MarkdownSerializer.SerializationError('Disk full'))
        )
        monkeypatch.setattr(
            'paper_enrichment_agent.main_survey_enricher.service.MarkdownSerializer',
            Mock(return_value=serializer, SerializationError=MarkdownSerializer.SerializationError),
        )

        with pytest.raises(MainSurveyEnricherService.MainSurveyEnricherError, match='Disk full'):
            await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

    @pytest.mark.asyncio
    async def test_raises_for_unsupported_format(self, service, monkeypatch, mock_db_client):

        monkeypatch.setattr(MainSurveyEnricherService, '_SUPPORTED_RENDERING_FORMATS', ())

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError, match='unsupported format markdown'
        ):
            await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

        mock_db_client.get_document_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_when_environment_creation_fails(self, service, monkeypatch):

        monkeypatch.setattr(
            'paper_enrichment_agent.main_survey_enricher.service.SerializationEnvironment.create',
            Mock(side_effect=SerializationEnvironment.SerializationEnvironmentError('No space')),
        )

        with pytest.raises(MainSurveyEnricherService.MainSurveyEnricherError, match='No space'):
            await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(self, service, mock_db_client):

        mock_db_client.download_image.side_effect = RuntimeError('Unexpected')

        with pytest.raises(RuntimeError, match='Unexpected'):
            await service.render_document('sample_paper_id', SerializedDocFormat.MARKDOWN)


class TestListRenderedDocuments:
    @pytest.mark.asyncio
    async def test_returns_rendered_documents(self, mock_metrics):

        rendered_db_client = MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient)
        rendered_docs = [
            RenderedDocumentMetadata(
                paper_id='p1', doc_format=SerializedDocFormat.MARKDOWN, size_bytes=1
            )
        ]
        rendered_db_client.list_rendered_documents.return_value = rendered_docs
        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=MagicMock(spec=DocDBClient),
            rendered_doc_db_client=rendered_db_client,
        )

        assert await service.list_rendered_documents('p1') == rendered_docs

        rendered_db_client.list_rendered_documents.assert_awaited_once_with('p1')
        mock_metrics.doc_db_operations_time.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_listing_fails(self, mock_metrics):

        rendered_db_client = MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient)
        rendered_db_client.list_rendered_documents.side_effect = (
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError('Access denied')
        )
        service = MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=MagicMock(spec=DocDBClient),
            rendered_doc_db_client=rendered_db_client,
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError,
            match='Failed to list the rendered documents',
        ):
            await service.list_rendered_documents()


class TestDownloadRenderedDocument:
    @pytest.fixture
    def mock_rendered_db_client(self) -> MagicMock:

        rendered_db_client = MagicMock(spec=rendered_doc_db_client.RenderedDocDBClient)
        rendered_db_client.rendered_document_exists.return_value = True

        return rendered_db_client

    @pytest.fixture
    def service(self, mock_metrics, mock_rendered_db_client):

        return MainSurveyEnricherService(
            metrics=mock_metrics,
            doc_db_client=MagicMock(spec=DocDBClient),
            rendered_doc_db_client=mock_rendered_db_client,
        )

    @pytest.mark.asyncio
    async def test_downloads_archive(self, service, mock_rendered_db_client, tmp_path):

        local_path = tmp_path / 'doc.zip'

        await service.download_rendered_document('p1', SerializedDocFormat.MARKDOWN, local_path)

        mock_rendered_db_client.download_rendered_document.assert_awaited_once_with(
            paper_id='p1', doc_format=SerializedDocFormat.MARKDOWN, local_path=local_path
        )

    @pytest.mark.asyncio
    async def test_raises_when_not_rendered(self, service, mock_rendered_db_client, tmp_path):

        mock_rendered_db_client.rendered_document_exists.return_value = False

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError, match='has not been rendered'
        ):
            await service.download_rendered_document(
                'p1', SerializedDocFormat.MARKDOWN, tmp_path / 'doc.zip'
            )

        mock_rendered_db_client.download_rendered_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_download_fails(self, service, mock_rendered_db_client, tmp_path):

        mock_rendered_db_client.download_rendered_document.side_effect = (
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError('Missing object')
        )

        with pytest.raises(
            MainSurveyEnricherService.MainSurveyEnricherError, match='Missing object'
        ):
            await service.download_rendered_document(
                'p1', SerializedDocFormat.MARKDOWN, tmp_path / 'doc.zip'
            )

    @pytest.mark.asyncio
    async def test_does_not_swallow_unexpected_errors(
        self, service, mock_rendered_db_client, tmp_path
    ):

        mock_rendered_db_client.download_rendered_document.side_effect = RuntimeError('Unexpected')

        with pytest.raises(RuntimeError, match='Unexpected'):
            await service.download_rendered_document(
                'p1', SerializedDocFormat.MARKDOWN, tmp_path / 'doc.zip'
            )
