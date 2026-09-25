import pytest
from unittest.mock import MagicMock, AsyncMock

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import FootnoteEnrichmentRequest
from paper_enrichment_agent.footnote_enrichment_agent.components.enrichment_context_manager import (
    EnrichmentContextManager,
)
from paper_enrichment_agent.footnote_enrichment_agent.service import FootnoteEnrichmentAgentService


@pytest.fixture(scope='session')
def sample_reference_document():
    return doc_models.Document(
        title='Sample Reference Document',
        abstract='This is a sample reference document abstract.',
        description='This is a sample reference document description.',
        sections=[
            doc_models.Section(
                title='Sample Section',
                components=[
                    doc_models.Paragraph(
                        elements=['This is a sample paragraph in the reference document.']
                    ),
                    doc_models.Figure(
                        caption='Sample Figure Caption',
                        subfigures=[
                            doc_models.ImgSubfigure(
                                caption='Sample Subfigure Caption',
                                image_src='https://example.com/sample_image.png',
                            )
                        ],
                    ),
                ],
            )
        ],
        footnotes=[],
        referenced_papers=[],
    )


@pytest.fixture(scope='session')
def sample_enrichment_request(sample_reference_document):

    return FootnoteEnrichmentRequest(
        session_id='enrichment_session_x',
        survey_title='Sample Survey',
        survey_abstract='This is a sample survey abstract.',
        reference_document=sample_reference_document,
        reference_document_title='Sample Reference Document',
        reference_id='ref_123',
        referencing_paragraph=doc_models.Paragraph(
            elements=[
                'This paragraph refers to a paper ',
                doc_models.Reference(ref_type='citation', content_text='1', target='bib_1'),
                ' and it is important for the discussion.',
            ]
        ),
        citation_context_info='Section 1 -> Section 1.3 -> Section -> 1.3.4',
    )


@pytest.fixture
def mock_enrichment_context_manager(sample_footnote):
    """Mocks the manager running the enrichment context of a session."""

    manager = MagicMock(spec=EnrichmentContextManager)
    manager.setup_mcp_for_agent_session.return_value.__aenter__.return_value = None
    manager.get_footnote_state.return_value = sample_footnote

    return manager


@pytest.fixture(scope='session')
def sample_footnote():
    return doc_models.Section(
        title='Enriched Footnote',
        components=[
            doc_models.Figure(
                caption='Sample Figure Caption',
                subfigures=[
                    doc_models.ImgSubfigure(
                        caption='Sample Subfigure Caption',
                        image_src='https://example.com/sample_image.png',
                    )
                ],
            )
        ],
    )


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    monkeypatch.setattr(
        'paper_enrichment_agent.footnote_enrichment_agent.service._logger', MagicMock()
    )


@pytest.fixture
def mock_metrics():
    return MagicMock()


class TestReferenceToFootnote:
    @pytest.mark.asyncio
    async def test_enriches_footnote_properly(
        self,
        sample_enrichment_request,
        sample_footnote,
        mock_enrichment_context_manager,
        mock_metrics,
    ):
        agent = AsyncMock()
        agent.invoke.return_value = 'The footnote has been composed.'

        service = FootnoteEnrichmentAgentService(
            metrics=mock_metrics,
            enrichment_context_manager=mock_enrichment_context_manager,
            enrichment_agent=agent,
        )

        result = await service.reference_to_footnote(sample_enrichment_request)

        assert result == sample_footnote

        mock_enrichment_context_manager.setup_mcp_for_agent_session.assert_called_once_with(
            sample_enrichment_request.session_id, sample_enrichment_request.reference_document
        )
        agent.invoke.assert_called_once_with(sample_enrichment_request)
        mock_enrichment_context_manager.get_footnote_state.assert_called_once_with(
            sample_enrichment_request.session_id
        )

        assert mock_metrics.enrichment_time.observe.call_args.args[0] >= 0
        mock_metrics.enrichment_requests.labels.assert_called_once_with(status='success')

    @pytest.mark.asyncio
    async def test_handles_enrichment_context_error(
        self, mock_metrics, sample_enrichment_request, mock_enrichment_context_manager
    ):
        mock_enrichment_context_manager.setup_mcp_for_agent_session.side_effect = (
            EnrichmentContextManager.EnrichmentContextManagerError('Failed to set up the context.')
        )

        service = FootnoteEnrichmentAgentService(
            metrics=mock_metrics,
            enrichment_context_manager=mock_enrichment_context_manager,
            enrichment_agent=AsyncMock(),
        )

        with pytest.raises(FootnoteEnrichmentAgentService.FootnoteEnrichmentAgentError):
            await service.reference_to_footnote(sample_enrichment_request)

        mock_metrics.enrichment_requests.labels.assert_called_once_with(status='failure')

    @pytest.mark.asyncio
    async def test_handles_missing_footnote_state(
        self, mock_metrics, sample_enrichment_request, mock_enrichment_context_manager
    ):
        mock_enrichment_context_manager.get_footnote_state.side_effect = (
            EnrichmentContextManager.EnrichmentContextManagerError('No tools state is set up.')
        )

        service = FootnoteEnrichmentAgentService(
            metrics=mock_metrics,
            enrichment_context_manager=mock_enrichment_context_manager,
            enrichment_agent=AsyncMock(),
        )

        with pytest.raises(FootnoteEnrichmentAgentService.FootnoteEnrichmentAgentError):
            await service.reference_to_footnote(sample_enrichment_request)

        mock_metrics.enrichment_requests.labels.assert_called_once_with(status='failure')
