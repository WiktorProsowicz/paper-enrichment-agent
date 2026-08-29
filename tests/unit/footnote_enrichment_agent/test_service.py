import pytest
from unittest.mock import MagicMock, AsyncMock

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import FootnoteEnrichmentRequest
from paper_enrichment_agent.footnote_enrichment_agent.service import FootnoteEnrichmentAgentService


@pytest.fixture(scope='session')
def sample_reference_document():
    return doc_models.Document(
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
def mock_enrichment_context_manager():
    enrichment_context_manager = MagicMock()
    enrichment_tools = MagicMock()
    enrichment_context_manager.setup_mcp_for_agent_session.return_value.__enter__.return_value = (
        enrichment_tools
    )
    return enrichment_context_manager, enrichment_tools


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    monkeypatch.setattr(
        'paper_enrichment_agent.footnote_enrichment_agent.service._logger', MagicMock()
    )


@pytest.fixture
def mock_metrics():
    return MagicMock()


class TestReferenceToFootnote:
    @pytest.fixture(scope='class')
    @classmethod
    def sample_footnote(cls):
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

    @pytest.fixture(scope='class')
    @classmethod
    def mock_agent(cls, sample_footnote):
        agent = AsyncMock()
        agent.invoke.return_value = sample_footnote
        return agent

    @pytest.mark.asyncio
    async def test_enriches_footnote_properly(
        self,
        sample_enrichment_request,
        mock_agent,
        sample_footnote,
        mock_enrichment_context_manager,
        mock_metrics,
    ):
        enrichment_context_manager, enrichment_tools = mock_enrichment_context_manager

        service = FootnoteEnrichmentAgentService(
            metrics=mock_metrics,
            enrichment_context_manager=enrichment_context_manager,
            enrichment_agent=mock_agent,
        )

        result = await service.reference_to_footnote(sample_enrichment_request)

        assert result == sample_footnote
        mock_agent.invoke.assert_called_once_with(sample_enrichment_request, enrichment_tools)

        assert mock_metrics.enrichment_time.observe.call_args.args[0] >= 0
        mock_metrics.enrichment_requests.labels.assert_called_once_with(status='success')

    @pytest.mark.asyncio
    async def test_handles_enrichment_context_manager_error(
        self, mock_metrics, sample_enrichment_request, mock_agent, mock_enrichment_context_manager
    ):
        enrichment_context_manager, _ = mock_enrichment_context_manager
        enrichment_context_manager.setup_mcp_for_agent_session.side_effect = (
            FootnoteEnrichmentAgentService.FootnoteEnrichmentAgentError(
                'Failed to set up enrichment context for agent session'
            )
        )

        service = FootnoteEnrichmentAgentService(
            metrics=mock_metrics,
            enrichment_context_manager=enrichment_context_manager,
            enrichment_agent=mock_agent,
        )

        with pytest.raises(FootnoteEnrichmentAgentService.FootnoteEnrichmentAgentError):
            await service.reference_to_footnote(sample_enrichment_request)
