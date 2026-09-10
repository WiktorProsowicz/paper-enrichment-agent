from typing import Any

import pydantic
from llm_eval import core as harness_core

from paper_enrichment_agent.common.models import document as doc_models


class FootnoteEnrichmentDSInputs(pydantic.BaseModel):
    """Used by the footnote enrichment agent to generate the footnote.

    See :class:`FootnoteEnrichmentRequest` for more details.
    """

    survey_title: str
    survey_arxiv_id: str
    cited_document_title: str
    cited_document_arxiv_id: str
    citing_paragraph_path: str
    reference_id: str
    citation_context: str


class FootnoteEnrichmentDSExpectations(pydantic.BaseModel):
    """Expected ratings for LLM Judges calibration."""

    footnote: doc_models.Section
    figure_usefulness_rating: int
    text_relevance_rating: int
    title_fitness_rating: int


type ScorerCalibrationEvalSample = harness_core.EvalSample[
    FootnoteEnrichmentDSInputs, FootnoteEnrichmentDSExpectations
]

type FootnoteEnrichmentEvalSample = harness_core.EvalSample[
    FootnoteEnrichmentDSInputs, dict[str, Any]
]
