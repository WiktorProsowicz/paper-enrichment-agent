"""Contains the request / response models of the app's services' HTTP APIs.

The models define the bodies of the modifying (POST) endpoints and the structured responses of the
services. They are shared by the services exposing the endpoints and by their clients (other
services, the LLM evaluation harness).
"""

import uuid
from typing import Annotated

import pydantic
from pydantic import Field

from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import (
    DocumentMetadata,
    PaperAccessInfo,
    PaperMetadata,
    SerializedDocFormat,
)


class FootnoteEnrichmentRequest(pydantic.BaseModel):
    """Contains request data for the footnote enrichment agent.

    The request data includes the context needed to convert a reference that explains the
    contribution of a document into the survey that references it. The enriched footnote is
    expected to be composed from excerpts taken from the referenced document, crucial for the exact
    context of the citation.
    """

    session_id: str = Field(
        description='The session ID of the enrichment request.',
        default_factory=lambda: uuid.uuid4().hex,
    )

    survey_title: Annotated[
        str, Field(description='The title of the survey that references the document.')
    ]

    survey_abstract: Annotated[
        str, Field(description='The abstract of the survey that references the document.')
    ]

    reference_document: Annotated[
        doc_models.Document, Field(description='The document referenced by the survey.')
    ]

    reference_id: Annotated[
        str, Field(description='The component_id of the reference to the document in the survey.')
    ]

    referencing_paragraph: Annotated[
        doc_models.Paragraph,
        Field(description='The paragraph in the survey that references the document.'),
    ]

    citation_context_info: Annotated[
        str,
        Field(
            description=(
                'Additional context information explaining the citation, e.g. info about'
                'the section it comes from.'
            )
        ),
    ]


class ArxivSurveyRegistrationRequest(pydantic.BaseModel):
    """Contains request data for registering an arXiv paper as a survey."""

    arxiv_id: Annotated[
        str, Field(description='The arXiv ID of the paper to register as a survey.', min_length=1)
    ]
    name: Annotated[str, Field(description='The name of the survey document.')]
    description: Annotated[str, Field(description='The description of the survey document.')]


class SurveyRemovalRequest(pydantic.BaseModel):
    """Contains request data for removing a survey together with its referenced documents."""

    survey_id: Annotated[str, Field(description='The ID of the survey to remove.', min_length=1)]


class DocumentComponentRemovalRequest(pydantic.BaseModel):
    """Contains request data for removing a component from a document."""

    paper_id: Annotated[
        str, Field(description='The ID of the document to remove the component from.', min_length=1)
    ]
    component_path: Annotated[
        str,
        Field(
            description=(
                'The POSIX-like path of the component to remove, e.g. `/sections/<id>/<id>`.'
            ),
            min_length=1,
        ),
    ]


class DocumentDetailsResponse(pydantic.BaseModel):
    """Contains detailed information about a document stored in the system."""

    metadata: Annotated[DocumentMetadata, Field(description='The metadata of the document.')]
    document: Annotated[doc_models.Document, Field(description='The structure of the document.')]


class DocumentRenderingRequest(pydantic.BaseModel):
    """Contains request data for serializing a document and storing the result in the database."""

    paper_id: Annotated[str, Field(description='The ID of the document to render.', min_length=1)]
    doc_format: Annotated[
        SerializedDocFormat, Field(description='The format to render the document into.')
    ]


class SurveyFootnoteEnrichmentRequest(pydantic.BaseModel):
    """Contains request data for enriching a citation of a survey into a footnote."""

    survey_id: Annotated[
        str, Field(description='The ID of the survey containing the citation.', min_length=1)
    ]
    citation_path: Annotated[
        str,
        Field(
            description=(
                'The POSIX-like path of the citation reference to enrich, e.g. '
                '`/sections/<id>/<paragraph_id>/<reference_id>`.'
            ),
            min_length=1,
        ),
    ]


class ArxivPaperParsingRequest(pydantic.BaseModel):
    """Contains request data for scraping and parsing an arXiv paper."""

    arxiv_id: Annotated[
        str,
        Field(description='The arXiv ID of the paper to parse, e.g. `1706.03762`.', min_length=1),
    ]


class ParsedPaperResponse(pydantic.BaseModel):
    """Contains a scraped paper parsed into the document structure, along with its metadata."""

    document: Annotated[doc_models.Document, Field(description='The structure of the paper.')]
    metadata: Annotated[
        PaperMetadata, Field(description='The bibliographic metadata of the paper.')
    ]


class CitationSignatureParsingRequest(pydantic.BaseModel):
    """Contains request data for finding, scraping and parsing a paper by its citation."""

    citation_signature: Annotated[
        str,
        Field(
            description=(
                'The citation signature of the paper, as found in the bibliography of the citing '
                'paper, e.g. `A. Vaswani et al. Attention is all you need. NeurIPS, 2017.`.'
            ),
            min_length=1,
        ),
    ]


class PaperAccessResolutionRequest(pydantic.BaseModel):
    """Contains request data for finding the best public source of a paper by its citation."""

    citation_signature: Annotated[
        str,
        Field(
            description=(
                'The citation signature of the paper, as found in the bibliography of the citing '
                'paper, e.g. `A. Vaswani et al. Attention is all you need. NeurIPS, 2017.`.'
            ),
            min_length=1,
        ),
    ]


class PaperAccessResolutionResponse(pydantic.BaseModel):
    """Contains the resolved public source of a cited paper, along with its metadata."""

    access_info: Annotated[
        PaperAccessInfo | None,
        Field(
            description=(
                'The access information of the paper, or None if no public version of it was found.'
            )
        ),
    ]
