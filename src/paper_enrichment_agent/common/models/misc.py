"""Contains miscellaneous data models that are shared by the app's services."""

import uuid
from typing import Annotated

import pydantic
from pydantic import Field

from paper_enrichment_agent.common.models import document as doc_models


class DocumentMetadata(pydantic.BaseModel):
    """Represents the metadata of a document being in the system.

    The `document` may be either a survey or a referenced paper.
    """

    paper_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description='The unique identifier of the document in the database.',
    )

    name: Annotated[str, Field(description='The name of the document.')]
    description: Annotated[str, Field(description='The description of the document.')]
    images: Annotated[
        dict[str, str],
        Field(
            description='A mapping of image paths to their paths in the database. The key is the'
            'image path in the document, and the value is the path in the database.'
        ),
    ]


class SurveyMetadata(pydantic.BaseModel):
    """Represents the metadata of a survey being in the system."""

    paper_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description='The identifier of the survey paper in the database.',
    )
    referenced_docs: Annotated[
        dict[str, str],
        Field(
            description=(
                'An id-to-id mapping of referenced documents in the survey.'
                ' The key is the ID of the reference in the Document class,'
                ' and the value is the `paper_id` of the referenced document.'
            )
        ),
    ]


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

    reference_document_title: Annotated[
        str, Field(description='The title of the document referenced by the survey.')
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
