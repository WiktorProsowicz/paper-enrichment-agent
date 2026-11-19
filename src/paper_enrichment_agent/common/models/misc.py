"""Contains miscellaneous data models that are shared by the app's services.

The request / response models of the services' HTTP APIs live in
:mod:`~paper_enrichment_agent.common.models.service_conn`.
"""

import datetime
import enum
import uuid
from typing import Annotated

import pydantic
from pydantic import Field


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


class PaperMetadata(pydantic.BaseModel):
    """Represents the bibliographic metadata of a scraped paper."""

    doi: Annotated[str, Field(description='The DOI of the paper.', min_length=1)]
    authors: Annotated[
        list[str], Field(description='The full names of the authors, in the publication order.')
    ]
    publication_date: Annotated[
        datetime.date, Field(description='The date of the first publication of the paper.')
    ]


class PaperAccessType(enum.StrEnum):
    """Enumerates the kinds of public sources of papers."""

    # The paper has an HTML version on arXiv, the source is its arXiv ID
    ARXIV = 'arxiv'

    # The paper is available as an open access PDF, the source is the URL of the PDF
    PDF = 'pdf'


class PaperAccessInfo(pydantic.BaseModel):
    """Describes how to access a publicly available version of a paper."""

    type: Annotated[PaperAccessType, Field(description='The kind of the public source.')]
    source: Annotated[
        str,
        Field(
            description=(
                'The arXiv ID of the paper (e.g. `1706.03762`) for the `arxiv` type, or the URL of '
                'the PDF file for the `pdf` type.'
            ),
            min_length=1,
        ),
    ]
    metadata: Annotated[
        PaperMetadata,
        Field(description='The bibliographic metadata of the paper.'),
    ]


class SerializedDocFormat(enum.StrEnum):
    """Enumerates the formats documents can be serialized (rendered) into."""

    MARKDOWN = 'markdown'


class RenderedDocumentMetadata(pydantic.BaseModel):
    """Represents the metadata of a document serialized into one of the supported formats."""

    paper_id: Annotated[
        str, Field(description='The identifier of the rendered document in the database.')
    ]
    doc_format: Annotated[
        SerializedDocFormat, Field(description='The format the document is rendered into.')
    ]
    size_bytes: Annotated[int, Field(description='The size of the rendered archive.', ge=0)]
    rendered_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description='The UTC time of the rendering.',
    )
