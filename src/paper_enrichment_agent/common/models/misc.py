"""Contains miscellaneous data models that are shared by the app's services."""

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
        description='The unique identifier of the document.',
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

    doc_metadata: Annotated[
        DocumentMetadata, Field(description='The metadata of the survey document.')
    ]
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
