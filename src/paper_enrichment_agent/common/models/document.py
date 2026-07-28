"""Contains the definitions of document components."""

from typing import Annotated, Literal

import pydantic
from pydantic import Field


class DocumentComponent(pydantic.BaseModel):
    """Base class for all components in a document."""

    component_id: Annotated[str, Field(description='Identifier of the component.')]
    description: Annotated[str | None, Field(description='Optional description of the component.')]


class Reference(DocumentComponent):
    """Represents a reference to another component in the document."""

    target_path: Annotated[
        str,
        Field(
            description=(
                'A path to the target component being referenced.'
                'A component path is a Posix-style path that uniquely identifies a'
                'component in the document.'
            )
        ),
    ]


class MathExpression(DocumentComponent):
    """Represents a mathematical expression in a document.

    A mathematical expression is a component that represents a mathematical formula or equation.
    It can be represented in various formats, such as LaTeX, MathML, or plain text. The specific
    format used for the expression is determined by the `format` attribute.
    """

    expression: Annotated[str, Field(description='The mathematical expression.')]
    format: Annotated[
        Literal['LaTeX', 'MathML'], Field(description='The format of the mathematical expression.')
    ]


class Figure(DocumentComponent):
    """Represents a figure in a document.

    A figure is a component that represents a visual element, such as an image, chart, or diagram.
    """

    image_paths: Annotated[
        list[str], Field(description='List of paths to the images associated with the figure.')
    ]
    caption: Annotated[str, Field(description='The caption of the figure.')]


class Paragraph(DocumentComponent):
    """Represents a textual paragraph in a document.

    A paragraph may contain multiple sentences, citations, links and inline math expressions. It
    is the role of a downstream service to parse the paragraph into its constituent components,
    if needed.
    """

    type InlineParagraphElement = str | Reference | MathExpression

    elements: Annotated[
        list[InlineParagraphElement], Field(description='List of elements in the paragraph.')
    ]


class Footnote(DocumentComponent):
    """Represents a footnote in a document.

    A footnote is an element, referenced in the document, that provides additional information or
    context to the main content. It is designed to be placed at the end of the chapter / document.
    A footnote is not necessarily a single text, but rather an entire named section.
    """

    title: Annotated[str, Field(description='The title of the footnote.')]
    components: Annotated[
        list[DocumentComponent], Field(description='List of components in the footnote.')
    ]
    referrer: Annotated[
        str | None, Field(description='The component that references the footnote.')
    ]


class Section(DocumentComponent):
    """Represents a section in a document.

    A section is a component that represents a distinct part of the document, typically containing
    a heading and one or more paragraphs. Sections can be nested to create a hierarchical
    structure within the document.
    """

    title: Annotated[str, Field(description='The title of the section.')]
    components: Annotated[
        list[DocumentComponent], Field(description='List of components in the section.')
    ]


class Document(pydantic.BaseModel):
    """Represents a research paper document.

    The class is a universal interface that can be used to represent either a survey during the
    enrichment process or a document referenced by the survey.
    """

    abstract: Annotated[str, Field(description='The abstract of the document.')]
    sections: Annotated[list[Section], Field(description='List of sections in the document.')]
    footnotes: Annotated[list[Footnote], Field(description='List of footnotes in the document.')]
    referenced_papers: Annotated[
        list['Document' | str], Field(description='List of reference papers in the document.')
    ]
