"""Contains the definitions of document components."""

import uuid
from typing import Annotated, Literal

import pydantic
from pydantic import Field


class DocumentComponent(pydantic.BaseModel):
    """Base class for all components in a document.

    By design each component should have a unique identifier in scope of the children of its parent
    in the document tree. This allows to unambiguously reference a component by a `path`. A `path`
    should be a posix-tyle absolute path from the root of the document tree to the referenced
    component.
    """

    component_id: Annotated[str, Field(description='Identifier of the component.')] = (
        uuid.uuid4().hex
    )

    description: Annotated[
        str | None, Field(description='Optional description of the component.')
    ] = None

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(component_id="{self.component_id}")'


class Reference(DocumentComponent):
    """Represents a reference to another component in the document."""

    target: Annotated[
        str,
        Field(
            description=(
                'The target of the reference.'
                'This can be either a string identifier or a path to the target component.'
            )
        ),
    ]
    content_text: Annotated[str, Field(description='Textual content of the reference.')]


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

    type SectionComponent = Paragraph | Figure | Section | MathExpression

    title: Annotated[str, Field(description='The title of the section.')]
    components: Annotated[
        list[SectionComponent], Field(description='List of components in the section.')
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
        list[str], Field(description='List of reference papers in the document.')
    ]
