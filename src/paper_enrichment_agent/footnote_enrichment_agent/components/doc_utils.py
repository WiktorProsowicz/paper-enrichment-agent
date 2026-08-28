"""Contains utilities used for document processing in the `footnote_enrichment_agent` service."""

import dataclasses
import itertools
from collections.abc import Iterator

from paper_enrichment_agent.common.models import document as doc_models


@dataclasses.dataclass
class StringifiedParagraphElement:
    """An element of the paragraph, rendered as a string.

    Attributes:
        doc_component: The component from the original document.
        str_content: The stringified content of the element.
    """

    doc_component: doc_models.Paragraph.InlineParagraphElement | None
    str_content: str


def stringify_paragraph_elements(
    paragraph: doc_models.Paragraph,
) -> list[StringifiedParagraphElement]:
    """Converts the elements of a paragraph into a list of stringified elements.

    The citation references are grouped together and represented as a single stringified element,
    while other elements are stringified individually. Links and math expressions are preserved so
     that they can be referenced later if needed.
    """

    str_elements: list[StringifiedParagraphElement] = []

    def is_citation(element: doc_models.Paragraph.InlineParagraphElement) -> bool:
        return isinstance(element, doc_models.Reference) and element.ref_type == 'citation'

    for is_sequence_of_citations, elements in itertools.groupby(
        paragraph.elements, key=is_citation
    ):
        if is_sequence_of_citations:
            str_elements.append(
                StringifiedParagraphElement(
                    str_content='[' + ', '.join(e.content_text for e in elements) + ']',  # type: ignore[union-attr]
                    doc_component=None,
                )
            )
        else:
            for element in elements:
                if isinstance(element, doc_models.MathExpression):
                    str_elements.append(
                        StringifiedParagraphElement(
                            str_content=f'${element.expression}$', doc_component=element
                        )
                    )

                elif isinstance(element, doc_models.Reference):
                    str_elements.append(
                        StringifiedParagraphElement(
                            str_content=element.content_text,
                            doc_component=element if element.ref_type == 'link' else None,
                        )
                    )

                else:
                    str_elements.append(
                        StringifiedParagraphElement(str_content=element, doc_component=None)
                    )

    return str_elements


def iter_str_elements_boundaries(
    str_elements: list[StringifiedParagraphElement],
) -> Iterator[tuple[int, int, StringifiedParagraphElement]]:
    """Yields stringified elements along with (start, end) indices in the concatenated string."""

    for start_idx, el in zip(
        itertools.accumulate((len(e.str_content) for e in str_elements), initial=0),
        str_elements,
        strict=False,
    ):
        yield start_idx, start_idx + len(el.str_content), el


def get_context_of_citation_in_paragraph(
    paragraph: doc_models.Paragraph, citation: doc_models.Reference, n_context_chars: int
) -> str:
    """Extracts the textual context surrounding a citation within a paragraph.

    The context is extracted from the stringified form of the paragraph. The context includes
    roughly `n_context_chars` / 2 characters before and after the citation, capped by the start and end
    of the paragraph.
    """

    str_elements = stringify_paragraph_elements(paragraph)
    paragraph_str = ''.join(e.str_content for e in str_elements)

    try:
        citation_start, citation_end = next(
            (start_idx, end_idx)
            for start_idx, end_idx, e in iter_str_elements_boundaries(str_elements)
            if citation is e.doc_component
        )
    except StopIteration as e:
        raise ValueError('Citation not found in paragraph elements.') from e

    context_start_idx = max(0, citation_start - n_context_chars // 2)
    context_end_idx = min(len(paragraph_str), citation_end + n_context_chars // 2)

    return paragraph_str[context_start_idx:context_end_idx]
