"""Contains a module that parses arXiv papers and converts them into a structured format.

See :class:`~paper_enrichment_agent.common.models.document.Document`. The arXiv papers are parsed
from their html format using the ar5iv.labs.arxiv.org/html/[id] endpoint.
"""

from typing import cast

import bs4
import requests
from bs4.element import NavigableString

from paper_enrichment_agent.common import document_getter
from paper_enrichment_agent.common.models import document as doc_models


class ArxivParser:
    """A parser that converts arXiv papers into a structured format.

    The internal state of the parser is built during parsing, therefore a single instance of the
    parser should be used for parsing a single paper.
    """

    class ParsingError(Exception):
        """Raised when an error occurs during the parsing of an arXiv paper."""

    def __init__(self) -> None:

        # Maps component ids to the respectice structs.
        self._referencable_components: dict[str, doc_models.DocumentComponent] = {}

    def parse(self, paper_id: str) -> doc_models.Document:
        """Downloads and parses an arXiv paper into a structured format."""

        try:
            response = requests.get(f'https://ar5iv.labs.arxiv.org/html/{paper_id}')
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise self.ParsingError(
                f'Failed to download arXiv paper with ID {paper_id}: {e}'
            ) from e

        soup = bs4.BeautifulSoup(response.content, 'html.parser')

        document = doc_models.Document(
            abstract=self._extract_abstract(soup),
            sections=[
                self._decode_section(section_tag) for section_tag in soup.select('div.ltx_section')
            ],
            footnotes=[],
            referenced_papers=[],
        )

        doc_getter = document_getter.DocumentGetter(document)

        for reference in doc_getter.iter_components_of_type(doc_models.Reference):
            if reference.target.startswith('#'):
                component_id = reference.target[1:]

                if component_id in self._referencable_components:
                    reference.target = doc_getter.get_path_of_component(
                        self._referencable_components[component_id]
                    )

                else:
                    raise self.ParsingError(
                        f'Failed to resolve reference target with component ID {component_id}.'
                    )

        return document

    def _decode_section(self, section_tag: bs4.Tag) -> doc_models.Section:
        """Decodes a section from the arXiv paper.

        If the `section_tag` is not a 'leaf' section, but rather a composite section, the method
        will recursively decode its child sections.
        """

        title_tag = self._safe_select_one(section_tag, '.ltx_title')

        title = title_tag.get_text(strip=True)

        components: list[doc_models.Section.SectionComponent] = []

        for child in title_tag.find_next_siblings():
            if 'ltx_para' in child['class']:
                components.append(self._decode_paragraph(child))

            elif 'ltx_figure' in child['class']:
                caption_tag = self._safe_select_one(child, 'p.ltx_p')

                figure = doc_models.Figure(
                    component_id=str(child['id']),
                    image_paths=[str(img_tag['src']) for img_tag in child.select('img.ltx_img')],
                    caption=caption_tag.get_text(strip=True),
                    description='Figure',
                )

                self._referencable_components[figure.component_id] = figure
                components.append(figure)

            elif 'ltx_equation' in child['class'] or 'ltx_equationgroup' in child['class']:
                for equation_tag in child.select('.ltx_equation'):
                    expression = ' '.join(
                        str(part['alt_text']) for part in equation_tag.select('math.ltx_Math')
                    )
                    equation = doc_models.MathExpression(
                        component_id=str(equation_tag['id']),
                        description='Equation',
                        expression=expression,
                        format='LaTeX',
                    )

                    self._referencable_components[equation.component_id] = equation
                    components.append(equation)

        return doc_models.Section(
            component_id=str(section_tag['id']), title=title, components=components
        )

    def _decode_paragraph(self, paragraph_tag: bs4.Tag) -> doc_models.Paragraph:
        """Decodes a paragraph from .ltx_para div.

        The following paragraph components are parsed:
        - text
        - inline math
        - equations
        """

        paragraph_contents: list[doc_models.Paragraph.InlineParagraphElement] = []

        for child in self._safe_select_one(paragraph_tag, 'p.ltx_p').children:
            if isinstance(child, NavigableString):
                paragraph_contents.append(str(child))
                continue

            child = cast(bs4.Tag, child)

            if 'ltx_cite' in child['class']:
                for link_tag in child.select('a.ltx_ref'):
                    paragraph_contents.append(
                        doc_models.Reference(
                            description='Citation reference',
                            target=str(link_tag['href']),
                            content_text=link_tag.get_text(),
                        )
                    )

            elif 'ltx_Math' in child['class']:
                math_exp = doc_models.MathExpression(
                    component_id=str(child['id']),
                    expression=str(child['alttext']),
                    description='Inline math expression',
                    format='LaTeX',
                )

                self._referencable_components[math_exp.component_id] = math_exp
                paragraph_contents.append(math_exp)

            elif 'ltx_ref' in child['class'] and 'ltx_url' not in child['class']:
                paragraph_contents.append(
                    doc_models.Reference(
                        description='Element reference',
                        target=str(child['href']),
                        content_text=child.get_text(),
                    )
                )

        return doc_models.Paragraph(
            component_id=str(paragraph_tag['id']),
            description='Paragraph',
            elements=paragraph_contents,
        )

    def _extract_abstract(self, soup: bs4.BeautifulSoup) -> str:
        """Extracts the abstract from the arXiv paper."""

        abstract_tag = soup.select_one('div.ltx_abstract > p')

        if not abstract_tag:
            raise self.ParsingError('Failed to extract abstract from arXiv paper.')

        return abstract_tag.get_text(strip=True)

    def _safe_select_one(self, tag: bs4.Tag, query: str) -> bs4.Tag:
        """Performs a select query with failure handling."""

        result = tag.select_one(query)

        if not result:
            tag_description = f'<{tag.name} id={str(tag["id"])}>'

            raise self.ParsingError(
                f'Failed select operation on tag {tag_description} for query "{query}".'
            )

        return result
