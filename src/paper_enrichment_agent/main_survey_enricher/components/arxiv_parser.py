"""Contains a module that parses arXiv papers and converts them into a structured format.

See :class:`~paper_enrichment_agent.common.models.document.Document`. The arXiv papers are parsed
from their html format using the ar5iv.labs.arxiv.org/html/[id] endpoint.
"""

import itertools
import re
from typing import cast

import bs4
import requests
from bs4.element import NavigableString

from paper_enrichment_agent.common import document_manipulators
from paper_enrichment_agent.common.models import document as doc_models


class ArxivParser:
    """A parser that converts arXiv papers into a structured format.

    The internal state of the parser is built during parsing, therefore a single instance of the
    parser should be used for parsing a single paper.

    The parser raises `ParsingError` if it encounters an error during the parsing of the arXiv
    paper.
    """

    _ALLOWED_SECTION_CLASSES = (
        'ltx_section',
        'ltx_appendix',
        'ltx_paragraph',
        'ltx_subsection',
        'ltx_subsubsection',
    )

    class ParsingError(Exception):
        """Raised when an error occurs during the parsing of an arXiv paper."""

    def __init__(self) -> None:

        # Maps component ids to the respective structs
        self._referencable_components: dict[str, doc_models.DocumentComponent] = {}

        # Keeps track of the bibitem ids that have been processed
        self._bib_items_ids: set[str] = set()

        self._footnotes: list[doc_models.Document.FootnoteType] = []

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
            description=f'arXiv:{paper_id}',
            sections=[
                self._decode_section(section_tag)
                for section_tag in soup.select('article > section')
                if any(cl in list(section_tag['class']) for cl in self._ALLOWED_SECTION_CLASSES)
            ],
            footnotes=[],
            referenced_papers=self._extract_bibliography(soup),
        )

        document.footnotes = self._footnotes.copy()

        self._resolve_references(document)

        self._merge_consecutive_strings_in_paragraphs(document)

        # Fix the image sources to point to the ar5iv.labs.arxiv.org domain.
        for image in document_manipulators.DocumentGetter(document).iter_components_of_type(
            doc_models.ImgSubfigure
        ):
            image.image_src = f'https://ar5iv.labs.arxiv.org{image.image_src}'

        return document

    def _resolve_references(self, document: doc_models.Document) -> None:
        """Fixes the reference targets and types after having processed the document."""

        doc_getter = document_manipulators.DocumentGetter(document)

        for reference in doc_getter.iter_components_of_type(doc_models.Reference):
            if reference.ref_type == 'element':
                # Both internal and external links are represented as .ltx_ref. Links that do not
                # reference an element id have to be treated as external links.
                if not reference.target.startswith('#'):
                    reference.ref_type = 'link'
                    reference.description = 'External link reference'
                    continue

                component_id = reference.target[1:]

                # Links to subfigures are folded into the parent figure
                if re.match(r'.+\.sf\d+$', component_id) or re.match(r'.+\.st\d+$', component_id):
                    component_id = '.'.join(component_id.split('.')[:-1])

                if component_id not in self._referencable_components:
                    raise self.ParsingError(
                        f'Failed to resolve reference target with component ID "{component_id}".'
                    )

                reference.target = doc_getter.get_path_of_component(
                    self._referencable_components[component_id]
                )

            if reference.ref_type == 'citation' and reference.target not in self._bib_items_ids:
                raise self.ParsingError(
                    f'Failed to resolve citation reference target with ID "{reference.target}".'
                )

    def _merge_consecutive_strings_in_paragraphs(self, document: doc_models.Document) -> None:
        """Merges consecutive str elements in paragraphs into a single str element."""

        doc_getter = document_manipulators.DocumentGetter(document)

        for paragraph in doc_getter.iter_components_of_type(doc_models.Paragraph):
            new_elements: list[doc_models.Paragraph.InlineParagraphElement] = []

            for is_sequence_of_strings, elements in itertools.groupby(
                paragraph.elements, key=lambda e: isinstance(e, str)
            ):
                if is_sequence_of_strings:
                    new_elements.append(''.join(elements))  # type: ignore[arg-type]
                else:
                    new_elements.extend(elements)

            paragraph.elements = new_elements

    def _decode_section(self, section_tag: bs4.Tag) -> doc_models.Section:
        """Decodes a section from the arXiv paper.

        If the `section_tag` is not a 'leaf' section, but rather a composite section, the method
        will recursively decode its child sections.
        """

        title_tag = self._safe_select_one(section_tag, '.ltx_title')

        title = self._extract_clean_text(title_tag)

        components: list[doc_models.Section.SectionComponent] = []

        for child in title_tag.find_next_siblings():
            if 'ltx_para' in child['class']:
                for paragraph_tag in child.children:
                    if not isinstance(paragraph_tag, bs4.Tag):
                        continue

                    if 'ltx_p' in paragraph_tag['class']:
                        components.append(self._decode_paragraph(paragraph_tag))

                    elif 'ltx_equation' in paragraph_tag['class']:
                        equation = self._decode_equation(paragraph_tag)

                        self._referencable_components[equation.component_id] = equation
                        components.append(equation)

                    elif 'ltx_equationgroup' in paragraph_tag['class']:
                        collected_equations: list[doc_models.MathExpression] = []

                        for equation_tag in paragraph_tag.select('tbody'):
                            equation = self._decode_equation(equation_tag)

                            self._referencable_components[equation.component_id] = equation
                            collected_equations.append(equation)

                        self._referencable_components[str(paragraph_tag['id'])] = (
                            collected_equations[0]
                        )
                        components.extend(collected_equations)

                    elif paragraph_tag.name == 'ol' or paragraph_tag.name == 'ul':
                        components.append(self._decode_list(paragraph_tag))

            elif child.name == 'figure':
                figure = self._decode_figure(child)
                self._referencable_components[figure.component_id] = figure
                components.append(figure)

            elif child.name == 'section' and any(
                cl in list(child['class']) for cl in self._ALLOWED_SECTION_CLASSES
            ):
                components.append(self._decode_section(child))

        section = doc_models.Section(
            component_id=str(section_tag['id']), title=title, components=components
        )

        self._referencable_components[section.component_id] = section

        return section

    def _decode_list(self, list_tag: bs4.Tag) -> doc_models.List:
        """Decodes a list from the given `list` tag paper."""

        decoded_list = doc_models.List(
            component_id=str(list_tag['id']),
            ordered=list_tag.name == 'ol',
            items=[],
        )

        for list_item_tag in list_tag.select('li.ltx_item'):
            for item_para_tag in list_item_tag.select('p.ltx_p'):
                paragraph_item = self._decode_paragraph(item_para_tag)
                paragraph_item.description = f'Original item id: {list_item_tag["id"]}'
                decoded_list.items.append(paragraph_item)

        return decoded_list

    def _decode_equation(self, equation_tag: bs4.Tag) -> doc_models.MathExpression:
        """Decodes an equation from the given `equation` tag paper."""

        expression = ' '.join(str(part['alttext']) for part in equation_tag.select('math.ltx_Math'))

        return doc_models.MathExpression(
            component_id=str(equation_tag['id']),
            description='Equation',
            expression=expression,
            format='LaTeX',
        )

    def _decode_figure(self, figure_tag: bs4.Tag) -> doc_models.Figure:
        """Decodes a figure from the given `figure` tag paper."""

        main_caption = self._safe_select_one(figure_tag, ':scope > figcaption.ltx_caption')

        figure = doc_models.Figure(
            component_id=str(figure_tag['id']),
            subfigures=[],
            caption=self._extract_clean_text(main_caption),
            description='Figure',
        )

        for subfigure_tag in figure_tag.select('img.ltx_graphics, table.ltx_tabular'):
            subcaption_tag = subfigure_tag.find_next_sibling(class_='ltx_caption')

            subcaption = self._extract_clean_text(subcaption_tag) if subcaption_tag else None

            if 'img' in subfigure_tag.name:
                figure.subfigures.append(
                    doc_models.ImgSubfigure(
                        description='Image subfigure',
                        image_src=str(subfigure_tag['src']),
                        caption=subcaption,
                    )
                )
            else:
                figure.subfigures.append(
                    doc_models.TableSubfigure(
                        description='Table subfigure',
                        table_contents=re.sub(r'\s+', ' ', str(subfigure_tag)),
                        caption=subcaption,
                    )
                )

        return figure

    def _decode_paragraph(self, paragraph_tag: bs4.Tag) -> doc_models.Paragraph:
        """Decodes a paragraph from .ltx_para div.

        The following paragraph components are parsed:
        - text
        - inline math
        - equations
        """

        paragraph_contents: list[doc_models.Paragraph.InlineParagraphElement] = []

        for child_idx, child in enumerate(paragraph_tag.children):
            if isinstance(child, NavigableString):
                paragraph_contents.append(re.sub(r'\s+', ' ', str(child)))
                continue

            child = cast(bs4.Tag, child)

            if 'ltx_cite' in child['class']:
                for link_idx, link_tag in enumerate(child.select('a.ltx_ref')):
                    paragraph_contents.append(
                        doc_models.Reference(
                            component_id=f'{child_idx}.{link_idx}',
                            ref_type='citation',
                            description='Citation reference',
                            target=str(link_tag['href'])[1:],
                            content_text=link_tag.get_text(separator=' ', strip=True),
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

            elif 'ltx_ref' in child['class'] and child.name == 'a':
                paragraph_contents.append(
                    doc_models.Reference(
                        component_id=str(child_idx),
                        ref_type='element',
                        description='Element reference',
                        target=str(child['href']),
                        content_text=child.get_text(separator=' ', strip=True),
                    )
                )

            elif 'ltx_note' in child['class']:
                footnote = self._decode_paragraph(self._safe_select_one(child, '.ltx_note_content'))
                footnote.description = 'Footnote'
                footnote.component_id = str(child['id'])

                self._referencable_components[footnote.component_id] = footnote
                self._footnotes.append(footnote)

                paragraph_contents.append(
                    doc_models.Reference(
                        component_id=str(child_idx),
                        ref_type='element',
                        description='Footnote reference',
                        target=f'#{footnote.component_id}',
                        content_text=self._safe_select_one(
                            child, ':scope > .ltx_note_mark'
                        ).get_text(strip=True),
                    )
                )

            else:
                decoded_stringlike = self._try_decode_stringlike_paragraph_element(child)

                if decoded_stringlike is not None:
                    paragraph_contents.append(decoded_stringlike)

        return doc_models.Paragraph(
            component_id=str(paragraph_tag['id']) if 'id' in paragraph_tag.attrs else '',
            description='Paragraph',
            elements=paragraph_contents,
        )

    def _extract_abstract(self, soup: bs4.BeautifulSoup) -> str:
        """Extracts the abstract from the arXiv paper."""

        abstract_tag = soup.select_one('div.ltx_abstract > p')

        if not abstract_tag:
            raise self.ParsingError('Failed to extract abstract from arXiv paper.')

        return re.sub(r'\s+', ' ', abstract_tag.get_text().strip())

    def _try_decode_stringlike_paragraph_element(self, element_tag: bs4.Tag) -> str | None:
        """Decodes a paragraph element that is not a string, yet should be represented as such.

        The string-like elements are not special document components (e.g. math expressions) and
        are to be handled as scalar HTML elements, such as <strong>.
        """

        if 'ltx_font_italic' in element_tag['class']:
            return f'<em>{self._extract_clean_text(element_tag)}</em>'

        if 'ltx_font_bold' in element_tag['class']:
            return f'<strong>{self._extract_clean_text(element_tag)}</strong>'

        if 'ltx_url' in element_tag['class']:
            return self._extract_clean_text(element_tag)

        return None

    def _extract_bibliography(self, soup: bs4.BeautifulSoup) -> list[tuple[str, str]]:
        """Extracts the bibliography from the arXiv paper."""

        bibliography_tag = self._safe_select_one(soup, 'section.ltx_bibliography')
        bibliography: list[tuple[str, str]] = []

        for bib_item in bibliography_tag.select('li.ltx_bibitem'):
            self._bib_items_ids.add(str(bib_item['id']))

            bibliography.append(
                (
                    str(bib_item['id']),
                    ' '.join(
                        self._extract_clean_text(part)
                        for part in bib_item.select('span.ltx_bibblock')
                    ).strip(),
                )
            )

        return bibliography

    def _safe_select_one(self, tag: bs4.Tag, query: str) -> bs4.Tag:
        """Performs a select query with failure handling."""

        result = tag.select_one(query)

        if not result:
            tag_description = f'<{tag.name} id={str(tag["id"])}>'

            raise self.ParsingError(
                f'Failed select operation on tag {tag_description} for query "{query}".'
            )

        return result

    def _extract_clean_text(self, tag: bs4.Tag) -> str:
        """Extracts clean text from a tag, removing any unwanted characters."""

        return re.sub(r'\s+', ' ', tag.get_text(separator=' ', strip=True))
