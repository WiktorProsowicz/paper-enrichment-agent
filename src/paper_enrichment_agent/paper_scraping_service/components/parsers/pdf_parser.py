"""Contains a module that parses PDF papers and converts them into a structured format.

See :class:`~paper_enrichment_agent.common.models.document.Document`. The PDF is downloaded from
its URL and converted by an external marker API server (`infrastructure/marker-api`), which returns
the paper as a JSON tree of blocks. The blocks ar then mapped onto the document components.

Exported classes:
    MarkerBlock
    PdfParser
"""

import itertools
import pathlib
import re
import urllib.parse
from typing import Annotated

import bs4
import httpx
import pydantic
from bs4.element import NavigableString, PageElement
from pydantic import Field

from paper_enrichment_agent.common.models import document as doc_models


class MarkerBlock(pydantic.BaseModel):
    """A single block of the marker's JSON output (`marker.renderers.json.JSONBlockOutput`).

    Only the fields used by :class:`PdfParser` are declared, the remaining ones (geometry, debug
    data) are ignored.
    """

    model_config = pydantic.ConfigDict(extra='ignore')

    id: Annotated[
        str,
        Field(
            description='Path-like id of the block, e.g. `/page/6/Figure/3`. Empty for the root.'
        ),
    ] = ''
    block_type: Annotated[str, Field(description='Type of the block, e.g. `Text`, `Table`.')]
    html: Annotated[
        str,
        Field(description='HTML of the block. Groups reference children via `<content-ref>`.'),
    ] = ''
    children: Annotated[list['MarkerBlock'] | None, Field(description='Child blocks, if any.')] = (
        None
    )
    images: Annotated[
        dict[str, str] | None,
        Field(description='Maps block ids to their base64-encoded images.'),
    ] = None


class PdfParser:
    """A parser that converts PDF papers into a structured format using the marker API.

    The parser is synchronous and blocking (network I/O and CPU-bound HTML parsing), so async
    callers should offload :meth:`parse` with `asyncio.to_thread`. The internal state of the
    parser is built during parsing, therefore a single instance of the parser should be used for
    parsing a single paper. The injected HTTP client may be shared between parser instances.

    The marker output is mapped as follows:
        - The first section header is the title of the document. The blocks placed between the
          title and the next header (authors, affiliations) are dropped.
        - The nesting of sections is derived from their numbering (`2.1`, `A.1.1`), falling back
          to the rank of the HTML heading tag for unnumbered headers, since the heading levels
          and the section hierarchy reported by marker are not reliable.
        - The contents of the "Abstract" section become the abstract, and the entries of the
          "References" / "Bibliography" section become the referenced papers.
        - Page headers, page footers and page footnotes are dropped.
        - Images are embedded in the subfigures as `data:` URIs.
        - Citations are kept as plain text, i.e. they are not resolved into references.

    The parser raises `ParsingError` if the paper cannot be downloaded, converted or parsed.
    """

    class ParsingError(Exception):
        """Raised when an error occurs during the parsing of a PDF paper."""

    class Configuration(pydantic.BaseModel):
        """Configuration of the parser."""

        marker_api_url: Annotated[
            str,
            Field(description='The base URL of the marker API server.', min_length=1),
        ]
        marker_api_token: Annotated[
            pydantic.SecretStr,
            Field(description='The bearer token authorizing requests to the marker API.'),
        ]
        marker_api_timeout: Annotated[
            float,
            Field(description='The timeout (in seconds) of a single PDF conversion.', gt=0),
        ]
        download_timeout: Annotated[
            float, Field(description='The timeout (in seconds) of the PDF download.', gt=0)
        ]
        max_pdf_size_bytes: Annotated[
            int, Field(description='The maximum accepted size of the PDF file in bytes.', gt=0)
        ]

    _PDF_MAGIC = b'%PDF'

    # Blocks not belonging to the main text of the paper
    _SKIPPED_BLOCK_TYPES = frozenset({'PageHeader', 'PageFooter', 'Footnote', 'TableOfContents'})

    # Blocks rendered as images
    _IMAGE_BLOCK_TYPES = frozenset({'Figure', 'Picture', 'Diagram'})

    # Blocks rendered as plain paragraphs
    _TEXT_BLOCK_TYPES = frozenset({'Text', 'Code', 'ComplexRegion', 'Handwriting', 'Form'})

    # Section numbering such as `2`, `2.1`, `A`, `A.1.1`, optionally followed by a dot
    _SECTION_NUMBER_PATTERN = re.compile(r'^(?:\d+|[A-Z])(?:\.\d+)*\.?(?=\s+\S)')

    # Superscripts that mark footnotes, e.g. `4`, `*`, `†`
    _FOOTNOTE_MARKER_PATTERN = re.compile(r'^[\d\s,*†‡§¶]+$')

    # Labels of numbered bibliography entries, e.g. `[12]`
    _BIB_LABEL_PATTERN = re.compile(r'^\[(\d+)\]\s*')

    _ABSTRACT_TITLES = frozenset({'abstract'})
    _BIBLIOGRAPHY_TITLES = frozenset({'references', 'bibliography'})

    def __init__(self, config: Configuration, http_client: httpx.Client) -> None:

        self._config = config
        self._http_client = http_client

        # Maps the marker block ids to the respective blocks
        self._blocks_by_id: dict[str, MarkerBlock] = {}

        # Maps the levels of the HTML heading tags of unnumbered headers to the section depths
        self._depths_by_heading_level: dict[int, int] = {}

    def parse(self, pdf_url: str) -> doc_models.Document:
        """Downloads a PDF paper, converts it with the marker API and parses the result.

        Args:
            pdf_url: The URL of the PDF file.

        Returns:
            The parsed document.

        Raises:
            ParsingError: If the PDF cannot be downloaded, converted or parsed.
        """

        pdf_content = self._download_pdf(pdf_url)
        marker_output = self._convert_pdf(pdf_content, file_name=self._get_file_name(pdf_url))

        return self.parse_marker_output(marker_output)

    def parse_marker_output(self, marker_output: MarkerBlock) -> doc_models.Document:
        """Converts the marker's JSON output into the document structure.

        Args:
            marker_output: The root block of the marker output.

        Returns:
            The parsed document.

        Raises:
            ParsingError: If the output contains no section headers.
        """

        blocks = self._flatten_page_blocks(marker_output)
        headers = [block for block in blocks if block.block_type == 'SectionHeader']

        if not headers:
            raise self.ParsingError('The converted PDF contains no section headers.')

        title_header, *section_headers = headers
        self._depths_by_heading_level = self._infer_depths_by_heading_level(section_headers)

        abstract = ''
        referenced_papers: list[tuple[str, str]] = []
        sections: list[doc_models.Section] = []

        # Stack of (depth, section) of the currently open sections
        open_sections: list[tuple[int, doc_models.Section]] = []
        previous_depth = 0

        title_index = blocks.index(title_header)
        header_indices = [
            i for i, block in enumerate(blocks) if block.block_type == 'SectionHeader'
        ]

        for header_index, next_header_index in itertools.pairwise([*header_indices, len(blocks)]):
            if header_index <= title_index:
                continue

            header = blocks[header_index]
            section_blocks = blocks[header_index + 1 : next_header_index]
            section_title = self._extract_clean_text(self._to_soup(header.html))
            normalized_title = self._strip_section_number(section_title).lower()

            if normalized_title in self._ABSTRACT_TITLES and not abstract:
                abstract = self._extract_text_of_blocks(section_blocks)
                continue

            if normalized_title in self._BIBLIOGRAPHY_TITLES:
                referenced_papers.extend(
                    self._extract_bibliography(section_blocks, offset=len(referenced_papers))
                )
                continue

            depth = self._get_section_depth(header, section_title, previous_depth)
            previous_depth = depth

            section = doc_models.Section(
                component_id=self._to_component_id(header.id),
                title=section_title,
                components=self._decode_section_blocks(section_blocks),
            )

            while open_sections and open_sections[-1][0] >= depth:
                open_sections.pop()

            if open_sections:
                open_sections[-1][1].components.append(section)
            else:
                sections.append(section)

            open_sections.append((depth, section))

        return doc_models.Document(
            title=self._extract_clean_text(self._to_soup(title_header.html)),
            abstract=abstract,
            description=None,
            sections=sections,
            footnotes=[],
            referenced_papers=referenced_papers,
        )

    def _download_pdf(self, pdf_url: str) -> bytes:
        """Downloads the PDF file and validates its size and format."""

        try:
            response = self._http_client.get(
                pdf_url, timeout=self._config.download_timeout, follow_redirects=True
            )
            response.raise_for_status()

        except httpx.HTTPError as e:
            raise self.ParsingError(f'Failed to download the PDF from {pdf_url}: {e}') from e

        content = response.content

        if len(content) > self._config.max_pdf_size_bytes:
            raise self.ParsingError(
                f'The PDF from {pdf_url} has {len(content)} bytes, exceeding the limit of '
                f'{self._config.max_pdf_size_bytes} bytes.'
            )

        if not content.startswith(self._PDF_MAGIC):
            raise self.ParsingError(f'The file downloaded from {pdf_url} is not a PDF document.')

        return content

    def _convert_pdf(self, pdf_content: bytes, file_name: str) -> MarkerBlock:
        """Sends the PDF to the marker API and validates the returned JSON."""

        # The token is attached per request, so that it is never sent to the PDF hosts
        token = self._config.marker_api_token.get_secret_value()

        try:
            response = self._http_client.post(
                f'{self._config.marker_api_url.rstrip("/")}/convert_pdf',
                files={'file': (file_name, pdf_content, 'application/pdf')},
                headers={'Authorization': f'Bearer {token}'},
                timeout=self._config.marker_api_timeout,
            )
            response.raise_for_status()

            return MarkerBlock.model_validate(response.json())

        except httpx.HTTPError as e:
            raise self.ParsingError(f'Failed to convert the PDF with the marker API: {e}') from e

        except (ValueError, pydantic.ValidationError) as e:
            raise self.ParsingError(f'The marker API returned an invalid output: {e}') from e

    def _flatten_page_blocks(self, marker_output: MarkerBlock) -> list[MarkerBlock]:
        """Returns the top-level blocks of all pages in reading order.

        All the blocks, including the members of groups, are registered by their ids.
        """

        def register(block: MarkerBlock) -> None:
            self._blocks_by_id[block.id] = block
            for child in block.children or []:
                register(child)

        register(marker_output)

        return [
            block
            for page in marker_output.children or []
            for block in page.children or []
            if block.block_type not in self._SKIPPED_BLOCK_TYPES
        ]

    def _decode_section_blocks(
        self, blocks: list[MarkerBlock]
    ) -> list[doc_models.Section.SectionComponent]:
        """Decodes the blocks placed directly under a section header."""

        components: list[doc_models.Section.SectionComponent] = []
        pending_visual: MarkerBlock | None = None

        def flush_pending_visual(caption_block: MarkerBlock | None) -> None:
            nonlocal pending_visual
            if pending_visual is not None:
                figure = self._decode_figure(
                    f'{self._to_component_id(pending_visual.id)}-figure',
                    [pending_visual],
                    [caption_block] if caption_block else [],
                )
                if figure is not None:
                    components.append(figure)
            pending_visual = None

        for block in blocks:
            if block.block_type == 'Caption':
                # A caption is attached to the immediately preceding standalone visual
                flush_pending_visual(block)
                continue

            flush_pending_visual(None)

            if block.block_type in self._IMAGE_BLOCK_TYPES or block.block_type == 'Table':
                pending_visual = block

            elif block.block_type in ('FigureGroup', 'TableGroup', 'PictureGroup'):
                members = self._get_group_members(block)
                figure = self._decode_figure(
                    self._to_component_id(block.id),
                    [m for m in members if m.block_type != 'Caption'],
                    [m for m in members if m.block_type == 'Caption'],
                )
                if figure is not None:
                    components.append(figure)

            elif block.block_type in self._TEXT_BLOCK_TYPES:
                paragraph = self._decode_paragraph(block)
                if paragraph.elements:
                    self._append_paragraph(components, paragraph)

            elif block.block_type == 'Equation':
                components.extend(self._decode_equations(block))

            elif block.block_type == 'ListGroup':
                decoded_list = self._decode_list(block)
                if decoded_list.items:
                    components.append(decoded_list)

        flush_pending_visual(None)

        return components

    def _append_paragraph(
        self,
        components: list[doc_models.Section.SectionComponent],
        paragraph: doc_models.Paragraph,
    ) -> None:
        """Appends the paragraph or merges it into the paragraph it continues.

        Marker splits paragraphs at page breaks. The first paragraph of a page continues the last
        paragraph of a previous page (possibly separated by figures placed in between), unless
        that paragraph ends with a terminal punctuation mark.
        """

        for component in reversed(components):
            if isinstance(component, doc_models.Figure):
                continue

            if (
                isinstance(component, doc_models.Paragraph)
                and self._get_page(component.component_id) != self._get_page(paragraph.component_id)
                and not self._ends_sentence(component)
            ):
                component.elements = self._merge_strings(
                    [*component.elements, ' ', *paragraph.elements]
                )
                return

            break

        components.append(paragraph)

    @staticmethod
    def _ends_sentence(paragraph: doc_models.Paragraph) -> bool:
        """Checks whether the paragraph ends with a terminal punctuation mark."""

        last_element = paragraph.elements[-1]

        return isinstance(last_element, str) and last_element.rstrip()[-1:] in ('.', '!', '?', ':')

    @staticmethod
    def _get_page(component_id: str) -> str:
        """Returns the page of the component, encoded in its id (`page-6-Text-3`)."""

        return component_id.split('-')[1] if component_id.startswith('page-') else ''

    def _decode_paragraph(self, block: MarkerBlock) -> doc_models.Paragraph:
        """Decodes a text block into a paragraph."""

        return doc_models.Paragraph(
            component_id=self._to_component_id(block.id),
            elements=self._decode_inline_elements(self._to_soup(block.html)),
        )

    def _decode_inline_elements(
        self, tag: bs4.Tag
    ) -> list[doc_models.Paragraph.InlineParagraphElement]:
        """Decodes the inline contents of the tag into paragraph elements."""

        elements: list[doc_models.Paragraph.InlineParagraphElement] = []

        def visit(node: PageElement) -> None:
            if isinstance(node, NavigableString):
                elements.append(str(node))

            elif isinstance(node, bs4.Tag):
                if node.name == 'math':
                    elements.append(
                        doc_models.MathExpression(
                            expression=node.get_text().strip(), format='LaTeX'
                        )
                    )

                elif node.name == 'a' and node.get('href'):
                    elements.append(
                        doc_models.Reference(
                            ref_type='link',
                            target=str(node['href']),
                            content_text=self._extract_clean_text(node),
                        )
                    )

                elif node.name == 'br':
                    elements.append(' ')

                elif node.name == 'sup' and self._FOOTNOTE_MARKER_PATTERN.match(node.get_text()):
                    pass

                else:
                    # Block-level siblings (e.g. consecutive `<p>`) are separated by spaces
                    if node.name in ('p', 'li', 'div') and elements:
                        elements.append(' ')
                    for child in node.children:
                        visit(child)

        visit(tag)

        return self._merge_strings(elements)

    def _merge_strings(
        self, elements: list[doc_models.Paragraph.InlineParagraphElement]
    ) -> list[doc_models.Paragraph.InlineParagraphElement]:
        """Merges consecutive strings, normalizes whitespace and strips the paragraph's ends."""

        merged: list[doc_models.Paragraph.InlineParagraphElement] = []

        for is_sequence_of_strings, group in itertools.groupby(
            elements, key=lambda e: isinstance(e, str)
        ):
            if is_sequence_of_strings:
                merged.append(re.sub(r'\s+', ' ', ''.join(group)))  # type: ignore[arg-type]
            else:
                merged.extend(group)

        if merged and isinstance(merged[0], str):
            merged[0] = merged[0].lstrip()
        if merged and isinstance(merged[-1], str):
            merged[-1] = merged[-1].rstrip()

        return [element for element in merged if element != '']

    def _decode_equations(self, block: MarkerBlock) -> list[doc_models.MathExpression]:
        """Decodes the display math expressions of an equation block."""

        soup = self._to_soup(block.html)
        expressions = [math.get_text().strip() for math in soup.find_all('math')]

        if not expressions:
            expressions = [soup.get_text().strip()]

        return [
            doc_models.MathExpression(
                component_id=self._to_component_id(block.id)
                + (f'-{i}' if len(expressions) > 1 else ''),
                expression=expression,
                format='LaTeX',
            )
            for i, expression in enumerate(expressions)
            if expression
        ]

    def _decode_list(self, block: MarkerBlock) -> doc_models.List:
        """Decodes a list group block into a list."""

        soup = self._to_soup(block.html)
        list_tag = soup.find(['ul', 'ol'])
        component_id = self._to_component_id(block.id)

        decoded_list = doc_models.List(
            component_id=component_id,
            ordered=isinstance(list_tag, bs4.Tag) and list_tag.name == 'ol',
            items=[],
        )

        item_tags = soup.find_all('li') or soup.find_all('p')

        for i, item_tag in enumerate(item_tags):
            elements = self._decode_inline_elements(item_tag)

            if elements and isinstance(elements[0], str):
                elements[0] = re.sub(r'^[•\-–∙◦▪]\s*', '', elements[0])
                if not elements[0]:
                    elements.pop(0)

            if elements:
                decoded_list.items.append(
                    doc_models.Paragraph(component_id=f'{component_id}-{i}', elements=elements)
                )

        return decoded_list

    def _decode_figure(
        self,
        figure_id: str,
        visual_blocks: list[MarkerBlock],
        caption_blocks: list[MarkerBlock],
    ) -> doc_models.Figure | None:
        """Decodes image / table blocks and their captions into a figure.

        Args:
            figure_id: The component id of the figure.
            visual_blocks: The image / table blocks, becoming the subfigures.
            caption_blocks: The caption blocks, joined into the figure's caption.

        Returns:
            The figure, or None if none of the visual blocks could be decoded.
        """

        subfigures: list[doc_models.TableSubfigure | doc_models.ImgSubfigure] = []

        for block in visual_blocks:
            if block.block_type == 'Table':
                subfigures.append(
                    doc_models.TableSubfigure(
                        component_id=self._to_component_id(block.id),
                        table_contents=block.html,
                        caption=None,
                    )
                )

            elif block.block_type in self._IMAGE_BLOCK_TYPES and (
                image := (block.images or {}).get(block.id)
            ):
                subfigures.append(
                    doc_models.ImgSubfigure(
                        component_id=self._to_component_id(block.id),
                        image_src=self._to_data_uri(image),
                        caption=None,
                    )
                )

        if not subfigures:
            return None

        caption_elements: list[doc_models.Paragraph.InlineParagraphElement] = []
        for caption_block in caption_blocks:
            caption_soup = self._to_soup(caption_block.html)
            caption_elements.extend([' ', *self._decode_inline_elements(caption_soup)])
        caption_elements = self._merge_strings(caption_elements)

        caption: doc_models.FigureCaption
        if all(isinstance(element, str) for element in caption_elements):
            caption = ''.join(caption_elements)  # type: ignore[arg-type]
        else:
            caption = doc_models.Paragraph(elements=caption_elements)

        return doc_models.Figure(
            component_id=figure_id,
            subfigures=subfigures,
            caption=caption,
        )

    def _get_group_members(self, group: MarkerBlock) -> list[MarkerBlock]:
        """Returns the members of a group block in their order of appearance."""

        if group.children:
            return group.children

        soup = self._to_soup(group.html)

        return [
            self._blocks_by_id[str(ref['src'])]
            for ref in soup.find_all('content-ref')
            if str(ref.get('src')) in self._blocks_by_id
        ]

    def _extract_bibliography(
        self, blocks: list[MarkerBlock], offset: int
    ) -> list[tuple[str, str]]:
        """Extracts the bibliography entries from the blocks of the references section.

        Numbered entries (`[12] ...`) get the id `bib-12`, the others are numbered in order of
        appearance.
        """

        bibliography: list[tuple[str, str]] = []

        for block in blocks:
            soup = self._to_soup(block.html)
            entry_tags = soup.find_all('li') or soup.find_all('p')

            for entry_tag in entry_tags:
                text = self._extract_clean_text(entry_tag)
                if not text:
                    continue

                if label_match := self._BIB_LABEL_PATTERN.match(text):
                    entry_id = f'bib-{label_match.group(1)}'
                    text = text[label_match.end() :]
                else:
                    entry_id = f'bib-{offset + len(bibliography) + 1}'

                bibliography.append((entry_id, text))

        return bibliography

    def _extract_text_of_blocks(self, blocks: list[MarkerBlock]) -> str:
        """Returns the plain text of the textual blocks, with inline math kept as LaTeX."""

        texts: list[str] = []

        for block in blocks:
            if block.block_type in self._TEXT_BLOCK_TYPES:
                texts.append(
                    ''.join(
                        element
                        if isinstance(element, str)
                        else f'${element.expression}$'
                        if isinstance(element, doc_models.MathExpression)
                        else element.content_text
                        for element in self._decode_inline_elements(self._to_soup(block.html))
                    )
                )

        return '\n'.join(text for text in texts if text)

    def _get_section_depth(self, header: MarkerBlock, title: str, previous_depth: int) -> int:
        """Returns the nesting depth (1 = top level) of the section."""

        if number_match := self._SECTION_NUMBER_PATTERN.match(title):
            return number_match.group(0).rstrip('.').count('.') + 1

        depth = self._depths_by_heading_level.get(self._get_heading_level(header), 1)

        return max(1, min(depth, previous_depth + 1))

    def _infer_depths_by_heading_level(self, headers: list[MarkerBlock]) -> dict[int, int]:
        """Infers the section depths implied by the heading tags, used for unnumbered headers.

        A heading level used by numbered headers maps to the most common depth of their numbers
        (e.g. `<h2>` -> 1 if most `<h2>` headers are top-level sections). The remaining levels
        map to their rank among all the heading levels used in the paper.
        """

        numbered_depths: dict[int, list[int]] = {}

        for header in headers:
            title = self._extract_clean_text(self._to_soup(header.html))
            if number_match := self._SECTION_NUMBER_PATTERN.match(title):
                numbered_depths.setdefault(self._get_heading_level(header), []).append(
                    number_match.group(0).rstrip('.').count('.') + 1
                )

        levels = sorted({self._get_heading_level(header) for header in headers})

        return {
            level: (
                max(set(depths), key=depths.count)
                if (depths := numbered_depths.get(level))
                else rank
            )
            for rank, level in enumerate(levels, start=1)
        }

    def _get_heading_level(self, header: MarkerBlock) -> int:
        """Returns the level of the HTML heading tag of the header (`<h2>` -> 2)."""

        heading_tag = self._to_soup(header.html).find(re.compile(r'^h[1-6]$'))

        return int(heading_tag.name[1]) if isinstance(heading_tag, bs4.Tag) else 6

    def _strip_section_number(self, title: str) -> str:
        """Removes the leading section number from the title."""

        return self._SECTION_NUMBER_PATTERN.sub('', title).strip()

    def _to_soup(self, html: str) -> bs4.BeautifulSoup:
        return bs4.BeautifulSoup(html, 'html.parser')

    def _extract_clean_text(self, tag: bs4.Tag) -> str:
        """Extracts clean text from a tag, normalizing the whitespace."""

        return re.sub(r'\s+', ' ', tag.get_text(separator=' ', strip=True)).strip()

    @staticmethod
    def _to_component_id(block_id: str) -> str:
        """Converts a marker block id (`/page/6/Figure/3`) into a path-safe component id."""

        return block_id.strip('/').replace('/', '-')

    @staticmethod
    def _to_data_uri(image_base64: str) -> str:
        """Wraps a base64-encoded image into a `data:` URI, detecting its format."""

        mime_type = 'image/png' if image_base64.startswith('iVBOR') else 'image/jpeg'

        return f'data:{mime_type};base64,{image_base64}'

    @staticmethod
    def _get_file_name(pdf_url: str) -> str:
        """Returns the name of the PDF file sent to the marker API."""

        name = pathlib.PurePosixPath(urllib.parse.urlparse(pdf_url).path).name or 'paper'

        return name if name.endswith('.pdf') else f'{name}.pdf'
