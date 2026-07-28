"""Contains a module that parses arXiv papers and converts them into a structured format.

See :class:`~paper_enrichment_agent.common.models.document.Document`. The arXiv papers are parsed
from their html format using the ar5iv.labs.arxiv.org/html/[id] endpoint.
"""

import re
from typing import cast

import requests
import pydantic
import bs4

from paper_enrichment_agent.common.models import document as doc_models


class ArxivParserCfg(pydantic.BaseModel):
    """Configuration for the ArxivParser class."""


class ArxivParser:
    """A parser that converts arXiv papers into a structured format."""

    class ParsingError(Exception):
        """Raised when an error occurs during the parsing of an arXiv paper."""

    def __init__(self) -> None:
        pass

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

        return doc_models.Document(
            abstract=self._extract_abstract(soup),
            sections=[
                self._decode_section(section_tag) for section_tag in soup.select('div.ltx_section')
            ],
            footnotes=[],
            referenced_papers=[],
        )

    def _decode_section(self, section_tag: bs4.Tag) -> doc_models.Section:
        """Decodes a section from the arXiv paper.

        If the `section_tag` is not a 'leaf' section, but rather a composite section, the method
        will recursively decode its child sections.
        """

        title_tag = section_tag.select_one('.ltx_title')

        if not title_tag:
            raise self.ParsingError(
                f'Failed to decode section title from section tag(id: {section_tag.get("id")}.'
            )

        title = title_tag.get_text(strip=True)

        components = []

        for child in title_tag.find_next_siblings():
            if 'ltx_para' in child.get('class', []):
                components.append(self._decode_paragraph(child))

        return doc_models.Section(title=title, components=components)

    def _decode_paragraph(self, paragraph_tag: bs4.Tag) -> doc_models.Paragraph:
        """Decodes a paragraph from .ltx_para div.

        The following paragraph components are parsed:
        - text
        - inline math
        - equations
        - figures
        """

        paragraph_contents: list[doc_models.Paragraph.InlineParagraphElement] = []

        for child in paragraph_tag.select_one('p.ltx_p').contents:
            if isinstance(child, bs4.NavigableString):
                paragraph_contents.append(str(child))
                continue

            child = cast(bs4.Tag, child)

            if 'ltx_cite' in child.get('class', []):
                link_tag = child.select_one('a')
                paragraph_contents.append(
                    doc_models.Reference(
                        ref_id=,
                        text=child.get_text(strip=True),
                    )
                )

            elif 

    def _extract_abstract(self, soup: bs4.BeautifulSoup) -> str:
        """Extracts the abstract from the arXiv paper."""

        abstract_tag = soup.select_one('div.ltx_abstract > p')

        if not abstract_tag:
            raise self.ParsingError('Failed to extract abstract from arXiv paper.')

        return abstract_tag.get_text(strip=True)
