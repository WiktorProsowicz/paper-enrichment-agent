"""Contains the serializer converting documents into zip archives with Markdown content.

The :class:`MarkdownSerializer` renders the document of a
:class:`~paper_enrichment_agent.main_survey_enricher.components.doc_serializers.core.SerializationEnvironment`
into a `paper.md` file placed next to the environment's `images/` folder, and packs both into the
environment's archive. The serializer is synchronous, since its work is bound to the CPU and the
file system, so asynchronous callers should run it in a separate thread.

Exported classes:
    MarkdownSerializer
"""

import itertools
import pathlib
import zipfile

from paper_enrichment_agent.common.document_manipulators import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.core import (
    SerializationEnvironment,
)


class MarkdownSerializer:
    """Serializes a document into a zip archive with Markdown content.

    The serializer works within the given environment, which determines the serialized document,
    the output directory and the path of the created archive. The images of the document are
    expected to be placed in the environment by the caller before the serialization, so a single
    instance of the serializer should be used for a single environment.

    The serialization raises `SerializationError` on failure.
    """

    class SerializationError(Exception):
        """Raised when serializing a document into a Markdown archive fails."""

    # Name of the rendered Markdown file, relative to the output directory.
    _MARKDOWN_FILE_NAME = 'paper.md'

    def __init__(self, environment: SerializationEnvironment) -> None:

        self._environment = environment
        self._document = environment.document

        # Maps the paths of image subfigures to the paths of their files, relative to the Markdown
        # file.
        self._image_files = {
            path: environment.image_path_for_component(path)
            .relative_to(environment.output_dir)
            .as_posix()
            for path in environment.image_component_paths
        }

        # Maps the paths of footnotes to their numbers in the rendered document.
        self._footnote_numbers = {
            f'/footnotes/{footnote.component_id}': number
            for number, footnote in enumerate(self._document.footnotes, start=1)
        }

        # Paths of the components targeted by element references. The components are preceded by
        # HTML anchors, so that the references can link to them.
        self._anchored_paths = self._collect_element_reference_targets()

    def serialize(self) -> pathlib.Path:
        """Serializes the document into a zip archive.

        The archive contains a `paper.md` file with the rendered document and an `images/` folder
        with the images of the document's image subfigures, which the Markdown references by
        relative paths. The content of the archive is kept in the environment's `output_dir`, so
        that it can be inspected before the environment is removed.

        Returns:
            The path of the created archive, i.e. the environment's `archive_path`.

        Raises:
            SerializationError: If writing the Markdown file or creating the archive fails.
        """

        try:
            (self._environment.output_dir / self._MARKDOWN_FILE_NAME).write_text(
                self._render_document(), encoding='utf-8'
            )

            self._create_archive()

        except OSError as e:
            raise self.SerializationError(
                f'Failed to serialize the document "{self._document.title}" into '
                f'{self._environment.archive_path}: {e}'
            ) from e

        return self._environment.archive_path

    def _collect_element_reference_targets(self) -> set[str]:
        """Returns the target paths of all element references in the document."""

        doc_getter = DocumentGetter(self._document)

        # Captions are not part of the document tree, so their references are collected manually
        caption_paragraphs = (
            caption
            for figure in doc_getter.iter_components_of_type(doc_models.Figure)
            for caption in (
                figure.caption,
                *(subfigure.caption for subfigure in figure.subfigures),
            )
            if isinstance(caption, doc_models.Paragraph)
        )
        caption_references = (
            element
            for caption in caption_paragraphs
            for element in caption.elements
            if isinstance(element, doc_models.Reference)
        )

        return {
            reference.target
            for reference in itertools.chain(
                doc_getter.iter_components_of_type(doc_models.Reference), caption_references
            )
            if reference.ref_type == 'element'
        }

    def _render_document(self) -> str:
        """Renders the whole document as Markdown."""

        blocks = [f'# {self._document.title}', '## Abstract', self._document.abstract]

        for section in self._document.sections:
            blocks.extend(self._render_section(section, '/sections', 0))

        if self._document.footnotes:
            blocks.append('## Footnotes')
            blocks.extend(self._render_footnote(footnote) for footnote in self._document.footnotes)

        if self._document.referenced_papers:
            blocks.append('## References')
            blocks.append(
                '\n'.join(
                    f'- <a id="{_citation_anchor(paper_id)}"></a>{description}'
                    for paper_id, description in self._document.referenced_papers
                )
            )

        return '\n\n'.join(blocks) + '\n'

    def _render_section(
        self, section: doc_models.Section, parent_path: str, depth: int
    ) -> list[str]:
        """Renders a section and its components as a list of Markdown blocks.

        The top-level sections are rendered as level-2 headings, since the level 1 is taken by the
        document's title. Markdown supports at most 6 heading levels, so deeper sections are
        flattened.
        """

        section_path = f'{parent_path}/{section.component_id}'
        heading = f'{"#" * min(depth + 2, 6)} {section.title}'

        blocks = [self._with_anchor(heading, section_path)]

        for component in section.components:
            blocks.extend(self._render_section_component(component, section_path, depth))

        return blocks

    def _render_section_component(
        self, component: doc_models.Section.SectionComponent, parent_path: str, depth: int
    ) -> list[str]:
        """Renders a component of a section as a list of Markdown blocks."""

        if isinstance(component, doc_models.Section):
            return self._render_section(component, parent_path, depth + 1)

        component_path = f'{parent_path}/{component.component_id}'

        if isinstance(component, doc_models.Paragraph):
            block = self._render_paragraph(component)

        elif isinstance(component, doc_models.Figure):
            block = self._render_figure(component, component_path)

        elif isinstance(component, doc_models.MathExpression):
            block = self._render_block_math(component)

        else:
            block = self._render_list(component)

        return [self._with_anchor(block, component_path)]

    def _render_paragraph(self, paragraph: doc_models.Paragraph) -> str:
        """Renders the inline elements of a paragraph, grouping consecutive citations."""

        def is_citation(element: doc_models.Paragraph.InlineParagraphElement) -> bool:
            return isinstance(element, doc_models.Reference) and element.ref_type == 'citation'

        rendered_elements: list[str] = []

        for is_sequence_of_citations, elements in itertools.groupby(
            paragraph.elements, key=is_citation
        ):
            if is_sequence_of_citations:
                citations = (
                    f'[{element.content_text}](#{_citation_anchor(element.target)})'  # type: ignore[union-attr]
                    for element in elements
                )
                rendered_elements.append(f'[{", ".join(citations)}]')

            else:
                rendered_elements.extend(
                    self._render_inline_element(element) for element in elements
                )

        return ''.join(rendered_elements)

    def _render_inline_element(self, element: doc_models.Paragraph.InlineParagraphElement) -> str:
        """Renders a non-citation inline element of a paragraph."""

        if isinstance(element, doc_models.MathExpression):
            return f'${element.expression}$' if element.format == 'LaTeX' else element.expression

        if not isinstance(element, doc_models.Reference):
            return element

        if element.ref_type == 'link':
            return f'[{element.content_text}]({element.target})'

        if footnote_number := self._footnote_numbers.get(element.target):
            return f'[^{footnote_number}]'

        return f'[{element.content_text}](#{_path_to_anchor(element.target)})'

    def _render_figure(self, figure: doc_models.Figure, figure_path: str) -> str:
        """Renders the subfigures of a figure with their captions, then the figure's caption."""

        blocks = []

        for subfigure in figure.subfigures:
            if isinstance(subfigure, doc_models.ImgSubfigure):
                image_file = self._image_files[f'{figure_path}/{subfigure.component_id}']
                blocks.append(f'![{subfigure.component_id}]({image_file})')

            else:
                blocks.append(subfigure.table_contents)

            if subfigure_caption := self._render_caption(subfigure.caption):
                blocks.append(f'*{subfigure_caption}*')

        if figure_caption := self._render_caption(figure.caption):
            blocks.append(f'*{figure_caption}*')

        return '\n\n'.join(blocks)

    def _render_caption(self, caption: doc_models.FigureCaption | None) -> str:
        """Renders a caption of a figure or a subfigure, rendering a missing one as empty."""

        if caption is None:
            return ''

        if isinstance(caption, doc_models.Paragraph):
            return self._render_paragraph(caption)

        return caption

    def _render_block_math(self, equation: doc_models.MathExpression) -> str:
        """Renders a standalone mathematical expression."""

        if equation.format == 'LaTeX':
            return f'$$\n{equation.expression}\n$$'

        # MathML is valid inline HTML, so it can be embedded as is.
        return equation.expression

    def _render_list(self, list_component: doc_models.List) -> str:
        """Renders an ordered or unordered list."""

        return '\n'.join(
            f'{f"{number}." if list_component.ordered else "-"} {self._render_paragraph(item)}'
            for number, item in enumerate(list_component.items, start=1)
        )

    def _render_footnote(self, footnote: doc_models.Document.FootnoteType) -> str:
        """Renders the definition of a footnote.

        Footnotes composed of sections (e.g. the enriched ones) are rendered as a bold title
        followed by their components. The continuation lines are indented, so that they belong to
        the footnote.
        """

        footnote_path = f'/footnotes/{footnote.component_id}'

        if isinstance(footnote, doc_models.Paragraph):
            body = self._render_paragraph(footnote)

        else:
            body = '\n\n'.join(
                [
                    f'**{footnote.title}**',
                    *(
                        block
                        for component in footnote.components
                        # Headings of nested sections are rendered at the lowest level.
                        for block in self._render_section_component(component, footnote_path, 4)
                    ),
                ]
            )

        first_line, *continuation_lines = body.split('\n')
        indented_body = '\n'.join(
            [first_line, *(f'    {line}' if line else line for line in continuation_lines)]
        )

        return f'[^{self._footnote_numbers[footnote_path]}]: {indented_body}'

    def _with_anchor(self, block: str, component_path: str) -> str:
        """Precedes the block with an HTML anchor if the component is targeted by a reference."""

        if component_path not in self._anchored_paths:
            return block

        return f'<a id="{_path_to_anchor(component_path)}"></a>\n{block}'

    def _create_archive(self) -> None:
        """Packs the content of the output directory into the archive."""

        output_dir = self._environment.output_dir

        with zipfile.ZipFile(
            self._environment.archive_path, 'w', compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for file_path in sorted(output_dir.rglob('*')):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.relative_to(output_dir))


def _path_to_anchor(component_path: str) -> str:
    """Converts a component path into an identifier usable as an HTML anchor."""

    return component_path.strip('/').replace('/', '-')


def _citation_anchor(paper_id: str) -> str:
    """Returns the anchor of a referenced paper's entry in the references."""

    return f'cite-{paper_id}'
