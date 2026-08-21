"""Utilities for managing setup/teardown lifecycle of footnote enrichment tools."""

from collections.abc import Generator
from contextlib import contextmanager
import difflib

import fastmcp

from paper_enrichment_agent.common import document_manipulators
from paper_enrichment_agent.common.models import document as doc_models


class EnrichmentTools:
    """Represents the state of the footnote-enrichment tools.

    The state represents two main components:
    1. The context built from the reference document, which is used to provide content for footnote
    enrichment.
    2. The state of the built footnote, modified by the enrichment agent.
    """

    type DocumentTree = str | list[DocumentTree] | dict[str, DocumentTree]

    def __init__(self, reference_document: doc_models.Document) -> None:

        self._document = reference_document
        self._doc_getter = document_manipulators.DocumentGetter(self._document)
        self._footnote = doc_models.Section(title='', components=[])

    @property
    def footnote(self) -> doc_models.Section:
        """Returns the current state of the footnote being enriched."""
        return self._footnote

    def get_document_tree(self) -> 'DocumentTree':
        """Returns a tree representation of the reference document.

        The tree representation is a nested structure that captures the hierarchy of sections,
        paragraphs, and other components in the document. It makes it possible to determine the id
        of an arbitrary component and a path to it.
        """
        return {
            'sections': [
                self._convert_component_to_tree(section) for section in self._document.sections
            ],
            'footnotes': [
                self._convert_component_to_tree(footnote) for footnote in self._document.footnotes
            ],
        }

    def set_footnote_title(self, title: str) -> None:
        """Sets the title of the footnote being enriched.

        The title should be set once after the content of the footnote is established and should
        reflect the role of the footnote in explaining the reference document.

        Args:
            title: The title to be set.

        Example:
            survey title: Attention Is All You Need
            citation context: The two most commonly used attention functions are additive attention
                [2], and dot-product (multiplicative) attention.
            reference title: Neural machine translation by jointly learning to align and translate.
            good footnote title: Description of additive attention mechanism
        """
        self._footnote.title = title

    def get_paragraph_content(self, path_to_paragraph: str) -> str:
        """Returns the content of a paragraph in the reference document.

        Args:
            path_to_paragraph: The path to the paragraph in the reference document.

        Returns:
            The textual content of the paragraph.
        """

        paragraph = self._obtain_paragraph(path_to_paragraph)
        return paragraph.elements[0]  # type: ignore

    def get_figure_details(self, path_to_figure: str) -> str:
        """Returns the details describing a given figure in the reference document.

        Args:
            path_to_figure: The path to the figure in the reference document.

        Returns:
            The textual description of the figure's contents, including its caption and subfigures.

        Example output:
            Caption: A diagram illustrating the architecture of the Transformer model.
            Subfigures:
                ImgSubfigure: A schematic representation of the multi-head attention mechanism.
                TableSubfigure: A table showing the hyperparameters used in the Transformer model.
                ...
        """

        figure = self._doc_getter.get_component_by_path(path_to_figure)

        if not isinstance(figure, doc_models.Figure):
            raise ValueError(f'Component at path {path_to_figure} is not a figure.')

        return (
            f'Caption: {figure.caption}\n'
            + 'Subfigures:\n'
            + '\n'.join(
                f'\t{subfigure.__class__.__name__}: {subfigure.caption}'
                for subfigure in figure.subfigures
            )
        )

    def extract_paragraph_citation(
        self, path_to_paragraph: str, begin_anchor: str, end_anchor: str
    ) -> None:
        """Extracts the citation from a given paragraph and moves it to the footnote.

        The citation is identified by the provided begin and end text anchors that help
        unambiguously locate the desired fragment within the paragraph. The extracted citation is
        added to the enriched footnote. Citations help provide necessary explanation of the
        referenced document, yet they reduce the amount of content that has to be read to
        understand the described content.

        Both the begin and end anchors should ideally span several words to avoid ambiguity.

        Args:
            path_to_paragraph: The path to the paragraph in the reference document.
            begin_anchor: The beginning text anchor of the citation to be extracted.
            end_anchor: The ending text anchor of the citation to be extracted.
        """

        paragraph_content: str = self._obtain_paragraph(path_to_paragraph).elements[0]  # type: ignore

        match_begin = difflib.SequenceMatcher(
            None, paragraph_content, begin_anchor
        ).find_longest_match(0, len(paragraph_content), 0, len(begin_anchor))

        if match_begin.size != len(begin_anchor):
            raise ValueError('Begin anchor not found in paragraph content.')

        match_end = difflib.SequenceMatcher(None, paragraph_content, end_anchor).find_longest_match(
            0, len(paragraph_content), 0, len(end_anchor)
        )

        if match_end.size != len(end_anchor):
            raise ValueError('End anchor not found in paragraph content.')

        citation = paragraph_content[match_begin.a + match_begin.size : match_end.a]

        self._footnote.components.append(doc_models.Paragraph(elements=[citation]))

    def extract_figure(self, path_to_figure: str, subfigures_ids: list[str] | None) -> None:
        """Extracts a figure from the reference document and adds it to the footnote.

        The extracted figure is added to the enriched footnote. Figures provide visual explanation
        of the concepts described in the reference document. They should be included in the enriched
        footnote especially if they are somehow referenced in the textual content that has been
        previously extracted from the reference document or are strictly relevant to the concepts
        the footnote aims to explain.

        Args:
            path_to_figure: The path to the figure in the reference document.
            subfigures_ids: The list of subfigure ids to be extracted. If None, all subfigures are
                extracted.
        """

        figure = self._doc_getter.get_component_by_path(path_to_figure)

        if not isinstance(figure, doc_models.Figure):
            raise ValueError(f'Component at path {path_to_figure} is not a figure.')

        if subfigures_ids is not None:
            existing_subfigures_ids = {subfigure.component_id for subfigure in figure.subfigures}
            invalid_ids = set(subfigures_ids) - existing_subfigures_ids

            if invalid_ids:
                raise ValueError(
                    f'Some subfigure ids do not exist in the figure: {invalid_ids}. '
                    f'Existing subfigure ids: {existing_subfigures_ids}.'
                )

            subfigures = [
                subfigure
                for subfigure in figure.subfigures
                if subfigure.component_id in subfigures_ids
            ]
        else:
            subfigures = figure.subfigures

        extracted_figure = doc_models.Figure(
            caption=figure.caption,
            subfigures=subfigures,
            component_id=figure.component_id,
        )

        self._footnote.components.append(extracted_figure)

    def _obtain_paragraph(self, path_to_paragraph: str) -> doc_models.Paragraph:
        """Locates and validates a paragraph in the reference document."""

        paragraph = self._doc_getter.get_component_by_path(path_to_paragraph)

        if not isinstance(paragraph, doc_models.Paragraph):
            raise ValueError(f'Component at path {path_to_paragraph} is not a paragraph.')

        return paragraph

    def _convert_component_to_tree(self, component: doc_models.DocumentComponent) -> 'DocumentTree':
        """Converts a document component to its tree representation."""

        base_repr: dict[str, EnrichmentTools.DocumentTree] = {
            'type': component.__class__.__name__,
            'id': component.component_id,
        }

        if isinstance(component, doc_models.Section):
            return {
                **base_repr,
                'title': component.title,
                'components': [self._convert_component_to_tree(c) for c in component.components],
            }

        if isinstance(component, doc_models.List):
            return {
                **base_repr,
                'items': [self._convert_component_to_tree(i) for i in component.items],
            }

        if isinstance(component, doc_models.Figure):
            return {
                **base_repr,
                'caption': component.caption,
            }

        return base_repr


class EnrichmentContextManager:
    """Manages the setup and teardown of the footnote-enrichment MCP tools state.

    The enrichment context manager creates and stores the state of the footnote-enrichment tools,
    which are exposed within MCP server to the LLM-based enrichment agent. The tools state is
    initialized for each agent session and may be reused on enrichment continuation.
    """

    class EnrichmentContextManagerError(Exception):
        """Base class for exceptions raised by the `EnrichmentContextManager`."""

    def __init__(self) -> None:

        self._tools_states: dict[str, EnrichmentTools] = {}
        self._mcp_endpoints: dict[str, fastmcp.FastMCP] = {}

    def get_mcp_for_agent_session(self, session_id: str) -> fastmcp.FastMCP:
        """Returns the MCP endpoint for a footnote-enrichment agent session.

        Args:
            session_id: The unique identifier of the footnote-enrichment agent session.
        """

        if session_id not in self._mcp_endpoints:
            raise self.EnrichmentContextManagerError(
                f'MCP endpoint for session {session_id} is not set up.'
            )

        return self._mcp_endpoints[session_id]

    @contextmanager
    def setup_mcp_for_agent_session(
        self, session_id: str, reference_document: doc_models.Document | None
    ) -> Generator[None, None, None]:
        """Sets up a child MCP endpoint for a footnote-enrichment agent session.

        The exposed MCP is attached to the primary MCP root endpoint and delegates the
        footnote-enrichment tool calls to the created / reused enrichment tools state.

        Args:
            session_id: The unique identifier of the footnote-enrichment agent session.
            reference_document: The document referenced by the survey, which is used to
                initialize the enrichment tools state. If None, the existing tools state is reused.
        """

        if not reference_document:
            if session_id not in self._tools_states:
                raise self.EnrichmentContextManagerError(
                    f'Cannot setup MCP for session {session_id} without a reference document.'
                )

        else:
            self._tools_states[session_id] = EnrichmentTools(reference_document)

        tools_state = self._tools_states[session_id]

        mcp_endpoint = fastmcp.FastMCP(
            name=f'footnote_enrichment_tools_{session_id}',
            on_duplicate='error',
            strict_input_validation=True,
        )

        mcp_endpoint.tool(tools_state.get_document_tree, name='get_document_tree')
        mcp_endpoint.tool(tools_state.set_footnote_title, name='set_footnote_title')
        mcp_endpoint.tool(tools_state.get_paragraph_content, name='get_paragraph_content')
        mcp_endpoint.tool(tools_state.get_figure_details, name='get_figure_details')
        mcp_endpoint.tool(tools_state.extract_paragraph_citation, name='extract_paragraph_citation')
        mcp_endpoint.tool(tools_state.extract_figure, name='extract_figure')

        self._mcp_endpoints[session_id] = mcp_endpoint

        yield

        self._mcp_endpoints.pop(session_id)
