"""Contains a class that retrieves stats and components from a parsed document.

See :class:`~paper_enrichment_agent.common.models.document.Document` for the document structure.
"""

from collections.abc import Iterator

from paper_enrichment_agent.common.models import document as doc_models


class DocumentGetter:
    """Retrieves stats and components from a parsed document.

    See :class:`~paper_enrichment_agent.common.models.document.DocumentComponent` for the path
        definition.
    """

    def __init__(self, document: doc_models.Document) -> None:
        self._document = document

    def get_path_of_component(self, target_component: doc_models.DocumentComponent) -> str:
        """Returns the path of the specified component in the document.

        Raises:
            ValueError: If the component is not found in the document.
        """

        for section in self._document.sections:
            for path, component in self._iter_section_components(section, root_path='/sections'):
                if component == target_component:
                    return path

        for footnote in self._document.footnotes:
            for path, component in self._iter_component(footnote, root_path='/footnotes'):
                if component == target_component:
                    return path

        raise ValueError(f'Component not found in the document: {target_component}')

    def get_component_by_path(self, target_path: str) -> doc_models.DocumentComponent:
        """Returns the component at the specified path in the document.

        Raises:
            ValueError: If the component is not found in the document.
        """

        for section in self._document.sections:
            for path, component in self._iter_section_components(section, root_path='/sections'):
                if path == target_path:
                    return component

        for footnote in self._document.footnotes:
            for path, component in self._iter_component(footnote, root_path='/footnotes'):
                if path == target_path:
                    return component

        raise ValueError(f'Component not found in the document: {target_path}')

    def iter_components_of_type[T: doc_models.DocumentComponent](
        self, component_type: type[T]
    ) -> Iterator[T]:
        """Returns all components of the specified type from the document."""

        for section in self._document.sections:
            for _, component in self._iter_section_components(section, root_path='/sections'):
                if isinstance(component, component_type):
                    yield component

        for footnote in self._document.footnotes:
            for _, component in self._iter_component(footnote, root_path='/footnotes'):
                if isinstance(component, component_type):
                    yield component

    def _iter_component(
        self, component: doc_models.DocumentComponent, root_path: str
    ) -> Iterator[tuple[str, doc_models.DocumentComponent]]:
        """Recursively iterates over all components in a document component.

        For each component, yields a tuple containing the path to the component and the component.
        """

        if isinstance(component, doc_models.Section):
            yield from self._iter_section_components(component, root_path)

        elif isinstance(component, doc_models.Paragraph):
            yield from self._iter_paragraph_components(component, root_path)

        elif isinstance(component, doc_models.Figure):
            yield from self._iter_figure_components(component, root_path)

        else:
            yield f'{root_path}/{component.component_id}', component

    def _iter_section_components(
        self,
        section: doc_models.Section,
        root_path: str = '',
    ) -> Iterator[tuple[str, doc_models.DocumentComponent]]:
        """Recursively iterates over all components in a section.

        For each component, yields a tuple containing the path to the component and the component.
        """

        section_path = f'{root_path}/{section.component_id}'

        yield section_path, section

        for component in section.components:
            yield from self._iter_component(component, section_path)

    def _iter_paragraph_components(
        self, paragraph: doc_models.Paragraph, root_path: str
    ) -> Iterator[tuple[str, doc_models.DocumentComponent]]:
        """Iterates over all DocumentComponent-type elements in a paragraph.

        For each element, yields a tuple containing the path to the element and the element.
        """

        yield f'{root_path}/{paragraph.component_id}', paragraph

        for element in paragraph.elements:
            if isinstance(element, doc_models.DocumentComponent):
                yield f'{root_path}/{paragraph.component_id}/{element.component_id}', element

    def _iter_figure_components(
        self, figure: doc_models.Figure, root_path: str
    ) -> Iterator[tuple[str, doc_models.DocumentComponent]]:
        """Iterates over all subfigures in a figure.

        For each subfigure, yields a tuple containing the path to the subfigure and the subfigure.
        """

        yield f'{root_path}/{figure.component_id}', figure

        for subfigure in figure.subfigures:
            yield f'{root_path}/{figure.component_id}/{subfigure.component_id}', subfigure
