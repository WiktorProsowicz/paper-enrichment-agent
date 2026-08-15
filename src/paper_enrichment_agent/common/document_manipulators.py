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
            for path, component in _iter_component(section, root_path='/sections'):
                if component == target_component:
                    return path

        for footnote in self._document.footnotes:
            for path, component in _iter_component(footnote, root_path='/footnotes'):
                if component == target_component:
                    return path

        raise ValueError(f'Component not found in the document: {target_component}')

    def get_component_by_path(self, target_path: str) -> doc_models.DocumentComponent:
        """Returns the component at the specified path in the document.

        Raises:
            ValueError: If the component is not found in the document.
        """

        for section in self._document.sections:
            for path, component in _iter_component(section, root_path='/sections'):
                if path == target_path:
                    return component

        for footnote in self._document.footnotes:
            for path, component in _iter_component(footnote, root_path='/footnotes'):
                if path == target_path:
                    return component

        raise ValueError(f'Component not found in the document: {target_path}')

    def iter_components_of_type[T: doc_models.DocumentComponent](
        self, component_type: type[T]
    ) -> Iterator[T]:
        """Returns all components of the specified type from the document."""

        for section in self._document.sections:
            for _, component in _iter_component(section, root_path='/sections'):
                if isinstance(component, component_type):
                    yield component

        for footnote in self._document.footnotes:
            for _, component in _iter_component(footnote, root_path='/footnotes'):
                if isinstance(component, component_type):
                    yield component


class DocumentSetter:
    """Manipulates the document by setting or updating components in a parsed document.

    See :class:`~paper_enrichment_agent.common.models.document.DocumentComponent` for the `path`
    definition.
    """

    def __init__(self, document: doc_models.Document) -> None:
        self._document = document

    def remove_component_with_path(self, target_path: str) -> None:
        """Removes the component at the specified path in the document.

        Raises:
            ValueError: If the component is not found in the document.
        """

        parent_path, component_id = target_path.rsplit('/', 1)
        parent_component = DocumentGetter(self._document).get_component_by_path(parent_path)

        if isinstance(parent_component, doc_models.Section):
            parent_component.components = [
                c for c in parent_component.components if c.component_id != component_id
            ]

        elif isinstance(parent_component, doc_models.List):
            parent_component.items = [
                c for c in parent_component.items if c.component_id != component_id
            ]

        elif isinstance(parent_component, doc_models.Paragraph):
            parent_component.elements = [
                e
                for e in parent_component.elements
                if not (
                    isinstance(e, doc_models.DocumentComponent) and e.component_id == component_id
                )
            ]

        elif isinstance(parent_component, doc_models.Figure):
            parent_component.subfigures = [
                s for s in parent_component.subfigures if s.component_id != component_id
            ]


def _iter_component(
    component: doc_models.DocumentComponent, root_path: str
) -> Iterator[tuple[str, doc_models.DocumentComponent]]:
    """Recursively iterates over all components in a document component.

    For each component, yields a tuple containing the path to the component and the component. The
    document tree is traversed pre-order (Node, Left, Right).
    """

    if isinstance(component, doc_models.Section):
        section = component
        section_path = f'{root_path}/{section.component_id}'

        yield section_path, section

        for component in section.components:
            yield from _iter_component(component, section_path)

    elif isinstance(component, doc_models.List):
        list_component = component
        yield f'{root_path}/{list_component.component_id}', list_component

        for item in list_component.items:
            yield from _iter_component(item, f'{root_path}/{list_component.component_id}')

    elif isinstance(component, doc_models.Paragraph):
        paragraph = component
        yield f'{root_path}/{paragraph.component_id}', paragraph

        for element in paragraph.elements:
            if isinstance(element, doc_models.DocumentComponent):
                yield f'{root_path}/{paragraph.component_id}/{element.component_id}', element

    elif isinstance(component, doc_models.Figure):
        figure = component
        yield f'{root_path}/{figure.component_id}', figure

        for subfigure in figure.subfigures:
            yield f'{root_path}/{figure.component_id}/{subfigure.component_id}', subfigure

    else:
        yield f'{root_path}/{component.component_id}', component
