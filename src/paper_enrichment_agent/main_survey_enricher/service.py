"""Contains the core logic of the `main_survey_enricher` service.

The `main_survey_enricher` service is responsible for the primary communication with the user-facing
component such as TUI or web interface. It orchestrates the operations that are necessary to
fulfill the designed use cases of the entire application.
"""

import asyncio
import pathlib
import tempfile
import time
from functools import cache

from prometheus_client import Counter, Gauge, Summary

from paper_enrichment_agent.common import document_manipulators, logging_setup
from paper_enrichment_agent.common.http_client import HTTPClient
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import (
    DocumentMetadata,
    RenderedDocumentMetadata,
    SerializedDocFormat,
    SurveyMetadata,
)
from paper_enrichment_agent.common.models.service_conn import FootnoteEnrichmentRequest
from paper_enrichment_agent.main_survey_enricher.components.arxiv_parser import ArxivParser
from paper_enrichment_agent.main_survey_enricher.components.db_clients import (
    rendered_doc_db_client,
)
from paper_enrichment_agent.main_survey_enricher.components.db_clients.doc_db_client import (
    DocDBClient,
)
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.core import (
    SerializationEnvironment,
)
from paper_enrichment_agent.main_survey_enricher.components.doc_serializers.markdown import (
    MarkdownSerializer,
)


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


class MainSurveyEnricherService:
    """The main service of the `main_survey_enricher` component."""

    class MainSurveyEnricherError(Exception):
        """Base class for exceptions raised by the `main_survey_enricher` service."""

    # Formats `render_document` has a serializer for.
    _SUPPORTED_RENDERING_FORMATS = (SerializedDocFormat.MARKDOWN,)

    def __init__(
        self,
        metrics: 'Metrics',
        doc_db_client: DocDBClient,
        rendered_doc_db_client: rendered_doc_db_client.RenderedDocDBClient,
        footnote_agent_client: HTTPClient,
    ) -> None:

        self._metrics = metrics
        self._doc_db_client = doc_db_client
        self._rendered_doc_db_client = rendered_doc_db_client
        self._footnote_agent_client = footnote_agent_client

    async def register_arxiv_survey(self, arxiv_id: str, name: str, description: str) -> None:
        """Registers a survey for the given arXiv ID.

        The paper is fetched and parsed in a separate thread, since the parser is synchronous and
        its HTML parsing is CPU-bound.

        Args:
            arxiv_id: The arXiv ID of the paper to register a survey for.
            name: The name of the survey document to register.
            description: The description of the survey document to register.
        """

        try:
            start = time.perf_counter()
            document = await asyncio.to_thread(ArxivParser().parse, arxiv_id)
            end = time.perf_counter()

            _logger().info(
                'Successfully parsed the arXiv paper.',
                arxiv_id=arxiv_id,
                name=name,
                description=description,
            )
            self._metrics.papers_parsed.labels(source='arxiv', status='success').inc()
            self._metrics.paper_parsing_time.observe(end - start)

            start = time.perf_counter()
            doc_metadata = await self._doc_db_client.add_document(
                document=document, name=name, description=description
            )
            await self._doc_db_client.register_as_survey(paper_id=doc_metadata.paper_id)
            end = time.perf_counter()

            _logger().info(
                'Successfully registered the survey for the arXiv paper.',
                arxiv_id=arxiv_id,
                name=name,
                description=description,
            )
            self._metrics.papers_added.labels(source='arxiv').inc()
            self._metrics.doc_db_operations_time.observe(end - start)

        except ArxivParser.ParsingError as e:
            _logger().error(
                'Failed to parse the arXiv paper.',
                arxiv_id=arxiv_id,
                name=name,
                description=description,
                error=str(e),
            )
            self._metrics.papers_parsed.labels(source='arxiv', status='failure').inc()

            raise self.MainSurveyEnricherError(
                f'Failed to parse the arXiv paper with ID {arxiv_id}: {e}'
            ) from e

        except DocDBClient.DocDBClientError as e:
            _logger().error(
                'Failed to register the survey for the arXiv paper.',
                arxiv_id=arxiv_id,
                name=name,
                description=description,
                error=str(e),
            )

            raise self.MainSurveyEnricherError(
                f'Failed to register the survey for the arXiv paper with ID {arxiv_id}: {e}'
            ) from e

    async def remove_survey(self, survey_id: str) -> None:
        """Removes a survey with the given survey ID.

        Args:
            survey_id: The ID of the survey to remove.
        """

        try:
            start = time.perf_counter()
            survey_metadata = await self._doc_db_client.get_survey_info(survey_id)
            await self._doc_db_client.delete_survey(survey_id)
            end = time.perf_counter()

            _logger().info(
                'Successfully removed the survey.',
                survey_id=survey_id,
                removed_references=len(survey_metadata.referenced_docs),
            )
            self._metrics.papers_removed.inc(len(survey_metadata.referenced_docs) + 1)
            self._metrics.doc_db_operations_time.observe(end - start)

        except DocDBClient.DocDBClientError as e:
            _logger().error(
                'Failed to remove the survey and its associated document.',
                survey_id=survey_id,
                error=str(e),
            )

            raise self.MainSurveyEnricherError(
                f'Failed to remove the survey with ID {survey_id}: {e}'
            ) from e

    async def list_surveys(self) -> list[SurveyMetadata]:
        """Lists all registered surveys.

        Returns:
            A list of `SurveyMetadata` objects representing the registered surveys.
        """

        try:
            start = time.perf_counter()
            surveys = await self._doc_db_client.get_available_surveys()
            end = time.perf_counter()

            _logger().info(
                'Successfully listed the registered surveys.',
                num_surveys=len(surveys),
            )
            self._metrics.doc_db_operations_time.observe(end - start)

            return surveys

        except DocDBClient.DocDBClientError as e:
            _logger().error(
                'Failed to list the registered surveys.',
                error=str(e),
            )

            raise self.MainSurveyEnricherError(f'Failed to list the registered surveys: {e}') from e

    async def get_survey_info(self, survey_id: str) -> SurveyMetadata:
        """Retrieves information about a specific survey.

        Args:
            survey_id: The ID of the survey to retrieve information for.

        Returns:
            A `SurveyMetadata` object representing the survey.
        """

        try:
            start = time.perf_counter()
            survey_info = await self._doc_db_client.get_survey_info(survey_id)
            end = time.perf_counter()

            _logger().info(
                'Successfully retrieved information about the survey.',
                survey_id=survey_id,
                num_references=len(survey_info.referenced_docs),
            )
            self._metrics.doc_db_operations_time.observe(end - start)

            return survey_info

        except DocDBClient.DocDBClientError as e:
            _logger().error(
                'Failed to retrieve information about the survey.',
                survey_id=survey_id,
                error=str(e),
            )

            raise self.MainSurveyEnricherError(
                f'Failed to retrieve information about the survey with ID {survey_id}: {e}'
            ) from e

    async def get_document_details(
        self, paper_id: str
    ) -> tuple[DocumentMetadata, doc_models.Document]:
        """Retrieves detailed information about a specific document.

        Args:
            paper_id: The ID of the document to retrieve information for.

        Returns:
            A tuple containing a `DocumentMetadata` object and a `Document` object representing the
            document.
        """

        try:
            start = time.perf_counter()
            metadata = await self._doc_db_client.get_document_metadata(paper_id)
            async with self._doc_db_client.get_document_struct_ref(paper_id) as document:
                document = document.model_copy(deep=True)
            end = time.perf_counter()

            _logger().info(
                'Successfully retrieved detailed information about the document.',
                paper_id=paper_id,
            )
            self._metrics.doc_db_operations_time.observe(end - start)

            return metadata, document

        except DocDBClient.DocDBClientError as e:
            _logger().error(
                'Failed to retrieve detailed information about the document.',
                paper_id=paper_id,
                error=str(e),
            )

            raise self.MainSurveyEnricherError(
                f'Failed to retrieve detailed information about the document with ID {paper_id}: {e}'
            ) from e

    async def remove_document_component(self, paper_id: str, component_path: str) -> None:
        """Removes a specific component from a document.

        Args:
            paper_id: The ID of the document to remove the component from.
            component_path: The path of the component to remove.
        """

        try:
            start = time.perf_counter()
            async with self._doc_db_client.get_document_struct_ref(paper_id) as document:
                doc_setter = document_manipulators.DocumentSetter(document)
                doc_setter.remove_component_with_path(component_path)
            end = time.perf_counter()

            _logger().info(
                'Successfully removed the component from the document.',
                paper_id=paper_id,
                component_path=component_path,
            )
            self._metrics.doc_db_operations_time.observe(end - start)

        except (DocDBClient.DocDBClientError, ValueError) as e:
            _logger().error(
                'Failed to remove the component from the document.',
                paper_id=paper_id,
                component_path=component_path,
                error=str(e),
            )

            raise self.MainSurveyEnricherError(
                f'Failed to remove component "{component_path}" from the document {paper_id}: {e}'
            ) from e

    async def render_document(
        self, paper_id: str, doc_format: SerializedDocFormat
    ) -> RenderedDocumentMetadata:
        """Serializes a document into the given format and stores the result in the database.

        The document and its images are fetched into a temporary directory, where the serializer
        creates the archive. The serialization runs in a separate thread, since it is bound to the
        CPU and the file system. The previous rendering in the same format, if any, is replaced.

        Args:
            paper_id: The ID of the document to render.
            doc_format: The format to render the document into.

        Returns:
            The metadata of the stored rendered document.
        """

        if doc_format not in self._SUPPORTED_RENDERING_FORMATS:
            _logger().error(
                'Unsupported rendering format.', paper_id=paper_id, doc_format=doc_format
            )

            raise self.MainSurveyEnricherError(
                f'Failed to render the document {paper_id}: unsupported format {doc_format}.'
            )

        try:
            start = time.perf_counter()
            metadata = await self._doc_db_client.get_document_metadata(paper_id)
            async with self._doc_db_client.get_document_struct_ref(paper_id) as document:
                document = document.model_copy(deep=True)

            with tempfile.TemporaryDirectory() as work_dir:
                environment = await asyncio.to_thread(
                    SerializationEnvironment.create, document, pathlib.Path(work_dir)
                )
                await self._place_images(environment=environment, metadata=metadata)

                if doc_format == SerializedDocFormat.MARKDOWN:
                    serializer = MarkdownSerializer(environment)

                archive_path = await asyncio.to_thread(serializer.serialize)

                rendered_metadata = await self._rendered_doc_db_client.add_rendered_document(
                    paper_id=paper_id, doc_format=doc_format, archive_path=archive_path
                )
            end = time.perf_counter()

            _logger().info(
                'Successfully rendered the document.',
                paper_id=paper_id,
                doc_format=doc_format,
                size_bytes=rendered_metadata.size_bytes,
            )
            self._metrics.documents_rendered.labels(format=doc_format, status='success').inc()
            self._metrics.document_rendering_time.observe(end - start)

            return rendered_metadata

        except (
            DocDBClient.DocDBClientError,
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            SerializationEnvironment.SerializationEnvironmentError,
            MarkdownSerializer.SerializationError,
            ValueError,
        ) as e:
            _logger().error(
                'Failed to render the document.',
                paper_id=paper_id,
                doc_format=doc_format,
                error=str(e),
            )
            self._metrics.documents_rendered.labels(format=doc_format, status='failure').inc()

            raise self.MainSurveyEnricherError(
                f'Failed to render the document {paper_id} into {doc_format}: {e}'
            ) from e

    async def list_rendered_documents(
        self, paper_id: str | None = None
    ) -> list[RenderedDocumentMetadata]:
        """Lists the rendered documents.

        Args:
            paper_id: If given, only the renderings of this document are listed.

        Returns:
            The metadata of the matching rendered documents.
        """

        try:
            start = time.perf_counter()
            rendered_docs = await self._rendered_doc_db_client.list_rendered_documents(paper_id)
            end = time.perf_counter()

            _logger().info(
                'Successfully listed the rendered documents.',
                paper_id=paper_id,
                num_rendered_docs=len(rendered_docs),
            )
            self._metrics.doc_db_operations_time.observe(end - start)

            return rendered_docs

        except rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError as e:
            _logger().error(
                'Failed to list the rendered documents.', paper_id=paper_id, error=str(e)
            )

            raise self.MainSurveyEnricherError(f'Failed to list the rendered documents: {e}') from e

    async def download_rendered_document(
        self, paper_id: str, doc_format: SerializedDocFormat, local_path: pathlib.Path
    ) -> None:
        """Downloads the archive of a rendered document to a local file.

        The file is written on the host running the service. The document must have been rendered
        into the given format beforehand, see `render_document`.

        Args:
            paper_id: The ID of the rendered document.
            doc_format: The format of the rendering.
            local_path: The path of the local file to be written. Its parent directory must exist.
        """

        try:
            start = time.perf_counter()
            if not await self._rendered_doc_db_client.rendered_document_exists(
                paper_id, doc_format
            ):
                raise ValueError(f'The document has not been rendered into {doc_format} yet.')

            await self._rendered_doc_db_client.download_rendered_document(
                paper_id=paper_id, doc_format=doc_format, local_path=local_path
            )
            end = time.perf_counter()

            _logger().info(
                'Successfully downloaded the rendered document.',
                paper_id=paper_id,
                doc_format=doc_format,
                local_path=str(local_path),
            )
            self._metrics.doc_db_operations_time.observe(end - start)

        except (
            rendered_doc_db_client.RenderedDocDBClient.RenderedDocDBClientError,
            ValueError,
        ) as e:
            _logger().error(
                'Failed to download the rendered document.',
                paper_id=paper_id,
                doc_format=doc_format,
                error=str(e),
            )

            raise self.MainSurveyEnricherError(
                f'Failed to download the {doc_format} rendering of document {paper_id}: {e}'
            ) from e

    async def enrich_footnote(self, survey_id: str, citation_path: str) -> doc_models.Section:
        """Enriches a citation of a survey into a footnote composed by the footnote agent.

        The document cited by the survey has to be assigned to the citation's bibliography entry
        beforehand (see `SurveyMetadata.referenced_docs`). The request to the
        `footnote_enrichment_agent` service is built from the survey, the cited document and the
        titles of the sections containing the citation. The composed footnote is appended to the
        survey's footnotes, and the citation is turned into an `element` reference pointing to it.

        The survey is read again after the agent responds, so that changes made to it during the
        enrichment are preserved. Hence, a citation can be enriched only once.

        Args:
            survey_id: The ID of the survey containing the citation.
            citation_path: The path of the citation reference in the survey.

        Returns:
            The footnote composed for the citation.
        """

        try:
            start = time.perf_counter()
            survey_info = await self._doc_db_client.get_survey_info(survey_id)
            async with self._doc_db_client.get_document_struct_ref(survey_id) as survey:
                survey = survey.model_copy(deep=True)

            paragraph, citation = self._resolve_citation(survey, citation_path)

            if (referenced_paper_id := survey_info.referenced_docs.get(citation.target)) is None:
                raise ValueError(
                    f'No document is assigned to the reference {citation.target} of survey '
                    f'{survey_id}.'
                )

            async with self._doc_db_client.get_document_struct_ref(
                referenced_paper_id
            ) as referenced_doc:
                referenced_doc = referenced_doc.model_copy(deep=True)

            request = FootnoteEnrichmentRequest(
                survey_title=survey.title,
                survey_abstract=survey.abstract,
                reference_document=referenced_doc,
                reference_id=citation.component_id,
                referencing_paragraph=paragraph,
                citation_context_info=self._build_citation_context_info(survey, citation_path),
            )
            footnote = await self._footnote_agent_client.apost(
                '/reference_to_footnote', request, response_schema=doc_models.Section
            )

            async with self._doc_db_client.get_document_struct_ref(survey_id) as survey:
                _, citation = self._resolve_citation(survey, citation_path)

                survey.footnotes.append(footnote)

                citation.description = (
                    f'Former citation {citation.target}. Now points to enriched footnote.'
                )
                citation.ref_type = 'element'
                citation.target = f'/footnotes/{footnote.component_id}'
            end = time.perf_counter()

            _logger().info(
                'Successfully enriched the citation into a footnote.',
                survey_id=survey_id,
                citation_path=citation_path,
                referenced_paper_id=referenced_paper_id,
                footnote_id=footnote.component_id,
            )
            self._metrics.footnotes_enriched.labels(status='success').inc()
            self._metrics.footnote_enrichment_time.observe(end - start)

            return footnote

        except (DocDBClient.DocDBClientError, HTTPClient.HTTPClientError, ValueError) as e:
            _logger().error(
                'Failed to enrich the citation into a footnote.',
                survey_id=survey_id,
                citation_path=citation_path,
                error=str(e),
            )
            self._metrics.footnotes_enriched.labels(status='failure').inc()

            raise self.MainSurveyEnricherError(
                f'Failed to enrich the citation {citation_path} of survey {survey_id}: {e}'
            ) from e

    def _resolve_citation(
        self, document: doc_models.Document, citation_path: str
    ) -> tuple[doc_models.Paragraph, doc_models.Reference]:
        """Returns the citation reference at the given path together with its paragraph.

        Raises:
            ValueError: If the path does not point to a citation placed in a paragraph.
        """

        doc_getter = document_manipulators.DocumentGetter(document)

        citation = doc_getter.get_component_by_path(citation_path)

        if not (isinstance(citation, doc_models.Reference) and citation.ref_type == 'citation'):
            raise ValueError(f'Component at path {citation_path} is not a citation reference.')

        paragraph_path = citation_path.rsplit('/', 1)[0]
        paragraph = doc_getter.get_component_by_path(paragraph_path)

        if not isinstance(paragraph, doc_models.Paragraph):
            raise ValueError(f'Component at path {paragraph_path} is not a paragraph.')

        return paragraph, citation

    def _build_citation_context_info(self, survey: doc_models.Document, citation_path: str) -> str:
        """Describes the location of the citation in the survey for the footnote agent."""

        sections_path = ' > '.join(
            f"'{component.title}'"
            for component in document_manipulators.DocumentGetter(survey).get_components_on_path(
                citation_path
            )
            if isinstance(component, doc_models.Section)
        )

        return (
            'The document was cited in the section specified by the following path in the '
            f'document tree: {sections_path}'
        )

    async def _place_images(
        self, environment: SerializationEnvironment, metadata: DocumentMetadata
    ) -> None:
        """Downloads the document's stored images to the paths given by the environment.

        Raises:
            ValueError: If an image subfigure of the document has no stored image.
        """

        for component_path in environment.image_component_paths:
            if component_path not in metadata.images:
                raise ValueError(
                    f'The image of component {component_path} is missing in the metadata of '
                    f'document {metadata.paper_id}.'
                )

            await self._doc_db_client.download_image(
                image_db_path=metadata.images[component_path],
                local_path=environment.image_path_for_component(component_path),
            )


class Metrics:
    """Metrics for the `main_survey_enricher` service.

    The metrics are registered in the default Prometheus registry, so a single `Metrics` instance
    should be created per process. They are exposed by the service's `/metrics` endpoint.
    """

    def __init__(self) -> None:

        self.papers_added = Counter(
            'papers_added',
            'Number of papers added to the system.',
            labelnames=['source'],
            namespace='main_survey_enricher',
        )

        self.papers_parsed = Counter(
            'papers_parsed',
            'Number of papers parsed by the system.',
            labelnames=['source', 'status'],  # status can be 'success' or 'failure'
            namespace='main_survey_enricher',
        )

        self.papers_removed = Counter(
            'papers_removed',
            'Number of papers removed from the system.',
            namespace='main_survey_enricher',
        )

        self.characters_translated = Gauge(
            'characters_translated',
            'Number of characters translated by the system.',
            namespace='main_survey_enricher',
        )

        self.footnotes_enriched = Counter(
            'footnotes_enriched',
            'Number of survey citations enriched into footnotes by the system.',
            labelnames=['status'],  # status can be 'success' or 'failure'
            namespace='main_survey_enricher',
        )

        self.footnote_enrichment_time = Summary(
            'footnote_enrichment_time',
            'Time spent on successful footnote enrichments, including the agent call.',
            namespace='main_survey_enricher',
        )

        self.paper_parsing_time = Summary(
            'paper_parsing_time',
            'Time spent on successful parsing of papers by the system.',
            namespace='main_survey_enricher',
        )

        self.doc_db_operations_time = Summary(
            'doc_db_operations_time',
            'Time spent on operations with the document database by the system.',
            namespace='main_survey_enricher',
        )

        self.documents_rendered = Counter(
            'documents_rendered',
            'Number of documents rendered (serialized) by the system.',
            labelnames=['format', 'status'],  # status can be 'success' or 'failure'
            namespace='main_survey_enricher',
        )

        self.document_rendering_time = Summary(
            'document_rendering_time',
            'Time spent on successful rendering of documents, including the database transfers.',
            namespace='main_survey_enricher',
        )

        self.papers_added.labels(source='arxiv')

        for doc_format in SerializedDocFormat:
            for status in ('success', 'failure'):
                self.documents_rendered.labels(format=doc_format, status=status)

        for status in ('success', 'failure'):
            self.papers_parsed.labels(source='arxiv', status=status)
            self.footnotes_enriched.labels(status=status)
