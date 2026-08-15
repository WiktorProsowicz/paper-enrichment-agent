"""Contains the core logic of the `main_survey_enricher` service.

The `main_survey_enricher` service is responsible for the primary communication with the user-facing
component such as TUI or web interface. It orchestrates the operations that are necessary to
fulfill the designed use cases of the entire application.
"""

import time
from functools import cache

from paper_enrichment_agent.common import document_manipulators, logging_setup
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import DocumentMetadata, SurveyMetadata
from paper_enrichment_agent.main_survey_enricher.components.arxiv_parser import ArxivParser
from paper_enrichment_agent.main_survey_enricher.components.doc_db_client import DocDBClient
from paper_enrichment_agent.main_survey_enricher.components.metrics import Metrics


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


class MainSurveyEnricherService:
    """The main service of the `main_survey_enricher` component."""

    class MainSurveyEnricherError(Exception):
        """Base class for exceptions raised by the `main_survey_enricher` service."""

    def __init__(self, metrics: Metrics, db_client: DocDBClient) -> None:

        self._metrics = metrics
        self._db_client = db_client

    def register_arxiv_survey(self, arxiv_id: str, name: str, description: str) -> None:
        """Registers a survey for the given arXiv ID.

        Args:
            arxiv_id: The arXiv ID of the paper to register a survey for.
            name: The name of the survey document to register.
            description: The description of the survey document to register.
        """

        try:
            start = time.perf_counter()
            document = ArxivParser().parse(arxiv_id)
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
            doc_metadata = self._db_client.add_document(
                document=document, name=name, description=description
            )
            self._db_client.register_as_survey(paper_id=doc_metadata.paper_id)
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
            self._metrics.papers_parsed.labels(source='arxiv', status='failed').inc()

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

    def remove_survey(self, survey_id: str) -> None:
        """Removes a survey with the given survey ID.

        Args:
            survey_id: The ID of the survey to remove.
        """

        try:
            start = time.perf_counter()
            survey_metadata = self._db_client.get_survey_info(survey_id)
            self._db_client.delete_survey(survey_id)
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

    def list_surveys(self) -> list[SurveyMetadata]:
        """Lists all registered surveys.

        Returns:
            A list of `SurveyMetadata` objects representing the registered surveys.
        """

        try:
            start = time.perf_counter()
            surveys = self._db_client.get_available_surveys()
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

    def get_survey_info(self, survey_id: str) -> SurveyMetadata:
        """Retrieves information about a specific survey.

        Args:
            survey_id: The ID of the survey to retrieve information for.

        Returns:
            A `SurveyMetadata` object representing the survey.
        """

        try:
            start = time.perf_counter()
            survey_info = self._db_client.get_survey_info(survey_id)
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

    def get_document_details(self, paper_id: str) -> tuple[DocumentMetadata, doc_models.Document]:
        """Retrieves detailed information about a specific document.

        Args:
            paper_id: The ID of the document to retrieve information for.

        Returns:
            A tuple containing a `DocumentMetadata` object and a `Document` object representing the
            document.
        """

        try:
            start = time.perf_counter()
            metadata = self._db_client.get_document_metadata(paper_id)
            with self._db_client.get_document_struct_ref(paper_id) as document:
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

    def remove_document_component(self, paper_id: str, component_path: str) -> None:
        """Removes a specific component from a document.

        Args:
            paper_id: The ID of the document to remove the component from.
            component_path: The path of the component to remove.
        """

        try:
            start = time.perf_counter()
            with self._db_client.get_document_struct_ref(paper_id) as document:
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
