"""Contains the core logic of the `main_survey_enricher` service.

The `main_survey_enricher` service is responsible for the primary communication with the user-facing
component such as TUI or web interface. It orchestrates the operations that are necessary to
fulfill the designed use cases of the entire application.
"""

import time
from functools import cache

from paper_enrichment_agent.common import logging_setup
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

        Raises:
            MainSurveyEnricherError: If the survey could not be registered.
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
            self._metrics.papers_added.labels(source='arxiv').inc()

            raise self.MainSurveyEnricherError(
                f'Failed to register the survey for the arXiv paper with ID {arxiv_id}: {e}'
            ) from e
