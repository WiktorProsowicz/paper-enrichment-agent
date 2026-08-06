"""Contains the core logic of the `main_survey_enricher` service.

The `main_survey_enricher` service is responsible for the primary communication with the user-facing
component such as TUI or web interface. It orchestrates the operations that are necessary to
fulfill the designed use cases of the entire application.
"""

from dataclasses import dataclass

from prometheus_client import Counter


@dataclass
class Metrics:
    """Metrics for the `main_survey_enricher` service."""

    papers_added: Counter = Counter(
        'main_survey_enricher_papers_added',
        'Number of papers added to the system.',
    )


class MainSurveyEnricherService:
    """The main service of the `main_survey_enricher` component."""

    def __init__(self) -> None:
        pass

    def register_arxiv_survey(self, arxiv_id: str, survey_name: str) -> None:
        """Registers a survey for the given arXiv ID.

        Args:
            arxiv_id (str): The arXiv ID of the paper to register a survey for.
            survey_name (str): The name of the survey to register.
        """

    def delete_survey(self, survey_name: str) -> None:
        """Deletes a survey for the given arXiv ID.

        Args:
            survey_name (str): The name of the survey to delete.
        """

    def 
