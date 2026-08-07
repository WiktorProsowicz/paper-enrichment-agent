"""Contains"""

from prometheus_client import Counter, Gauge, Summary


class Metrics:
    """Metrics for the `main_survey_enricher` service."""

    def __init__(self) -> None:

        self.papers_added = Counter(
            'papers_added',
            'Number of papers added to the system.',
            labelnames=['source'],
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

        self.footnotes_enriched = Gauge(
            'footnotes_enriched',
            'Number of footnotes enriched by the system.',
            namespace='main_survey_enricher',
        )

        self.paper_parsing_time = Summary(
            'paper_parsing_time',
            'Time spent parsing papers by the system.',
            namespace='main_survey_enricher',
        )
