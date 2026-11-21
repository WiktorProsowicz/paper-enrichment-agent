"""Contains the models of the paper access evaluation dataset records."""

from typing import Self

import pydantic

from paper_enrichment_agent.common.models import misc as misc_models


class PaperAccessDSInputs(pydantic.BaseModel):
    """Used by the paper access agent to resolve the public source of a paper.

    See :class:`~paper_enrichment_agent.common.models.service_conn.PaperAccessResolutionRequest`
    for more details.
    """

    citation_signature: str


class PaperAccessDSExpectations(pydantic.BaseModel):
    """The expected public source of the cited paper.

    If the agent is expected not to find the paper (answer `null`), `paper_found` is false and the
    type and the source are omitted. MLflow turns each expectation into an assessment, which cannot
    hold a None value, so the omitted fields must not be stored as `null`.
    """

    paper_found: bool
    type: misc_models.PaperAccessType | None = None
    source: str | None = None

    @pydantic.model_validator(mode='after')
    def _check_source_matches_paper_found(self) -> Self:
        """Checks that the type and the source are set if and only if the paper is found."""

        if (self.type is not None) != self.paper_found or (self.source is not None) != (
            self.paper_found
        ):
            raise ValueError(
                'The expected type and source must be set if and only if `paper_found` is true.'
            )

        return self
