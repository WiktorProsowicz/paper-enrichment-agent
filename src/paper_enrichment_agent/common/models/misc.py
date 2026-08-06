"""Contains miscellaneous data models that are shared by the app's services."""

from typing import Annotated

import pydantic
from pydantic import Field


class SurveyMetadata(pydantic.BaseModel):
    """Represents the metadata of a survey added to the system."""

    survey_id: Annotated[str, Field(description='The unique identifier of the survey.')]
    survey_name: Annotated[str, Field(description='The name of the survey.')]
