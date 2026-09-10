import pathlib

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_litellm import ChatLiteLLM
import pydantic


from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models import misc as misc_models

from mlflow.genai.evaluation.entities import Feedback
from llm_eval import core as harness_core
from llm_eval.suites.test_footnote_enrichment import data as harness_data


_PROMPT_TEMPLATES_DIR = pathlib.Path(__file__).parent / 'llm_judge_prompts'


class RatingWithRationaleResponse(pydantic.BaseModel):
    """Response model for LLM Judges that provide an integer rating with a rationale."""

    rating: int
    rationale: str


def spawn_figure_usefulness_rating(llm: ChatLiteLLM):
    """Dynamically creates the FigureUsefulnessRating LLM Judge."""

    with open(_PROMPT_TEMPLATES_DIR.joinpath('figure_usefulness_rating_system_prompt.md')) as f:
        system_prompt = f.read()

    message_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template_file(
                _PROMPT_TEMPLATES_DIR.joinpath('figure_usefulness_rating_user_prompt.md'),
                input_variables=[
                    'base_article_title',
                    'base_article_abstract',
                    'citing_paragraph_fragment',
                    'citing_paragrapg_additional_context',
                    'cited_article_title',
                    'cited_article_abstract',
                    'cited_article_figures',
                    'enriched_footnote',
                ],
            ),
        ]
    )

    output_parser = PydanticOutputParser(pydantic_object=RatingWithRationaleResponse)

    @harness_core.scorer_with_typed_args()
    def figure_usefulness_rating(
        inputs: misc_models.FootnoteEnrichmentRequest, outputs: doc_models.Section
    ) -> Feedback | None:
        """Assigns the FigureUsefulnessRating to the enriched footnote.

        FigureUsefulnessRating is a discrete integer score on scale (1-5) that indicates how
        adequately the enriched footnote uses figures from the referenced document to support
        the survey's claims.
        """

        if not any(isinstance(x, doc_models.Figure) for x in outputs.components):
            return None

        model_response = llm.invoke(messages=message_template.format_messages())
        parsed_response = output_parser.parse(model_response.content)

        return Feedback(
            value=parsed_response.rating,
            rationale=parsed_response.rationale,
        )

    return figure_usefulness_rating
