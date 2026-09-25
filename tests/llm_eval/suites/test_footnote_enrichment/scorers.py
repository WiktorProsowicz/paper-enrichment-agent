import pathlib
import re

import mlflow
from mlflow.entities import SpanType, Trace
from mlflow.genai import Scorer, scorer
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
from paper_enrichment_agent.common import document_manipulators
from paper_enrichment_agent.footnote_enrichment_agent.components import doc_utils

from mlflow.genai.evaluation.entities import Feedback
from llm_eval import core as harness_core
from llm_eval.suites.test_footnote_enrichment import data as harness_data


AGENT_INVOCATION_SPAN_NAME = 'FootnoteEnrichmentAgent::invoke'
CALL_TOOL_SPAN_PATTERN = re.compile(r'^call_tool\b')
AGENT_REASONING_SPAN_NAME = 'agent_reasoning'
AGENT_ACTION_SPAN_NAME = 'agent_action'


class FigureUsefulnessResponseModel(pydantic.BaseModel):
    """Response model for the FigureUsefulnessRating LLM Judge."""

    rating: int = pydantic.Field(ge=1, le=5)
    rationale: str


def spawn_figure_usefulness_rating(llm: ChatLiteLLM) -> Scorer:
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
                    'citing_paragraph_additional_context',
                    'cited_article_title',
                    'cited_article_abstract',
                    'cited_article_figures',
                    'footnote_content',
                ],
            ),
        ]
    )

    output_parser = PydanticOutputParser(pydantic_object=FigureUsefulnessResponseModel)

    @harness_core.scorer_with_typed_args()
    @mlflow.trace(name='figure_usefulness_rating', span_type=SpanType.EVALUATOR)
    def figure_usefulness_rating(trace: Trace, outputs: doc_models.Section) -> Feedback | None:
        """Assigns the FigureUsefulnessRating to the enriched footnote.

        FigureUsefulnessRating is a discrete integer score on scale (1-5) that indicates how
        adequately the enriched footnote uses figures from the referenced document to support
        the survey's claims.
        """

        enrichment_request = misc_models.FootnoteEnrichmentRequest(
            **trace.search_spans(name=AGENT_INVOCATION_SPAN_NAME)[0].inputs['request']
        )

        citation_reference = next(
            element
            for element in enrichment_request.referencing_paragraph.elements
            if isinstance(element, doc_models.Reference)
            and element.component_id == enrichment_request.reference_id
        )

        citing_paragraph_fragment = doc_utils.get_context_of_citation_in_paragraph(
            enrichment_request.referencing_paragraph,
            citation_reference,
            n_context_chars=500,
        )

        cited_article_figures = _render_all_figures_from_document(
            enrichment_request.reference_document
        )

        if not cited_article_figures.strip():
            return None

        enriched_footnote_repr = _stringify_generated_footnote(outputs)

        model_response = llm.invoke(
            message_template.format_messages(
                base_article_title=enrichment_request.survey_title,
                base_article_abstract=enrichment_request.survey_abstract,
                citing_paragraph_fragment=citing_paragraph_fragment,
                citing_paragraph_additional_context=enrichment_request.citation_context_info,
                cited_article_title=enrichment_request.reference_document.title,
                cited_article_abstract=enrichment_request.reference_document.abstract,
                cited_article_figures=cited_article_figures,
                footnote_content=enriched_footnote_repr,
            )
        )
        parsed_response = output_parser.parse(model_response.content)

        return Feedback(
            value=parsed_response.rating,
            rationale=parsed_response.rationale,
        )

    return figure_usefulness_rating


@scorer
def agent_execution_stats_metrics(trace: Trace) -> list[Feedback]:
    """Computes execution statistics for the agent based on the trace."""

    call_tool_spans = trace.search_spans(name=CALL_TOOL_SPAN_PATTERN)
    reason_spans = trace.search_spans(name=AGENT_REASONING_SPAN_NAME)
    action_spans = trace.search_spans(name=AGENT_ACTION_SPAN_NAME)

    successful_tool_calls = len(
        [span for span in call_tool_spans if not span.attributes['tool_is_error']]
    )
    tool_call_success_rate = (
        (successful_tool_calls / len(call_tool_spans)) if call_tool_spans else 0
    )

    return [
        Feedback(name='call_tool_count', value=len(call_tool_spans)),
        Feedback(name='agent_reasoning_count', value=len(reason_spans)),
        Feedback(name='agent_action_count', value=len(action_spans)),
        Feedback(name='tool_call_success_rate', value=tool_call_success_rate),
    ]


def spawn_specific_tool_usage_metric(
    tool_name: str, metric_name: str, only_successful_calls: bool = False
) -> Scorer:
    """Spawns a metric for tracking the usage of a specific tool."""

    def specific_tool_usage_metric(trace: Trace) -> list[Feedback]:
        """Measures the call count for a specific tool."""
        call_tool_spans = trace.search_spans(name=CALL_TOOL_SPAN_PATTERN)
        specific_tool_calls = len(
            [
                span
                for span in call_tool_spans
                if span.attributes['tool_name'] == tool_name
                and (not only_successful_calls or not span.attributes['tool_is_error'])
            ]
        )

        return [Feedback(name=metric_name, value=specific_tool_calls)]

    specific_tool_usage_metric.__name__ = metric_name

    return scorer(specific_tool_usage_metric)


_PROMPT_TEMPLATES_DIR = pathlib.Path(__file__).parent / 'llm_judge_prompts'


def _render_all_figures_from_document(document: doc_models.Document) -> str:
    """Renders the figures from the document into a string representation."""

    doc_getter = document_manipulators.DocumentGetter(document)

    rendered_figures: list[str] = []

    for figure in doc_getter.iter_components_of_type(doc_models.Figure):
        if isinstance(figure, doc_models.Figure):
            rendered_figures.append(
                f'Figure: {figure.caption}\n'
                + 'Subfigures:\n'
                + '\n'.join(
                    f'\t{subfigure.__class__.__name__}: {subfigure.caption}'
                    for subfigure in figure.subfigures
                )
            )

    return '\n\n'.join(rendered_figures)


def _stringify_generated_footnote(footnote: doc_models.Section) -> str:
    """Converts the generated footnote into a string representation."""

    str_components: list[str] = []

    for component in footnote.components:
        if isinstance(component, doc_models.Paragraph):
            str_components.append(
                ''.join(e.str_content for e in doc_utils.stringify_paragraph_elements(component))
            )

        elif isinstance(component, doc_models.Figure):
            str_components.append(
                f'Figure: {component.caption}\n'
                + 'Subfigures:\n'
                + '\n'.join(
                    f'\t{subfigure.__class__.__name__}: {subfigure.caption}'
                    for subfigure in component.subfigures
                    if isinstance(subfigure, doc_models.ImgSubfigure)
                )
            )

    return '\n\n'.join(str_components)
