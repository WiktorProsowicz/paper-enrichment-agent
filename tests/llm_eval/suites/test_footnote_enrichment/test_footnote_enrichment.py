from typing import Any

import mlflow
from langchain_litellm import ChatLiteLLM

from llm_eval.suites.test_footnote_enrichment import data as harness_data
from llm_eval.suites.test_footnote_enrichment import scorers as harness_scorers
from llm_eval import core as harness_core

from paper_enrichment_agent.common.models import misc as misc_models
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common import document_manipulators
from paper_enrichment_agent.main_survey_enricher.components import arxiv_parser
from paper_enrichment_agent.common.http_client import HTTPClient


@mlflow.test
def test_judge_calibration(eval_run: harness_core.EvaluationRun):
    """Calculates the calibration scores for the LLM Judges configured for the test suite."""

    async def predict(ds_inputs: dict[str, Any]) -> doc_models.Section:

        inputs = harness_data.FootnoteEnrichmentDSInputs.model_validate(ds_inputs)

        survey_doc = arxiv_parser.ArxivParser().parse(inputs.survey_arxiv_id)
        cited_doc = arxiv_parser.ArxivParser().parse(inputs.cited_document_arxiv_id)

        doc_getter = document_manipulators.DocumentGetter(survey_doc)

        referencing_paragraph = doc_getter.get_component_by_path(inputs.citing_paragraph_path)

        assert isinstance(referencing_paragraph, doc_models.Paragraph)

        request = misc_models.FootnoteEnrichmentRequest(
            survey_title=inputs.survey_title,
            survey_abstract=survey_doc.abstract,
            reference_document=cited_doc,
            reference_document_title=inputs.cited_document_title,
            reference_id=inputs.reference_id,
            referencing_paragraph=referencing_paragraph,
            citation_context_info=inputs.citation_context,
        )

        http_client = HTTPClient(base_url=eval_run.suite_config['footnote_enrichment_service_url'])

        return await http_client.apost(
            '/reference_to_footnote', request, response_schema=doc_models.Section
        )

    eval_dataset = harness_core.load_dataset(
        eval_run.suite_config['calibration_dataset'], harness_data.ScorerCalibrationEvalSample
    )

    judge_llm = ChatLiteLLM(
        name=eval_run.suite_config['judge_llm_config']['model_name'],
        api_key=eval_run.suite_config['judge_llm_config']['api_key'],
        **eval_run.suite_config['judge_llm_config'].get('model_params', {}),
    )

    eval_results = mlflow.genai.evaluate(
        eval_dataset, [harness_scorers.spawn_figure_usefulness_rating(judge_llm)], predict
    )
