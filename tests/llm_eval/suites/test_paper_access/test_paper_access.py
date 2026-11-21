"""Evaluates the paper access agent of the paper scraping service."""

import mlflow
import pytest

from llm_eval import core as harness_core
from llm_eval import scorers as harness_common_scorers
from llm_eval.suites.test_paper_access import data as harness_data
from llm_eval.suites.test_paper_access import scorers as harness_scorers
from paper_enrichment_agent.common.http_client import HTTPClient
from paper_enrichment_agent.common.models import service_conn as conn_models

# The tools available to the paper access agent
_AGENT_TOOLS = (
    'scrape_metadata_from_openalex',
    'has_arxiv_html_version',
    'is_arxiv_pdf_location_correct',
    'query_unpaywall',
)


@mlflow.test
def test_paper_access_resolution(suite_run: harness_core.EvaluationRun) -> None:
    """Resolves the public sources of papers cited with differently formatted signatures.

    The access information resolved by the agent is compared with the expected one:
    1. Whether the kind of the source (`arxiv` HTML, `pdf`, or none) is correct.
    2. Whether the source itself (the arXiv ID or the PDF URL) is correct.

    Additionally, the agent's execution is measured with the number of its tool calls, reasoning
    and action steps, and the call count and success rate of each tool.
    """

    run_config = suite_run.suite_config

    @harness_core.predict_with_typed_inputs
    @mlflow.trace(name='root_eval_span')
    async def predict(
        inputs: harness_data.PaperAccessDSInputs,
    ) -> conn_models.PaperAccessResolutionResponse:

        # The execution time is bounded by the per-test-case timeout (`test_case_exec_timeout`).
        http_client = HTTPClient(
            base_url=run_config['paper_scraping_service_url'],
            timeout=999999,
        )

        return await http_client.apost(
            '/resolve_paper_access',
            conn_models.PaperAccessResolutionRequest(citation_signature=inputs.citation_signature),
            response_schema=conn_models.PaperAccessResolutionResponse,
        )

    mlflow.log_params({'dataset_name': run_config['datasets']['paper_access']})

    eval_dataset = harness_core.load_dataset(
        run_config['datasets']['paper_access'],
        inputs_type=harness_data.PaperAccessDSInputs,
        expectations_type=harness_data.PaperAccessDSExpectations,
    )

    eval_results = mlflow.genai.evaluate(
        eval_dataset,
        [
            harness_scorers.access_type_correctness,
            harness_scorers.access_source_correctness,
            harness_common_scorers.agent_step_count_metrics,
            *(harness_common_scorers.spawn_specific_tool_usage_metrics(t) for t in _AGENT_TOOLS),
        ],
        predict,
    )

    fail_reason = harness_core.validate_results(
        suite_run.mlflow_run.info.run_id,
        eval_results,
        run_config['thresholds']['paper_access'],
    )

    if fail_reason:
        pytest.fail(fail_reason)
