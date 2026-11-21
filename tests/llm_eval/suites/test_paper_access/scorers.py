"""Contains the scorers of the paper access agent's evaluation suite.

The scorers are deterministic: they compare the access information resolved by the agent with the
expected one. The agent-agnostic execution metrics live in :mod:`llm_eval.scorers`.
"""

import re
import urllib.parse
from typing import Any

from mlflow.entities import Trace, TraceState
from mlflow.entities.assessment import Feedback

from llm_eval import core as harness_core
from llm_eval.suites.test_paper_access import data as harness_data
from paper_enrichment_agent.common.models import service_conn as conn_models


@harness_core.scorer_with_typed_args()
def access_type_correctness(
    outputs: Any, expectations: harness_data.PaperAccessDSExpectations, trace: Trace
) -> Feedback:
    """Checks whether the agent resolved the expected kind of source (or no source at all)."""

    if trace.info.state == TraceState.ERROR or outputs is None:
        return Feedback(name='access_type_correct', value=0, rationale=_FAILED_RATIONALE)

    access_info = conn_models.PaperAccessResolutionResponse.model_validate(outputs).access_info
    resolved_type = access_info.type if access_info else None

    return Feedback(
        name='access_type_correct',
        value=int(resolved_type == expectations.type),
        rationale=f'Expected type: {expectations.type}, resolved type: {resolved_type}.',
    )


@harness_core.scorer_with_typed_args()
def access_source_correctness(
    outputs: Any, expectations: harness_data.PaperAccessDSExpectations, trace: Trace
) -> Feedback:
    """Checks whether the agent resolved the expected source (arXiv ID or PDF URL).

    The sources are compared after normalization, so that e.g. the arXiv ID version suffixes and
    the `.pdf` extensions of the URLs do not matter.
    """

    if trace.info.state == TraceState.ERROR or outputs is None:
        return Feedback(name='access_source_correct', value=0, rationale=_FAILED_RATIONALE)

    access_info = conn_models.PaperAccessResolutionResponse.model_validate(outputs).access_info
    resolved_source = access_info.source if access_info else None

    return Feedback(
        name='access_source_correct',
        value=int(_normalize_source(resolved_source) == _normalize_source(expectations.source)),
        rationale=f'Expected source: {expectations.source}, resolved source: {resolved_source}.',
    )


# A failed prediction (e.g. the service responded with an error) produces no outputs. It scores 0,
# so that it is not mistaken for a paper that was correctly not found.
_FAILED_RATIONALE = 'The prediction failed, so no access information was resolved.'


def _normalize_source(source: str | None) -> str | None:
    """Normalizes an arXiv ID or a PDF URL for comparison."""

    if source is None:
        return None

    source = source.strip()

    if '://' not in source:
        return _ARXIV_VERSION_PATTERN.sub('', source.removeprefix('arXiv:'))

    url = urllib.parse.urlsplit(source)
    path = url.path.rstrip('/').removesuffix('.pdf')
    host = url.netloc.lower().removeprefix('www.')

    if host == 'arxiv.org' and path.startswith('/pdf/'):
        path = _ARXIV_VERSION_PATTERN.sub('', path)

    return urllib.parse.urlunsplit(('https', host, path, url.query, ''))


# Matches the version suffix of an arXiv ID, e.g. `v7` in `1409.0473v7`
_ARXIV_VERSION_PATTERN = re.compile(r'v\d+$')
