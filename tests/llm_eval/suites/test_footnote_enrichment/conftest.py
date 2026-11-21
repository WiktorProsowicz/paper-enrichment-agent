import pytest


@pytest.fixture(scope='module')
def suite_name() -> str:
    return 'llm-eval/test-footnote-enrichment'
