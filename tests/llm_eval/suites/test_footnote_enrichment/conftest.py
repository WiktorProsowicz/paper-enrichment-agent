import pytest


@pytest.fixture(scope='module')
def suite_name() -> str:
    return 'test_footnote_enrichment'
