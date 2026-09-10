"""This file contains per-eval-run setup.

Each test suite (e.g. test footnote enrichment) should use its own mlflow experiment. The
improvement / regression on a specific system's capability is then visible by comparing individual
run within the same experiment.

:func:`suite_run` creates a new MLFlow run and should be injected into each @mlflow.test decorated
test case.
"""

from collections.abc import Generator
import pytest
import mlflow
import omegaconf

from llm_eval import core as harness_core


def pytest_configure(config: pytest.Config) -> None:
    if not config.pluginmanager.hasplugin('mlflow.pytest.plugin'):
        raise pytest.UsageError('tests/llm_eval requires the MLflow pytest plugin')


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        '--llm-eval-cfg', action='store', default='{}', help='JSON-serialized Hydra configuration'
    )


@pytest.fixture(scope='module')
def suite_name() -> str:
    """Returns the name of the current test suite.

    Should be overriden by each suite in its `conftest.py` module.
    """
    raise NotImplementedError('Each test suite must override the `suite_name` fixture.')


@pytest.fixture(scope='session')
def _evaluation_config(request: pytest.FixtureRequest) -> harness_core.EvaluationConfig:
    """Fixture that exposes the loaded Hydra config dictionary to tests."""
    raw_cfg = request.config.getoption('--llm-eval-cfg')
    return harness_core.EvaluationConfig.model_validate_json(raw_cfg)


@pytest.fixture(scope='module')
def suite_config(
    suite_name: str, _evaluation_config: harness_core.EvaluationConfig
) -> omegaconf.DictConfig:
    """Returns the configuration for the given test suite.

    See :func:`suite_name` for more details.
    """
    return _evaluation_config.suite_configs[suite_name]


@pytest.fixture(scope='session')
def _mlflow_connected(_evaluation_config: harness_core.EvaluationConfig) -> None:
    mlflow.set_tracking_uri(_evaluation_config.mlflow_tracking_uri)


@pytest.fixture(scope='function', autouse=True)
def suite_run(
    request: pytest.FixtureRequest,
    suite_name: str,
    _evaluation_config: harness_core.EvaluationConfig,
    _mlflow_connected: None,
) -> Generator[harness_core.EvaluationRun, None, None]:
    """Sets up connection with the mlflow server and prepares evaluation run.

    This fixture should be injected to each @mlflow.test decorated test case.
    """

    mlflow.set_experiment(suite_name)

    with mlflow.start_run(run_name=request.node.name) as run:
        yield harness_core.EvaluationRun(
            mlflow_run=run, suite_config=_evaluation_config.suite_configs[suite_name]
        )
