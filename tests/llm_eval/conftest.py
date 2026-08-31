"""This file contains per-eval-run setup.

Each test suite (e.g. test footnote enrichment) should use its own mlflow experiment. The
improvement / regression on a specific system's capability is then visible by comparing individual
run within the same experiment.
"""

import os

import pytest
import mlflow


@pytest.fixture(scope='session', autouse=True)
def setup_mlflow():
    """Sets up connection with the mlflow server and prepares evaluation run."""

    required_env_vars = (
        'LLMEVAL_MLFLOW_URI',
        'LLMEVAL_MLFLOW_EXPERIMENT_NAME',
        'LLMEVAL_JUDGE_LLM_API',
    )

    assert all(var in os.environ for var in required_env_vars), (
        'Not all required environment variables are set.'
    )

    mlflow.set_tracking_uri(os.environ['LLMEVAL_MLFLOW_URI'])
    mlflow.set_experiment(os.environ['LLMEVAL_MLFLOW_EXPERIMENT_NAME'])
