"""Global utilities and core functions for the LLM evaluation framework."""

from typing import Annotated, Any
from collections.abc import Callable
import inspect

import mlflow
from mlflow import entities as mlflow_entities
from mlflow import MlflowClient
from mlflow.genai.datasets import EvaluationDataset
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.models import MetricThreshold
from mlflow.models.evaluation.validation import ModelValidationFailedException
import pydantic
from pydantic import Field
import omegaconf


class EvaluationConfig(pydantic.BaseModel):
    """Pydantic model for the evaluation configuration.

    Is mapped from the llm_eval/cfg.yaml file.
    """

    mlflow_tracking_uri: Annotated[
        str, Field(description='The URI for the MLflow tracking server.')
    ]
    active_suites: Annotated[
        list[str] | None,
        Field(
            description='The list of active test suites. If None, all suites are considered active.'
        ),
    ]
    suite_configs: Annotated[
        dict[str, omegaconf.DictConfig], Field(description='The per-suite configurations')
    ]


class EvaluationRun(pydantic.BaseModel):
    """Represents a single run of the evaluation suite.

    The run is injected to the @mlflow.test decorated cases.
    """

    mlflow_run: mlflow_entities.Run
    suite_config: omegaconf.DictConfig


class EvalSample[InputsT, ExpectationsT](pydantic.BaseModel):
    """One record of an evaluation dataset, as a suite reads it.

    Each suite that wants to use a custom type for test data samples should override the
    from_ds_record method.
    """

    model_config = pydantic.ConfigDict(extra='forbid', frozen=True)

    record_id: str
    inputs: InputsT
    expectations: ExpectationsT


def validate_results(run_id: str, result: EvaluationResult, thresholds: dict[str, float]) -> str:
    """Validates the aggregate scores and records the verdict on the run.

    Args:
        run_id: The run opened for the test case being scored.
        result: What ``mlflow.genai.evaluate`` returned.

    Returns:
        MLflow's failure message, or an empty string when every scorer cleared its
        threshold.
    """
    active_thresholds = {
        f'{name}/mean': MetricThreshold(threshold=value, greater_is_better=True)
        for name, value in thresholds.items()
        if f'{name}/mean' in result.metrics
    }

    failure_reason: str | None = None

    try:
        mlflow.validate_evaluation_results(active_thresholds, candidate_result=result)
    except ModelValidationFailedException as e:
        failure_reason = str(e)

    client = MlflowClient()
    for name, value in result.metrics.items():
        client.log_metric(run_id, name, value)

    client.set_tag(run_id, 'run_passed', not failure_reason)

    if failure_reason:
        client.set_tag(run_id, 'llm_eval_breaches', failure_reason)

    return failure_reason


def load_dataset(dataset_name: str, sample_type: type[EvalSample]) -> EvaluationDataset:
    """Fetches a registered MLFlow dataset and checks every record's shape.

    Args:
        dataset_name: Registered evaluation dataset to read.
        sample_type: The suite's sample class used to validate the json-serialized data.

    Returns:
        The dataset, ready to hand to ``mlflow.genai.evaluate``.

    Raises:
        ValueError: If the dataset holds no records.
        pydantic.ValidationError: If a record does not conform to the expected shape.
    """
    dataset = mlflow.genai.get_dataset(name=dataset_name)
    records = dataset.to_df().to_dict(orient='records')

    if not records:
        message = f'Evaluation dataset {dataset_name!r} holds no records; seed it first'
        raise ValueError(message)

    for record in records:
        sample_type.model_validate(record)

    return dataset


def scorer_with_typed_args() -> Callable[[Callable[..., Any]], mlflow.genai.Scorer]:
    """Enables using pydantic models for scorer arguments.

    MLflow passes a scorer only the arguments it declares, out of a fixed set --
    ``inputs``, ``outputs``, ``expectations``, ``trace``, ``session`` -- and picks
    them by reading the function's signature, so a custom parameter cannot simply be
    added. The scorer decorator returned by this function allows the scorer to declare its arguments
    as pydantic models, and will validate the values passed by MLflow before calling the scorer.

    Returns:
        A decorator turning a typed scorer into an MLflow scorer, keeping the
        function's name.
    """

    def decorate(score: Callable[..., Any]) -> mlflow.genai.Scorer:
        wanted_params_dict = inspect.signature(score).parameters

        def adapter(inputs: dict, outputs: Any, expectations: dict, trace: Any) -> Any:
            available = {
                'inputs': inputs,
                'outputs': outputs,
                'expectations': expectations,
                'trace': trace,
            }

            for arg_name, arg_spec in wanted_params_dict.items():
                if issubclass(arg_spec.annotation, pydantic.BaseModel):
                    available[arg_name] = arg_spec.annotation.model_validate(available[arg_name])

            return score(
                **{name: get() for name, get in available.items() if name in wanted_params_dict}
            )

        # Deliberately not functools.wraps: MLflow reads the registered function's
        # signature to decide what to pass, and wraps would make it report the inner
        # one (sample, ...), which MLflow cannot fill.
        adapter.__name__ = score.__name__
        adapter.__doc__ = score.__doc__

        return mlflow.genai.scorer(adapter)

    return decorate
