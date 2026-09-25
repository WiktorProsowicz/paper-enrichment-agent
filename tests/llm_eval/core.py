"""Global utilities and core functions for the LLM evaluation framework."""

import dataclasses
import functools
from typing import Annotated, Any, get_type_hints
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
        dict[str, dict[str, Any]], Field(description='The per-suite configurations')
    ]


@dataclasses.dataclass
class EvaluationRun:
    """Represents a single run of the evaluation suite.

    The run is injected to the @mlflow.test decorated cases.
    """

    mlflow_run: mlflow_entities.Run
    suite_config: omegaconf.DictConfig


def validate_results(run_id: str, result: EvaluationResult, thresholds: dict[str, float]) -> str:
    """Validates the aggregate scores and records the verdict on the run.

    Args:
        run_id: The run opened for the test case being scored.
        result: What ``mlflow.genai.evaluate`` returned.

    Returns:
        MLflow's failure message, or an empty string when every scorer cleared its
        threshold.
    """

    failure_reason: str | None = None

    try:
        mlflow.validate_evaluation_results(
            validation_thresholds={
                f'{name}/mean': MetricThreshold(threshold=value, greater_is_better=True)
                for name, value in thresholds.items()
            },
            candidate_result=result,
        )
    except ModelValidationFailedException as e:
        failure_reason = str(e)

    client = MlflowClient()
    for name, value in result.metrics.items():
        client.log_metric(run_id, name, value)

    client.set_tag(run_id, 'run_passed', not failure_reason)

    if failure_reason:
        client.set_tag(run_id, 'llm_eval_breaches', failure_reason)

    return failure_reason


def load_dataset(
    dataset_name: str,
    inputs_type: type[pydantic.BaseModel],
    expectations_type: type[pydantic.BaseModel] | None = None,
) -> EvaluationDataset:
    """Fetches a registered MLFlow dataset and checks every record's shape.

    Args:
        dataset_name: Registered evaluation dataset to read.
        inputs_type: The type of the inputs for each sample.
        expectations_type: The type of the expectations for each sample, if applicable.

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
        inputs_type.model_validate(record['inputs'])

        if expectations_type:
            expectations_type.model_validate(record['expectations'])

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
        wanted_params_dict = get_type_hints(score)

        def adapter(inputs: dict, outputs: Any, expectations: dict, trace: Any) -> Any:
            available = {
                'inputs': inputs,
                'outputs': outputs,
                'expectations': expectations,
                'trace': trace,
            }

            for arg_name, arg_spec in wanted_params_dict.items():
                if (
                    arg_name in available
                    and inspect.isclass(arg_spec)
                    and issubclass(arg_spec, pydantic.BaseModel)
                ):
                    available[arg_name] = arg_spec.model_validate(available[arg_name])

            return score(
                **{name: arg for name, arg in available.items() if name in wanted_params_dict}
            )

        # Deliberately not functools.wraps: MLflow reads the registered function's
        # signature to decide what to pass, and wraps would make it report the inner
        # one which MLflow cannot fill.
        adapter.__name__ = score.__name__
        adapter.__doc__ = score.__doc__

        return mlflow.genai.scorer(adapter)

    return decorate


def predict_with_typed_inputs(
    predict_fn: Callable[[type[pydantic.BaseModel]], Any],
) -> Callable[..., Any]:
    """Enables using a single `inputs` pydantic model for the predict function.

    The decorator wraps a function with a signature like `def predict(inputs: MyInputsModel) -> Any`
    into a function that is compatible with MLflow's `mlflow.genai.evaluate`, which parameters
    for each field in the inputs dictionary, like `def predict(arg1, arg2, ...) -> Any`.
    """

    inputs_type: type[pydantic.BaseModel] = next(iter(get_type_hints(predict_fn).values()))

    def build_inputs(kwargs: dict[str, Any]) -> pydantic.BaseModel:
        inputs_dict = {
            key: value for key, value in kwargs.items() if key in inputs_type.model_fields
        }
        return inputs_type.model_validate(inputs_dict)

    if inspect.iscoroutinefunction(predict_fn):

        @functools.wraps(predict_fn)
        async def async_adapter(**kwargs: Any) -> Any:
            inputs_model = build_inputs(kwargs)
            return await predict_fn(inputs_model)

        return async_adapter

    @functools.wraps(predict_fn)
    def sync_adapter(**kwargs: Any) -> Any:
        inputs_model = build_inputs(kwargs)
        return predict_fn(inputs_model)

    return sync_adapter
