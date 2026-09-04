"""Global utilities and core functions for the LLM evaluation framework."""

from typing import Annotated, Any
from collections.abc import Callable
import inspect

from mlflow import entities as mlflow_entities
import mlflow
from mlflow.genai.datasets import EvaluationDataset
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


class EvalSample[InputsT, ExpectationsT](pydantic.BaseModel):
    """One record of an evaluation dataset, as a suite reads it.

    Each suite that wants to use a custom type for test data samples should override the
    from_ds_record method.
    """

    model_config = pydantic.ConfigDict(extra='forbid', frozen=True)

    record_id: str
    inputs: InputsT
    expectations: ExpectationsT


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


def sample_scorer(
    sample_type: type[EvalSample],
) -> Callable[[Callable[..., Any]], mlflow.genai.Scorer]:
    """Enables using a custom type of data sample in MLflow scorers.

    MLflow passes a scorer only the arguments it declares, out of a fixed set --
    ``inputs``, ``outputs``, ``expectations``, ``trace``, ``session`` -- and picks
    them by reading the function's signature, so a custom parameter cannot simply be
    added.

    Args:
        sample_type: The suite's sample class.

    Returns:
        A decorator turning ``(sample, outputs, trace) -> Feedback | None`` into a
        scorer, keeping the function's name.
    """

    def decorate(score: Callable[..., Any]) -> mlflow.genai.Scorer:
        wanted = set(inspect.signature(score).parameters)

        def adapter(inputs: dict, outputs: Any, expectations: dict, trace: Any) -> Any:
            available = {
                'sample': lambda: sample_type.model_validate(
                    {
                        'record_id': '',
                        'inputs': inputs,
                        'expectations': expectations,
                    }
                ),
                'outputs': lambda: outputs,
                'trace': lambda: trace,
            }
            return score(**{name: get() for name, get in available.items() if name in wanted})

        # Deliberately not functools.wraps: MLflow reads the registered function's
        # signature to decide what to pass, and wraps would make it report the inner
        # one (sample, ...), which MLflow cannot fill.
        adapter.__name__ = score.__name__
        adapter.__doc__ = score.__doc__

        return mlflow.genai.scorer(adapter)

    return decorate
