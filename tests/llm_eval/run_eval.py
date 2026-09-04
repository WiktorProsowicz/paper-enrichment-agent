"""Runs the evaluation suites, based on the provided configuration.

The role of this script is to handle argument overrides (e.g. for secret keys passed to LLM judges)
and filter out inactive test suites.
"""

import subprocess
import sys

import hydra
import omegaconf
from functools import cache

from llm_eval import core as harness_core
from paper_enrichment_agent.common import logging_setup


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


@hydra.main(config_path='.', config_name='cfg')
def main(eval_cfg_dict: omegaconf.DictConfig) -> None:
    """Main entry point for the evaluation script."""

    _logger().info('Starting evaluation script.')

    eval_cfg = harness_core.EvaluationConfig.model_validate(
        omegaconf.OmegaConf.to_container(eval_cfg_dict)
    )

    if eval_cfg.active_suites is None:
        _logger().info('No active suites specified. Running all test suites.')

        subprocess.run(
            [
                sys.executable,
                '-m',
                'pytest',
                'tests/llm_eval/suites',
                '--llm-eval-cfg',
                eval_cfg.model_dump_json(),
            ],
            check=True,
        )
        return

    _logger().info('Running active test suites.')

    for suite_name in eval_cfg.active_suites:
        _logger().info(f'Running test suite: {suite_name}')

        subprocess.run(
            [
                sys.executable,
                '-m',
                'pytest',
                f'tests/llm_eval/suites/{suite_name}',
                '--llm-eval-cfg',
                eval_cfg.model_dump_json(),
            ],
            check=True,
        )


if __name__ == '__main__':
    main()
