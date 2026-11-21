"""Runs the evaluation suites, based on the provided configuration.

The role of this script is to handle argument overrides (e.g. for secret keys passed to LLM judges)
and filter out inactive test suites.
"""

import subprocess
import sys

import hydra
import omegaconf
from functools import cache

import core as harness_core  # type: ignore
from paper_enrichment_agent.common import logging_setup


@cache
def _logger() -> logging_setup.LoggerType:
    return logging_setup.get_logger(__name__)


@hydra.main(version_base='1.3', config_path='.', config_name='cfg')
def main(eval_cfg_dict: omegaconf.DictConfig) -> None:
    """Main entry point for the evaluation script."""

    _logger().info('Starting evaluation script.')

    eval_cfg = harness_core.EvaluationConfig.model_validate(
        omegaconf.OmegaConf.to_container(eval_cfg_dict)
    )

    if eval_cfg.active_suites is None:
        _logger().info('No active suites specified. Running all test suites.')

        _run_pytest(eval_cfg, 'tests/llm_eval/suites/')
        return

    _logger().info('Running active test suites.')

    for suite_name in eval_cfg.active_suites:
        _logger().info(f'Running test suite: {suite_name}')

        _run_pytest(eval_cfg, f'tests/llm_eval/suites/{suite_name}')


def _run_pytest(eval_cfg: harness_core.EvaluationConfig, tests_path: str) -> None:
    """Runs pytest on the given tests, exiting with pytest's exit code if it fails.

    The failed command is not reported, since it contains the configuration, including secrets.
    """

    try:
        subprocess.run(
            [
                sys.executable,
                '-m',
                'pytest',
                '--capture=tee-sys',
                '--show-capture=all',
                '-v',
                '--llm-eval-cfg',
                eval_cfg.model_dump_json(),
                tests_path,
            ],
            check=True,
        )

    except subprocess.CalledProcessError as e:
        _logger().critical(
            'Running pytest caused an error.', tests_path=tests_path, exit_code=e.returncode
        )
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
