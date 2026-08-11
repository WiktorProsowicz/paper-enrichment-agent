import sys
import subprocess
import json
import textwrap
import pathlib

import pytest
import pydantic

from paper_enrichment_agent.common import logging_setup


def run_logging(script_content: str, log_file_path: pathlib.Path) -> dict:
    """Runs the given script in a subprocess and gathers everything it has logged.

    The script is handed to the interpreter as-is, so it may contain arbitrary code. The path
    of the JSON log file is passed to it as `sys.argv[1]`.
    """

    process = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script_content), str(log_file_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    with open(log_file_path, 'r', encoding='utf-8') as log_file:
        return {
            'file_log_objects': [json.loads(line) for line in log_file.read().strip().splitlines()],
            'process_log_lines': [
                line for line in process.stdout.decode('utf-8').strip().splitlines()
            ],
            'process_err': process.stderr.decode('utf-8'),
            'log_file_path': log_file_path,
        }


@pytest.fixture
def logging_with_native_module(tmp_path: pathlib.Path):

    log_file_path = tmp_path / 'test_log.jsonl'

    script_content = """
        import pathlib
        import sys

        import pydantic
        import structlog

        from paper_enrichment_agent.common import logging_setup

        class TestModel(pydantic.BaseModel):
            field: str

        logging_setup.setup_logging(pathlib.Path(sys.argv[1]))
        logger = structlog.get_logger("paper_enrichment_agent.test")

        logger.info("Test log message", extra_info="extra_value")
        logger.error("Test error code: %s", 123)
        logger.warning("Test warning message")
        logger.debug("Test debug message", pydantic_obj=TestModel(field="test_value"))
        logger.critical("Test critical message")
        """

    return run_logging(script_content, log_file_path)


@pytest.fixture
def logging_with_other_module(tmp_path: pathlib.Path):

    log_file_path = tmp_path / 'test_log.jsonl'

    script_content = """
        import pathlib
        import sys

        import structlog

        from paper_enrichment_agent.common import logging_setup

        logging_setup.setup_logging(pathlib.Path(sys.argv[1]))
        logger = structlog.get_logger("some_other_module")

        logger.info("Log message")
        logger.warning("Log message")
        logger.error("Log message")
        logger.debug("Log message")
        logger.critical("Log message")
        """

    return run_logging(script_content, log_file_path)


class TestLoggingSetup:
    def test_produces_no_errors(
        self, logging_with_native_module, logging_with_other_module
    ) -> None:
        assert logging_with_native_module['process_err'] == ''
        assert logging_with_other_module['process_err'] == ''

    @pytest.mark.dependency(name='test_logs_to_json_file', scope='class')
    def test_logs_to_json_file(self, logging_with_native_module) -> None:
        log_objects = logging_with_native_module['file_log_objects']

        assert len(log_objects) == 5
        assert all(log['logger'] == 'paper_enrichment_agent.test' for log in log_objects)

        assert log_objects[0]['level'] == 'info'
        assert log_objects[0]['extra_info'] == 'extra_value'
        assert log_objects[0]['event'] == 'Test log message'

        assert log_objects[1]['level'] == 'error'
        assert log_objects[1]['event'] == 'Test error code: 123'

        assert log_objects[2]['level'] == 'warning'
        assert log_objects[2]['event'] == 'Test warning message'

        assert log_objects[3]['level'] == 'debug'
        assert log_objects[3]['event'] == 'Test debug message'
        assert log_objects[3]['pydantic_obj'] == {'field': 'test_value'}

        assert log_objects[4]['level'] == 'critical'
        assert log_objects[4]['event'] == 'Test critical message'

    @pytest.mark.dependency(name='test_logs_to_stdout', scope='class')
    def test_logs_to_stdout(self, logging_with_native_module) -> None:
        log_lines = logging_with_native_module['process_log_lines']

        assert len(log_lines) == 5
        assert all('paper_enrichment_agent.test' in line for line in log_lines)

        assert 'info' in log_lines[0]
        assert 'Test log message' in log_lines[0]
        assert 'extra_info' in log_lines[0]
        assert 'extra_value' in log_lines[0]
        assert 'Test log message' in log_lines[0]

        assert 'error' in log_lines[1]
        assert 'Test error code: 123' in log_lines[1]

        assert 'warning' in log_lines[2]
        assert 'Test warning message' in log_lines[2]

        assert 'debug' in log_lines[3]
        assert 'Test debug message' in log_lines[3]
        assert 'pydantic_obj' in log_lines[3]
        assert "{'field': 'test_value'}" in log_lines[3]

        assert 'critical' in log_lines[4]
        assert 'Test critical message' in log_lines[4]

    @pytest.mark.dependency(depends=['test_logs_to_json_file', 'test_logs_to_stdout'])
    def test_other_modules_log_only_warning(self, logging_with_other_module) -> None:
        assert len(logging_with_other_module['file_log_objects']) == 3
        assert len(logging_with_other_module['process_log_lines']) == 3
        assert all(
            log['level'] not in ('debug', 'info')
            for log in logging_with_other_module['file_log_objects']
        )

    def test_not_fails_when_called_in_test_code(self, tmp_path) -> None:

        class TestModel(pydantic.BaseModel):
            field: str

        logging_setup.setup_logging(tmp_path / 'test_log.jsonl')
        logger = logging_setup.get_logger('paper_enrichment_agent.test')

        logger.info('Test log message', extra_info='extra_value')
        logger.error('Test error code: %s', 123)
        logger.warning('Test warning message')
        logger.debug('Test debug message', pydantic_obj=TestModel(field='test_value'))
        logger.critical('Test critical message')
