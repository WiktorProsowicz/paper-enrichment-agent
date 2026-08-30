# Builds the project in development mode
build_project_dev:
    @echo "Building project in development mode..."
    uv venv --python 3.12
    uv sync --group dev
    uv pip install -e .

# Runs local checks for code quality and formatting
run_local_checks:
    @echo "Running pre-commit checks..."
    uv run pre-commit run --files `git ls-files --cached --others --exclude-standard`
    @echo "Running mypy type checks..."
    uv run mypy src/paper_enrichment_agent
    @echo "Running ruff checks..."
    uv run ruff check --diff
    uv run ruff format --check --diff

# Runs the unit tests and reports the coverage of the sources
run_unit_tests:
    @echo "Cleaning test results directory..."
    rm -rf test_results
    mkdir -p test_results

    @echo "Running tests..."
    uv run python -m pytest --import-mode=prepend -s \
        tests/unit --tb=short -v \
        --junitxml=test_results/tests_report.xml \
        -W ignore::DeprecationWarning \
        --cov=src \
        --cov-report=html:test_results/coverage_html_report \
        --disable-warnings

    @echo "Reporting coverage..."
    uv run coverage report

setup_mlflow_server:
    @echo "Setting up MLflow server..."
    export $(cat mlflow.env | xargs)
    docker compose -f infrastructure/mlflow-server.yaml up -d

setup_mlflow_server_clean:
    #!/usr/bin/env bash
    echo "Cleaning up MLflow server..."
    export $(cat mlflow.env | xargs)

    docker compose -f infrastructure/mlflow-server.yaml down
    docker volume rm -f mlflow-server_mlflow-postgres-data mlflow-server_mlflow-artifacts
    docker compose -f infrastructure/mlflow-server.yaml up --build -d

run_llm_eval:
    #!/usr/bin/env bash
    echo "Cleaning up MLflow server..."
    export $(cat app.env | xargs)

    docker compose -f infrastructure/llm-eval.yaml up --attach llm-eval_eval_runner