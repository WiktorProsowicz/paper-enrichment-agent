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
    #!/usr/bin/env bash
    set -euo pipefail
    docker compose -f infrastructure/unit-tests.yaml up --build --abort-on-container-failure

setup_mlflow_server:
    #!/usr/bin/env bash
    echo "Setting up MLflow server..."

    if [ ! -f mlflow.env ]; then
        echo "mlflow.env file not found. Please create it with the necessary environment variables."
        exit 1
    fi

    export $(cat mlflow.env | xargs)
    docker compose -f infrastructure/mlflow-server.yaml up -d

setup_mlflow_server_clean:
    #!/usr/bin/env bash
    echo "Cleaning up MLflow server..."

    if [ ! -f mlflow.env ]; then
        echo "mlflow.env file not found. Please create it with the necessary environment variables."
        exit 1
    fi

    export $(cat mlflow.env | xargs)

    docker compose -f infrastructure/mlflow-server.yaml down
    docker volume rm -f mlflow-server_mlflow-postgres-data mlflow-server_mlflow-artifacts
    docker compose -f infrastructure/mlflow-server.yaml up --build -d

run_llm_eval:
    #!/usr/bin/env bash
    echo "Cleaning up MLflow server..."

    if [ ! -f app.env ]; then
        echo "app.env file not found. Please create it with the necessary environment variables."
        exit 1
    fi

    export $(cat app.env | xargs)

    docker compose -f infrastructure/llm-eval.yaml up --attach llm-eval_eval_runner