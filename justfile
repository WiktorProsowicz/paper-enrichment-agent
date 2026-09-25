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
    docker build -t app-base:eval --target eval -f infrastructure/app-base.Dockerfile .
    docker compose -f infrastructure/unit-tests.yaml up --build --abort-on-container-failure

# Sets up a local MLFlow server for running evaluation pipelines
setup_mlflow_server:
    #!/usr/bin/env bash
    echo "Setting up MLflow server..."

    if [ ! -f mlflow.env ]; then
        echo "mlflow.env file not found. Please create it with the necessary environment variables."
        exit 1
    fi

    docker compose -f infrastructure/mlflow-server.yaml --env-file mlflow.env up -d

# Cleans up and rebuilds the local MLFlow server
setup_mlflow_server_clean:
    #!/usr/bin/env bash
    echo "Cleaning up MLflow server..."

    if [ ! -f mlflow.env ]; then
        echo "mlflow.env file not found. Please create it with the necessary environment variables."
        exit 1
    fi

    docker compose -f infrastructure/mlflow-server.yaml --env-file mlflow.env down -v
    docker compose -f infrastructure/mlflow-server.yaml --env-file mlflow.env up --build -d

# Runs the LLM evaluation pipeline
run_llm_eval:
    #!/usr/bin/env bash
    echo "Cleaning up MLflow server..."

    docker build -t app-base:eval --target eval -f infrastructure/app-base.Dockerfile .
    docker build -t app-base:runtime --target runtime -f infrastructure/app-base.Dockerfile .

    if [ ! -f llm-eval.env ]; then
        echo "llm-eval.env file not found. Please create it with the necessary environment variables."
        exit 1
    fi

    docker compose -f infrastructure/llm-eval.yaml --env-file llm-eval.env down -v
    docker compose -f infrastructure/llm-eval.yaml --env-file llm-eval.env up \
        --build --attach eval-runner --abort-on-container-exit --exit-code-from eval-runner
