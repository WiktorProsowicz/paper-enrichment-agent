# Builds the project in development mode
build_project_dev:
    @echo "Building project in development mode..."
    uv venv --python 3.12
    uv sync

# Runs local checks for code quality and formatting
run_local_checks:
    @echo "Running pre-commit checks..."
    uv run pre-commit run --files `git ls-files --cached --others --exclude-standard`
    @echo "Running mypy type checks..."
    uv run mypy src/paper_enrichment_agent
    @echo "Running ruff checks..."
    uv run ruff check --diff
    uv run ruff format --check --diff
