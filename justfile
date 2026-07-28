# Builds the project in development mode
build_project_dev:
    @echo "Building project in development mode..."
    uv venv --python 3.12
    uv sync


