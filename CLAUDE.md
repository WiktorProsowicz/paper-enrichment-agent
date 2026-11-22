# CLAUDE.md

Guidelines for Claude Code working in this repository.

The project overview, the architecture diagrams, the backend infrastructure and the full tree of
source files are in [README.md](README.md). Read it first. It is the place for information that
changes often and is shared with the user, so keep it up to date:
- When you add, remove or rename a module, update the `## Project structure` tree.
- When you add or change a compose stack, a service or a `just` recipe, update
  `### Backend infrastructure`.

Stack: Python 3.12, `uv`, FastAPI + uvicorn, pydantic v2, Hydra configs, LangGraph + LangChain +
LiteLLM, FastMCP / `mcp`, MLflow (tracing and LLM evaluation), structlog, prometheus-client, boto3
(S3-compatible SeaweedFS), BeautifulSoup, httpx / requests.

## Skills

Project skills live in `.claude/skills/<skill-name>/SKILL.md`. When you add one, list it here
with a one-line description of when to use it.

- `agent-spec`: write or update the specification of an agent in `docs/agents_spec/<agent_name>/`.
  Use it before implementing a new agent or a change of its graph, prompts, tools, inputs or
  outputs.

## Coding guidelines

### Layout of a backend service

Each service is a subpackage of `src/paper_enrichment_agent/` and has the same layout:
- `cfg/main.yaml`: Hydra config. Secrets and model names default to `''` and are passed as
  `key=value` overrides by docker compose.
- `main.py`: the entrypoint and the **only** place where components are wired together. It holds:
  - `AppCfg`, a pydantic model validated from the Hydra config. It references the nested
    `Configuration` classes of the components.
  - `create_app(...)`, the factory that builds the FastAPI app. Route handlers are thin and only
    delegate to the service.
  - `main()`, decorated with `@hydra.main`.
- `service.py`: the `<Name>Service` class with the business logic, and the `Metrics` class with
  the service's Prometheus metrics.
- `components/`: self-contained building blocks (clients, parsers, agents, tools, serializers),
  injected into the service through its constructor. Related components are grouped in
  subpackages (`db_clients/`, `parsers/`); a subpackage's shared base classes and protocols go to
  `core.py`.
- `components/prompt_templates/*.md`: prompts of the service's agents.

Code used by more than one service goes to `common/`, which holds no service-specific logic. HTTP
request/response bodies shared by services and the eval harness go to
`common/models/service_conn.py`.

Services are run with `uv run python -m paper_enrichment_agent.<service>.main key=value ...`.

### Placement of entities in source modules

- Place protected functions and constants inside the class that uses them. Keep them at module
  level only when more than one class uses them. A module-level protected function or constant
  sits below its last caller.
- A public struct (e.g. a pydantic model) used by a class or function, but not defined inside it,
  sits right above its first user. In `paper_scraping_service/components/fetchers.py`, for
  example, `OpenAlexWork` precedes `OpenAlexScraper`, and `OpenAccessLocation` and
  `UnpaywallResult` precede `UnpaywallFetcher`.
- Inside a class, order members as follows:
  1. Protected constants, internal protected classes and the public `Configuration` class (if
     any).
  2. Public properties.
  3. Public methods.
  4. Protected methods.
- Each package `__init__.py` imports its submodules and lists them in `__all__`.

### Placement of entities in test modules

- `tests/unit/` mirrors the package layout (`tests/unit/<service>/[<subpackage>/]test_<module>.py`).
  Each directory has an `__init__.py`.
- Tests are grouped in classes per unit under test (`class TestRegisterArxivSurvey:`). In tests of
  `service.py` modules, each service method is tested in its own `Test*` class.
- Test names describe behavior: `test_raises_when_parsing_fails`,
  `test_reports_metrics_on_success`, `test_does_not_swallow_unexpected_errors`.
- A fixture used by only one `Test*` class is a method fixture of that class. Fixtures shared by
  several classes, and helper functions, stay at module level, right above their first caller.
- Test data files (saved ar5iv pages, marker outputs) sit next to the test module that uses them,
  e.g. `tests/unit/paper_scraping_service/parsers/example_arxiv_paper_<id>.html`.

### Code style

**Naming**
- Give functions descriptive names that say what they do and to what. This matters most for
  protected functions, when a class has many of them: `PdfParser` defines
  `_extract_text_of_blocks`, not `_extract_text` or `_extracted_text`.

**Typing**
- Annotate all public and protected functions in `src/` and `tests/llm_eval/`.
  `# type: ignore[...]` is acceptable only with a specific code and only where a third-party
  library forces it.
- Use modern 3.12 syntax: PEP 695 generics (`def f[T: pydantic.BaseModel](...)`), `type X = ...`
  aliases, `X | None`, `collections.abc` types, `Self`.

**Imports**
- Use absolute imports from `paper_enrichment_agent...`. Model modules are imported with aliases:
  `from paper_enrichment_agent.common.models import document as doc_models` (also `misc_models`,
  `conn_models`).

**Classes and configuration**
- Configuration is a nested pydantic `Configuration(BaseModel)` class inside the component it
  configures (`EnrichmentTools.Configuration`), referenced from `AppCfg`.
- Write pydantic fields as `name: Annotated[type, Field(description=..., ...)]`, with constraints
  (`min_length`, `ge`, `le`). Secrets are `pydantic.SecretStr`. IDs default to `uuid.uuid4().hex`
  via `default_factory`.
- Each component or service defines its own exception class nested in the class
  (`ArxivParser.ParsingError`). Wrap lower-level errors with `raise self.XError('...') from e`. The
  service layer catches its components' errors, logs them, updates metrics and re-raises them as
  the service error. It must **not** swallow unexpected exceptions; the tests check this.
- When construction needs I/O or validation, use a `@classmethod create(...)` factory. Pass
  dependencies into constructors, never create them inside, so they can be mocked.

**Async**
- Service methods and route handlers are `async`. Blocking work (boto3, `requests`, CPU-bound
  parsing or serialization) is offloaded with `asyncio.to_thread(...)`. Synchronous components say
  so in their docstrings.

**Logging, metrics, tracing**
- Get the module logger lazily:
  ```python
  @functools.cache
  def _logger() -> logging_setup.LoggerType:
      return logging_setup.get_logger(__name__)
  ```
  Log with structured key/value pairs: `_logger().info('Message.', paper_id=..., error=str(e))`.
- Prometheus metrics live in the service's `Metrics` class with `namespace=<service_name>`.
  Pre-initialize the label combinations (`success`/`failure`). Create a single `Metrics` instance
  per process. Time operations with `time.perf_counter()`. When you add metrics, update the
  service's dashboard in `infrastructure/observability/grafana/dashboards/`.
- Agent code is traced with `@mlflow.trace(...)` and `mlflow.start_span`. The eval scorers depend
  on span names (`FootnoteEnrichmentAgent::invoke`, `agent_reasoning`, `agent_action`,
  `call_tool <name>`), so keep them in sync with the `scorers.py` files of `tests/llm_eval/`.

**Agents and prompts**
- Prompts are Markdown files next to the code that uses them (`prompt_templates/`,
  `llm_judge_prompts/`), loaded via `*PromptTemplate.from_template_file` with explicit
  `input_variables`. Do not inline long prompts in Python.
- Agent tools are public methods of a tools class (`EnrichmentTools`, `PaperAccessTools`). **Their
  docstrings are the tool descriptions the LLM sees**, so write them for the agent: when to use the
  tool, example outputs, `Args:`. Raise `ValueError` with an actionable message, since it is
  returned to the agent as the tool error.

**HTTP APIs**
- Endpoints are named after the service method they delegate to. Reads are `GET` with query
  params. Modifications are `POST` with a JSON body model from `common/models/service_conn.py`;
  file uploads use a multipart form. Every service exposes `GET /health` and mounts `/metrics`.

## Documentation standards

- **Inline comments**: avoid them. Let the names and the docstrings explain the code, and comment
  only non-obvious behavior (a workaround, a surprising third-party quirk, a reason that cannot be
  read from the code). Do not add comments that restate what the code does.
- **Modules**: every module starts with a docstring that gives a general description of its role.
  Modules that export classes list them under `Exported classes:`:
  ```python
  """Contains the client class that communicates with the document database.

  The client implements the layout of the documents and surveys within the database, ...

  Exported classes:
      DocDBClient
  """
  ```
- **Classes**: the docstring describes the contract: lifecycle, raised errors, threading and async
  behavior.
- **Public functions and methods**: Google-style docstrings with `Args:`, `Returns:` and `Raises:`.
- **Protected functions and methods**: a one-line docstring (`"""Decodes a text block into a
  paragraph."""`). Use the multi-line form only when the behavior needs a detailed explanation.
- **Constants**: a short comment above each constant:
  ```python
  # Hard limit (in seconds) for a single test case, including prediction and scoring.
  _TEST_CASE_HARD_TIMEOUT = 2 * 60 * 60
  ```
  Comment class fields the same way only when their meaning is not obvious from the name and type.
- **docker compose files**: a short comment above each service describing its role, and above
  each non-obvious setting (ports, healthchecks, environment variables).
- Cross-reference code with Sphinx roles, e.g.
  `` :class:`~paper_enrichment_agent.common.models.document.Document` ``.

## Static checks

Run them with `just run_local_checks` before finishing a change. The recipe runs:
1. **pre-commit** on all tracked and untracked files (`.pre-commit-config.yaml`): trailing
   whitespace, end-of-file, docstring-first, YAML formatting (`yamlfmt`), JSON5 validity,
   `double-quote-string-fixer`, private key detection, and more.
2. **mypy** in `strict` mode on `src/paper_enrichment_agent` and `tests/llm_eval`
   (`tests/unit` is excluded).
3. **ruff check** with the `E, W, F, I, B, UP, SIM` rules and the Google docstring convention
   (`tests/unit` is excluded from linting).
4. **ruff format --check**: line length **100**, **single quotes**, space indentation.

To fix formatting, run `uv run ruff format` and `uv run ruff check --fix`. The configuration lives in
`pyproject.toml`.

## Unit tests

- pytest with `pytest-asyncio` (no global asyncio mode, so every async test needs
  `@pytest.mark.asyncio`), `pytest-mock`, `unittest.mock` and `monkeypatch`.
- **Unit tests never hit the network.** External I/O is always mocked: boto3 clients raise
  `BotocoreClientError`, and `requests.get` / httpx calls are patched on the module path where they
  are used. Fixtures build small hand-made `doc_models.Document` trees. MLflow spans are mocked.
- Coverage must be **≥ 90%**. `main.py` entrypoints and the agent graph modules are omitted
  (`[tool.coverage.report]` in `pyproject.toml`); the agents are covered by the LLM evaluation.
- Run them with `just run_unit_tests` (dockerized, results in `test_results/`). For a quick local
  run: `MLFLOW_DISABLE_AGENT_HINT=1 uv run pytest tests/unit -p no:cacheprovider`. Without that
  variable, MLflow prints an "agent hint" to stderr and `test_logging_setup.py` fails.

## LLM evaluation

The evaluation in `tests/llm_eval/` is built on MLflow GenAI evaluation and the MLflow pytest
plugin (`@mlflow.test`).

### Files

```
tests/llm_eval/
├── cfg.yaml          # Hydra config: MLflow URI, test case timeout, active_suites,
│                     #   judge_llm_config, suite_configs (service URLs, datasets, thresholds)
├── run_eval.py       # Entrypoint: validates cfg, runs pytest once per active suite
├── conftest.py       # Global fixtures: suite_config, MLflow connection, autouse `suite_run`
│                     #   (one MLflow run per test case), hard timeout per test case
├── core.py           # EvaluationConfig, EvaluationRun, load_dataset, validate_results,
│                     #   @scorer_with_typed_args, @predict_with_typed_inputs
├── scorers.py        # Agent-agnostic trace-based scorers shared by the suites
└── suites/test_<suite>/
    ├── __init__.py
    ├── conftest.py   # Overrides the `suite_name` fixture
    ├── data.py       # pydantic models of the dataset `inputs` / `expectations`
    ├── scorers.py    # Suite-specific deterministic scorers and LLM judge factories
    ├── llm_judge_prompts/*.md  # System/user prompts of the LLM judges (if any)
    └── test_<suite>.py         # Test cases
```

### Naming a suite

For a suite evaluating `<capability>`:
- Directory: `tests/llm_eval/suites/test_<capability>/`, test module `test_<capability>.py`. The
  directory name is the entry in `active_suites` of `cfg.yaml`.
- Suite name: `llm-eval/test-<capability>` (kebab case), returned by the `suite_name` fixture in
  the suite's `conftest.py`. It is both the MLflow experiment name and the key in `suite_configs`.
- Dataset: a registered MLflow dataset named `<capability>_ds/v<N>`, e.g. `paper_access_ds/v1`.
  Datasets must be seeded in MLflow beforehand; their local YAML sources in `data/` are gitignored.
- Thresholds: `<metric_name>/mean: <minimal value>` under `suite_configs.<suite name>.thresholds`.
- LLM judges: `spawn_<name>_rating(llm)` factories in the suite's `scorers.py`, with prompts
  `llm_judge_prompts/<name>_rating_{system,user}_prompt.md`.

### Conventions

- **One suite = one MLflow experiment. One test case = one MLflow run.** Only one suite runs per
  process.
- A test case defines an async `predict` (decorated with `@harness_core.predict_with_typed_inputs`
  and `@mlflow.trace`) calling the running service over `HTTPClient`, loads the dataset with
  `harness_core.load_dataset(...)`, runs `mlflow.genai.evaluate(...)`, then calls
  `harness_core.validate_results(...)` and `pytest.fail(...)` on a threshold breach.
- A metric missing from the results only produces a warning note on the run.
- Imports inside the harness use `from llm_eval import core as harness_core` and
  `from llm_eval import scorers as harness_common_scorers` (`tests/` is the pytest rootdir).
- When you add a new evaluated service, add its eval instance to `infrastructure/llm-eval.yaml`
  and its URL to the suite's entry in `cfg.yaml`.

### Running

`just run_llm_eval` builds both images and runs `infrastructure/llm-eval.yaml`: fresh instances of
the evaluated services plus the `eval-runner`. It requires the MLflow server
(`just setup_mlflow_server`) and `llm-eval.env`. To run a subset of suites, change `active_suites`
in `cfg.yaml`. The evaluation calls paid LLM APIs, so do not run it unless the user asks.

## Contribution rules

- **Commits**: conventional-commit style with an optional scope in plain words, e.g.
  `feat(enrichment tools): ...`, `fix(arxiv parser): ...`, `refactor: ...`, `ci(llm eval): ...`,
  `docs: ...`.
- **Before finishing a change**, run `just run_local_checks` and the unit tests, and update the
  relevant docs: README (structure, infrastructure), this file (guidelines), module docstrings,
  compose comments.
- **Secrets**: `mlflow.env`, `app.env` and `llm-eval.env` are gitignored and contain secrets. Never
  commit them or print their contents.
- **New service**: follow the service layout above, add it to `infrastructure/app-backend.yaml`
  with a describing comment, add a Prometheus scrape job and a Grafana dashboard, and make it log to
  `/var/log/paper-enrichment-agent/<service>.jsonl`.
