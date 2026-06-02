# Contributing to CogBase

CogBase is in early development. The highest-impact contributions right now are improvements to the existing v1 implementations — not new features. See the [Roadmap](README.md#roadmap) for the specific gaps.

## Setting up

```bash
git clone https://github.com/CogBaseAI/cogbase
cd cogbase
pip install -e ".[dev]"
```

The `dev` extra includes pytest, ruff, mypy, and the store adapters needed for tests. You will need an `OPENAI_API_KEY` set in your environment for tests that hit the LLM or embedder.

## Running tests

```bash
pytest
```

Most tests are unit or integration tests against in-memory stores and do not require external services. Tests that need Postgres or pgvector are skipped automatically when the database is not reachable.

## Code style

```bash
ruff check .          # lint
ruff format .         # format
mypy cogbase api      # type-check
```

All three must pass before submitting a PR. The project targets Python 3.11+ and uses strict mypy settings.

## What to work on

**Improving implementations** — these have the most immediate impact:

- **Pipeline** — native document parsing (PDF, DOCX, HTML); semantic chunking alternatives to the sliding window; extraction retry with schema validation
- **Query runner** — skill selection is re-invoked on every loop turn; wire up `compact_messages` for long sessions
- **Workflows** — per-step timeouts; partial-failure recovery; parallel `foreach`
- **API** — authentication, rate limiting, response pagination
- **Tests** — integration coverage for the query runner loop, workflow execution, and API end-to-end paths

**Extending the framework:**

- **Store adapter** — implement `StructuredStoreBase` or `VectorStoreBase` for a new backend (e.g. Chroma, Weaviate, DynamoDB)
- **Example** — a `config.yaml` + JSON schema + extraction prompt for a new vertical; see `examples/` for the pattern
- **Skill** — any stateless capability that implements the skill interface; see `cogbase/skills/skill.py`

**Planned features** (larger efforts):

- Memory layer (short-term, episodic, long-term)
- Adaptive evolution engine and suggestion API
- App generator

## Submitting a PR

- Keep PRs focused — one concern per PR
- Add or update tests for any changed behavior
- Update docstrings and `docs/` if the public interface changes
- If you are adding a new store adapter or example, include a short README explaining how to use it

## Filing issues

Use GitHub Issues. For bugs, include the CogBase version, the config YAML (redact any secrets), the request that triggered the problem, and the full traceback.
