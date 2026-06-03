# REST API reference

Applications are created and managed through the REST API. Configuration lives in a YAML file bundled as a ZIP with any referenced prompt templates and JSON schemas.

## App generator

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/generate` | Start a generation session from a natural-language description; returns `session_id` + `draft_config` |
| `GET` | `/generate/{session_id}` | Retrieve the current draft config for a session |
| `POST` | `/generate/{session_id}/revise` | Send a follow-up instruction to revise the draft |
| `POST` | `/generate/{session_id}/deploy` | Deploy the current draft as a new application |

## Application lifecycle

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/applications` | Create an application from a ZIP bundle |
| `GET` | `/applications` | List all applications |
| `GET` | `/applications/{name}` | Get application metadata |
| `PATCH` | `/applications/{name}` | Update config and restart |
| `DELETE` | `/applications/{name}` | Remove an application |

## Document ingestion and query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/applications/{name}/upload_documents` | Upload documents; each is saved to the document store and then ingested asynchronously via a background task |
| `POST` | `/applications/{name}/query` | Answer a query (blocking); accepts optional `system_prompt`, `top_k` in the request body; response includes `input_tokens` and `output_tokens` |
| `POST` | `/applications/{name}/query/stream` | Stream query response as Server-Sent Events; same request body as blocking query |

## Documents

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/docs` | List all documents for an application |
| `GET` | `/applications/{name}/docs/{doc_id}` | Get a single document record |
| `DELETE` | `/applications/{name}/docs/{doc_id}` | Delete a document; cascades to workflow tasks and cleans up associated vector and structured store data |

## Tasks

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/tasks` | List ingest and workflow task records for an application |
| `GET` | `/applications/{name}/tasks/{task_id}` | Get a single task record |

## Workflows

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/workflows` | List registered workflow names |
| `GET` | `/applications/{name}/workflows/{workflow_name}/docs` | List documents with their status for a given workflow (`pending`, `running`, `done`, `failed`) |
| `POST` | `/applications/{name}/workflows/{workflow_name}/stream` | Run a workflow, stream records as SSE |

## Skills management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/skills` | List skills assigned to an application |
| `POST` | `/applications/{name}/skills` | Assign a skill to an application |
| `DELETE` | `/applications/{name}/skills/{skill}` | Remove a skill from an application |
| `GET` | `/skills` | List all skills in the system registry |

## System configuration

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/system/config` | Configure LLM and embedding providers at runtime; changes are persisted in the system database and take effect immediately without a restart |

## Adaptive evolution

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/applications/{name}/suggestions` | List pending suggestions with supporting evidence (example queries, score distributions, session IDs) |
| `POST` | `/applications/{name}/suggestions/{id}/accept` | Accept a suggestion; triggers config patch + targeted re-ingest |
| `POST` | `/applications/{name}/suggestions/{id}/reject` | Reject a suggestion |

## Application config format

Applications are configured via a `config.yaml` inside a ZIP bundle. Any files referenced by filename (JSON schemas, prompt templates) must also be present flat at the ZIP root.

For working config examples, see:
- [`examples/contract_analyst_demo/config.yaml`](../examples/contract_analyst_demo/config.yaml)
- [`examples/contract_compliance_demo/config.yaml`](../examples/contract_compliance_demo/config.yaml)
- [`examples/vc_portfolio_demo/config.yaml`](../examples/vc_portfolio_demo/config.yaml)
- [`examples/legal_case_prep_demo/config.yaml`](../examples/legal_case_prep_demo/config.yaml)
