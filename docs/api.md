# REST API reference

Applications are created and managed through the REST API. Configuration lives in a YAML file bundled as a ZIP with any referenced prompt templates and JSON schemas.

## Tenancy

Every request carries two tenancy dimensions:

- **`account_id`** — the tenant and security boundary. Supplied via the `X-Account-Id`
  header; defaults to `default` when absent, so single-tenant callers keep working.
- **`namespace_id`** — an organizational unit *inside* an account. Supplied as the
  `{namespace}` URL path segment; defaults to `default`.

An application is unique by `(account_id, namespace_id, name)`. **Name-addressed
application routes** are nested under `/namespaces/{namespace}/applications/...`.
**Account-wide routes** (e.g. `GET /applications`, `/skills`, `/system/config`)
omit the namespace segment and operate across the whole account.

## Namespaces

Namespaces are account-scoped; the namespace id is immutable once created.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/namespaces` | Create a namespace (`namespace_id` + optional `display_name`/`description`) |
| `GET` | `/namespaces` | List the calling account's namespaces |
| `GET` | `/namespaces/{namespace}` | Fetch one namespace |
| `PATCH` | `/namespaces/{namespace}` | Update `display_name`/`description` |
| `DELETE` | `/namespaces/{namespace}` | Delete an empty namespace (refuses `default` and namespaces still holding apps) |

Creating an application auto-registers its namespace (idempotent), so it surfaces in the listing even when never explicitly created.

## App generator

The generator is a stateless, account-scoped chat loop: the client holds the full
message history and sends it each turn; the server runs the agent loop and returns
the assistant reply plus a validated `config_yaml` once one is ready. Deploy is
namespace-scoped, since that is where the application is created.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/generate/chat` | One stateless chat turn; returns `content` + (when ready) a validated `config_yaml` |
| `POST` | `/generate/chat/stream` | Same as `/generate/chat`, streamed as Server-Sent Events |
| `POST` | `/namespaces/{namespace}/generate/deploy` | Deploy a generated `config_yaml` as a new application in the namespace |

## Application lifecycle

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/namespaces/{namespace}/applications` | Create an application from a ZIP bundle |
| `GET` | `/applications` | List all applications across the account (all namespaces) |
| `GET` | `/namespaces/{namespace}/applications` | List applications in the namespace |
| `GET` | `/namespaces/{namespace}/applications/{name}` | Get application metadata |
| `PATCH` | `/namespaces/{namespace}/applications/{name}` | Update config and restart |
| `DELETE` | `/namespaces/{namespace}/applications/{name}` | Remove an application |

## Document ingestion and query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/namespaces/{namespace}/applications/{name}/upload_documents` | Upload documents; each is saved to the document store and then ingested asynchronously via a background task |
| `POST` | `/namespaces/{namespace}/applications/{name}/query` | Answer a query (blocking); accepts optional `system_prompt`, `top_k` in the request body; response includes `input_tokens` and `output_tokens` |
| `POST` | `/namespaces/{namespace}/applications/{name}/query/stream` | Stream query response as Server-Sent Events; same request body as blocking query |

## Documents

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/namespaces/{namespace}/applications/{name}/docs` | List all documents for an application |
| `GET` | `/namespaces/{namespace}/applications/{name}/docs/{doc_id}` | Get a single document record |
| `DELETE` | `/namespaces/{namespace}/applications/{name}/docs/{doc_id}` | Delete a document; cascades to workflow tasks and cleans up associated vector and structured store data |

## Tasks

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/namespaces/{namespace}/applications/{name}/tasks` | List ingest and workflow task records for an application |
| `GET` | `/namespaces/{namespace}/applications/{name}/tasks/{task_id}` | Get a single task record |

## Workflows

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/namespaces/{namespace}/applications/{name}/workflows` | List registered workflow names |
| `GET` | `/namespaces/{namespace}/applications/{name}/workflows/{workflow_name}/docs` | List documents with their status for a given workflow (`pending`, `running`, `done`, `failed`) |
| `POST` | `/namespaces/{namespace}/applications/{name}/workflows/{workflow_name}/stream` | Run a workflow, stream records as SSE |

## Skills management

Application-scoped skill assignment is namespace-scoped; the system skill registry
(upload, list, replace) is account-scoped and shared across namespaces.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/namespaces/{namespace}/applications/{name}/skills` | List skills assigned to an application |
| `POST` | `/namespaces/{namespace}/applications/{name}/skills` | Assign a skill to an application |
| `DELETE` | `/namespaces/{namespace}/applications/{name}/skills/{skill}` | Remove a skill from an application |
| `POST` | `/skills` | Upload a skill bundle to the account's registry |
| `PUT` | `/skills/{skill_name}` | Replace a skill's bundle, keeping its id |
| `GET` | `/skills` | List all skills in the account's registry |
| `DELETE` | `/skills/{skill_name}` | Remove a skill from the account's registry |

## System configuration

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/system/config` | Configure LLM and embedding providers at runtime; changes are persisted in the system database and take effect immediately without a restart |

## Adaptive evolution

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/namespaces/{namespace}/applications/{name}/suggestions` | List pending suggestions with supporting evidence (example queries, score distributions, session IDs) |
| `POST` | `/namespaces/{namespace}/applications/{name}/suggestions/{id}/accept` | Accept a suggestion; triggers config patch + targeted re-ingest |
| `POST` | `/namespaces/{namespace}/applications/{name}/suggestions/{id}/reject` | Reject a suggestion |

## Application config format

Applications are configured via a `config.yaml` inside a ZIP bundle. Any files referenced by filename (JSON schemas, prompt templates) must also be present flat at the ZIP root.

For working config examples, see:
- [`examples/contract_analyst_demo/config.yaml`](../examples/contract_analyst_demo/config.yaml)
- [`examples/contract_compliance_demo/config.yaml`](../examples/contract_compliance_demo/config.yaml)
- [`examples/vc_portfolio_demo/config.yaml`](../examples/vc_portfolio_demo/config.yaml)
- [`examples/legal_case_prep_demo/config.yaml`](../examples/legal_case_prep_demo/config.yaml)
