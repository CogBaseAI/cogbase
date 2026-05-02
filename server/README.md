# Running CogBase locally with Docker

The demo setup uses SQLite + FAISS + local file storage — no external databases required. The only prerequisite is an OpenAI API key.

## Start the service

```bash
cd server
export OPENAI_API_KEY=sk-...
docker compose -f docker-compose.demo.yml up --build
```

The API is available at `http://localhost:8000`. API docs are at `http://localhost:8000/docs`.

## Data persistence

All data is written to `../data/` (relative to `server/`) on the host:

| Path | Contents |
|------|----------|
| `data/cogbase_system.db` | Application registry |
| `data/cogbase.db` | Structured extraction data |
| `data/faiss_vector_store/` | Vector index |
| `data/documents/` | Ingested document text |

Remove the `data/` directory to reset to a clean state.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.demo` | Builds the service image |
| `docker-compose.demo.yml` | Wires up ports, env vars, and the data volume |
| `cogbase_system.demo.yaml` | System config: SQLite + FAISS + local document store |
| `.dockerignore` | Excludes caches and build artifacts from the image |
