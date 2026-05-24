# Running CogBase locally with Docker

The demo setup uses SQLite + FAISS + local file storage — no external databases required. Configure your LLM and embedding provider (including API key) via the UI Settings tab after the container starts.

## Option 1: Pull and run a pre-built image (simpler)

```bash
# Run latest, no data persistence
./server/run_docker_hub_demo.sh

# Run a specific version with a local data directory for persistence
./server/run_docker_hub_demo.sh 0.1.0 /path/to/local/data
```

The API is available at `http://localhost:8000`. API docs are at `http://localhost:8000/docs`.

To stop and remove the container:

```bash
docker rm -f cogbase-demo
```

## Option 2: Build and run from source

```bash
docker compose -f server/docker-compose.demo.yml up --build
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
| `run_docker_hub_demo.sh` | Pull a pre-built image from Docker Hub and run it |
| `docker_hub_demo.sh` | Build and push the demo image to Docker Hub |
| `Dockerfile.demo` | Builds the service image |
| `docker-compose.demo.yml` | Wires up ports, env vars, and the data volume |
| `cogbase_system.demo.yaml` | System config: SQLite + FAISS + local document store |
| `../.dockerignore` | Excludes caches and build artifacts from the image |
