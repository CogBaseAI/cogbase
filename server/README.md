# Running CogBase locally with Docker

The demo setup uses SQLite + FAISS + local file storage — no external databases required. Configure your LLM and embedding provider (including API key) via the UI Settings tab after the container starts.

## Option 1: Pull and run a pre-built image (simpler)

```bash
# Pull and run latest, no data persistence
./server/docker_hub_demo.sh pull
./server/docker_hub_demo.sh run

# Pull and run a specific version with a local data directory for persistence
./server/docker_hub_demo.sh pull 0.2.0
./server/docker_hub_demo.sh run 0.2.0 /path/to/local/data
```

The API is available at `http://localhost:8000`. API docs are at `http://localhost:8000/docs`.

To stop and remove the container:

```bash
./server/docker_hub_demo.sh stop
```

## Option 2: Build and run from source

```bash
docker compose -f server/docker-compose.demo.yml up --build
```

The API is available at `http://localhost:8000`. API docs are at `http://localhost:8000/docs`.

## Data persistence

All data lives under `/data` inside the container, with these paths:

| Path | Contents |
|------|----------|
| `/data/cogbase_system.db` | Application registry and background task tracking |
| `/data/cogbase.db` | Structured extraction data |
| `/data/faiss_vector_store/` | Vector index |
| `/data/documents/` | Ingested document text |

- **Option 1 without a DIR**: `/data` is inside the container and is lost when the container is removed.
- **Option 1 with a DIR**: `/data` is mounted from the host path you specified.
- **Option 2**: `/data` is mounted from `../data/` relative to `server/` on the host.

To reset to a clean state, remove the host data directory (Options 1+DIR or 2) or stop and remove the container (Option 1 without DIR).

## Files

| File | Purpose |
|------|---------|
| `docker_hub_demo.sh` | Pull, run, stop, build, push, and release the demo image |
| `Dockerfile.demo` | Builds the service image |
| `docker-compose.demo.yml` | Wires up ports, env vars, and the data volume |
| `cogbase_system.demo.yaml` | System config: SQLite + FAISS + local document store |
| `../.dockerignore` | Excludes caches and build artifacts from the image |
