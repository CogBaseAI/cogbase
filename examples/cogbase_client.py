"""Shared REST client and interactive-loop helpers for CogBase example demos."""

from __future__ import annotations

import json
import logging
import sys

import httpx

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"


def configure_logging() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=_LOG_FORMAT)


class CogBaseClient:
    def __init__(
        self,
        app_name: str,
        api_base: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.app_name = app_name
        self.api_base = api_base.rstrip("/")
        self._http = http_client

    async def list_apps(self) -> list[dict]:
        resp = await self._http.get(f"{self.api_base}/applications", timeout=10)
        resp.raise_for_status()
        return resp.json()["applications"]

    async def get_app(self) -> dict | None:
        resp = await self._http.get(
            f"{self.api_base}/applications/{self.app_name}", timeout=10
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def create_app(self, bundle: bytes) -> dict:
        resp = await self._http.post(
            f"{self.api_base}/applications",
            files={"bundle": ("bundle.zip", bundle, "application/zip")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    async def delete_app(self, name: str | None = None) -> None:
        target = name or self.app_name
        resp = await self._http.delete(
            f"{self.api_base}/applications/{target}", timeout=10
        )
        if resp.status_code not in (204, 404):
            resp.raise_for_status()

    async def ingest_documents(
        self, documents: list[dict], timeout: float = 120
    ) -> list[dict]:
        resp = await self._http.post(
            f"{self.api_base}/applications/{self.app_name}/ingest_documents",
            json={"documents": documents, "concurrency": 3},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    async def query_stream(self, text: str) -> None:
        async with self._http.stream(
            "POST",
            f"{self.api_base}/applications/{self.app_name}/query/stream",
            json={"text": text},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            print("Answer:\n")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "token" in data:
                    print(data["token"], end="", flush=True)
                elif "result" in data:
                    result = data["result"]
                    if result.get("passthrough") and result.get("structured_records"):
                        print(json.dumps(result["structured_records"], indent=2))
                elif "error" in data:
                    print(f"\n  ERROR: {data['error']}")
            print()

    async def list_collections(self) -> dict:
        """Returns {"structured": [...], "vector": [...]}."""
        resp = await self._http.get(
            f"{self.api_base}/applications/{self.app_name}/collections", timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    async def query_structured(
        self,
        collection: str,
        filters: list[dict] | None = None,
    ) -> list[dict]:
        resp = await self._http.post(
            f"{self.api_base}/applications/{self.app_name}/collections/{collection}/query",
            json={"filters": filters or [], "fields": None},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["records"]


# ---------------------------------------------------------------------------
# Interactive-loop command helpers
# ---------------------------------------------------------------------------


async def cmd_startup(client: CogBaseClient, bundle: bytes) -> dict | None:
    """Get or create the app. Returns app_info, or None if creation failed."""
    app_info = await client.get_app()
    if app_info is None:
        print(f"Creating application '{client.app_name}'...")
        try:
            app_info = await client.create_app(bundle)
        except httpx.HTTPStatusError as exc:
            print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
            return None
        print(f"  status: {app_info['status']}")
        if app_info.get("error"):
            print(f"  error:  {app_info['error']}")
    else:
        print(f"Application '{client.app_name}' already exists (status: {app_info['status']})")
    return app_info


async def cmd_list(client: CogBaseClient) -> None:
    try:
        apps = await client.list_apps()
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    if not apps:
        print("  No applications found.")
    else:
        for app in apps:
            print(f"  {app['name']:<28}  status: {app['status']}")


async def cmd_create(client: CogBaseClient, bundle: bytes) -> None:
    existing = await client.get_app()
    if existing is not None:
        print(f"  Application '{client.app_name}' already exists (status: {existing['status']})")
        return
    print(f"Creating application '{client.app_name}'...")
    try:
        result = await client.create_app(bundle)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    print(f"  status: {result['status']}")
    if result.get("error"):
        print(f"  error:  {result['error']}")


async def cmd_delete(client: CogBaseClient, raw: str) -> None:
    name = raw[len("delete "):].strip()
    if not name:
        print("  Usage: delete <name>")
        return
    confirm = input(f"  Delete application '{name}' and all its data? [y/N] ").strip().lower()
    if confirm == "y":
        try:
            await client.delete_app(name)
        except httpx.HTTPStatusError as exc:
            print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
            return
        print(f"  Application '{name}' deleted.")


async def cmd_reset(client: CogBaseClient) -> bool:
    """Returns True if the app was deleted (caller should break the loop)."""
    confirm = input("  Delete application and all data? [y/N] ").strip().lower()
    if confirm == "y":
        await client.delete_app()
        print("  Application deleted. Restart the demo to start fresh.")
        return True
    return False


async def cmd_list_collections(client: CogBaseClient) -> None:
    try:
        cols = await client.list_collections()
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    print(f"  structured: {cols.get('structured', [])}")
    print(f"  vector:     {cols.get('vector', [])}")


async def cmd_query_structured(
    client: CogBaseClient,
    collection: str,
    filters: list[dict] | None = None,
) -> None:
    print(f"Querying structured collection '{collection}'...")
    try:
        records = await client.query_structured(collection, filters)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    if not records:
        print("  No records found.")
    else:
        print(json.dumps(records, indent=2))
