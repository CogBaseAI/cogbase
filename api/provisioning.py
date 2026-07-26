"""Default-workspace provisioning for a freshly-minted account.

When a brand-new account is minted at signup (the no-invite path), we seed it
with a starter workspace so the user lands on something usable rather than an
empty console: a ``legal-team`` namespace holding a ``contract-analyst``
application built from ``examples/contract_analyst_demo``.

The app is created *empty* — no demo documents are ingested — so the user starts
with their own uploads. Provisioning is best-effort: any failure here is logged
and swallowed so it can never fail an otherwise-successful signup (see
``provision_default_workspace``).
"""

from __future__ import annotations

import io
import json
import logging
import pathlib
import zipfile
from datetime import datetime, timezone

from api.app_cache import AppCache, cache_key
from api.factory import build_app
from api.routers.applications import _parse_bundle
from api.system_resources import SystemResources
from api.system_store import AppRecord, NamespaceRecord, SystemStore, new_app_id

logger = logging.getLogger(__name__)

#: The namespace every new account starts with.
DEFAULT_NAMESPACE_NAME = "legal-team"
DEFAULT_NAMESPACE_DESCRIPTION = "Contract review and analysis for the legal team."

_DEMO_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples" / "contract_analyst_demo"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_contract_analyst_bundle() -> bytes:
    """Assemble the contract-analyst ZIP bundle from the demo directory.

    Mirrors ``examples/contract_analyst_demo/demo.py::_build_bundle`` — the two
    JSON schemas the config references are generated from the demo's Pydantic
    models at runtime (they are intentionally not committed to git).
    """
    from examples.contract_analyst_demo.schema import (
        ContractExtraction,
        ContractExtractionRecord,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(_DEMO_DIR / "config.yaml", "config.yaml")
        zf.writestr(
            "contracts_record_schema.json",
            json.dumps(ContractExtractionRecord.model_json_schema(), indent=2),
        )
        zf.writestr(
            "contracts_extraction_schema.json",
            json.dumps(ContractExtraction.model_json_schema(), indent=2),
        )
        zf.write(_DEMO_DIR / "contracts_prompt.txt", "contracts_prompt.txt")
    return buf.getvalue()


async def provision_default_workspace(
    account_id: str,
    *,
    system_store: SystemStore,
    app_cache: AppCache,
    system_resources: SystemResources,
) -> None:
    """Seed a new account with the default namespace + contract-analyst app.

    Best-effort and idempotent: an existing namespace or app is left untouched,
    and no demo documents are ingested. Never raises — the caller (signup) must
    not fail because starter provisioning did.
    """
    try:
        namespace_id = DEFAULT_NAMESPACE_NAME
        if await system_store.get_namespace(account_id, namespace_id) is None:
            now = _now()
            await system_store.save_namespace(NamespaceRecord(
                account_id=account_id,
                namespace_id=namespace_id,
                name=namespace_id,
                description=DEFAULT_NAMESPACE_DESCRIPTION,
                created_at=now,
                updated_at=now,
            ))
            logger.info("Provisioned default namespace '%s' (account=%s)", namespace_id, account_id)

        yaml_text, config = _parse_bundle(_build_contract_analyst_bundle())

        if await system_store.get_app(account_id, namespace_id, config.name) is not None:
            return  # already provisioned — nothing to do

        now = _now()
        app_id = new_app_id()
        record = AppRecord(
            app_id=app_id,
            account_id=account_id,
            namespace_id=namespace_id,
            name=config.name,
            config_yaml=yaml_text,
            status="initializing",
            created_at=now,
            updated_at=now,
        )
        await system_store.save_app(record)

        # Build the app eagerly (mirrors create_application) so its backing
        # collections are created now — a record persisted as "active" without a
        # build would skip that DDL and break the first query/ingest.
        try:
            app = await build_app(
                config, app_id=app_id, account_id=account_id,
                namespace_id=namespace_id, system=system_resources,
                app_status=record.status, task_store=system_store,
            )
            app_cache.add(cache_key(account_id, namespace_id, config.name), app)
            record = record.model_copy(update={"status": "active", "updated_at": _now()})
            logger.info("Provisioned default app '%s' (account=%s)", config.name, account_id)
        except Exception as exc:
            logger.exception("Failed to build default app for account=%s", account_id)
            record = record.model_copy(
                update={"status": "error", "error": str(exc), "updated_at": _now()}
            )
        await system_store.save_app(record)
    except Exception:
        logger.exception("Failed to provision default workspace for account=%s", account_id)
