"""System store — persists application metadata in a configurable structured store."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel

from cogbase.core.models import DocWorkflowStatus, TaskStatus
from cogbase.stores import Col, CollectionSchema, FieldSchema, FieldType, StructuredStoreBase

# Tenancy: every system-store record carries ``account_id`` (the tenant / security
# boundary, supplied via the X-Account-Id header) and ``namespace_id`` (an in-account
# organizational unit, addressed as a URL path segment). An application is unique by
# ``(account_id, namespace_id, name)``; ``app_id`` remains the global UUID primary key.
DOC_REGISTRY_SCHEMA = CollectionSchema(
    name="doc_registry",
    description="Document registry: one record per successfully ingested document per application.",
    primary_fields=["app_id", "doc_id"],
    fields={
        "account_id":   FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "namespace_id": FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "app_id":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "doc_id":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "status":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "ingested_at": FieldSchema(type=FieldType.STRING, nullable=False),
        "metadata":    FieldSchema(type=FieldType.STRING, nullable=True),  # JSON blob
    },
)


APP_RECORDS_SCHEMA = CollectionSchema(
    name="app_records",
    description="CogBase application registry: configuration, status, and error state per application.",
    primary_fields=["app_id"],
    fields={
        "app_id":      FieldSchema(type=FieldType.STRING, nullable=False),
        "account_id":   FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "namespace_id": FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "name":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),  # client-facing handle (unique per account+namespace)
        "config_yaml": FieldSchema(type=FieldType.STRING, nullable=False),
        "status":      FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "error":       FieldSchema(type=FieldType.STRING, nullable=True),
        "created_at":  FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at":  FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


NAMESPACE_RECORDS_SCHEMA = CollectionSchema(
    name="namespace_records",
    description="Namespace registry: one record per namespace per account (an in-account organizational unit).",
    primary_fields=["account_id", "namespace_id"],
    fields={
        "account_id":   FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "namespace_id": FieldSchema(type=FieldType.STRING, nullable=False, index=True),  # stable internal id
        "name":         FieldSchema(type=FieldType.STRING, nullable=False, index=True),  # user-facing handle (unique per account); the name→id lookup indexes this
        "description":  FieldSchema(type=FieldType.STRING, nullable=True),
        "created_at":   FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at":   FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


SYSTEM_CONFIG_OVERRIDES_SCHEMA = CollectionSchema(
    name="system_config_overrides",
    description="Runtime overrides for system-level LLM and embedding configuration, set via PATCH /system/config.",
    primary_fields=["key"],
    fields={
        "key":        FieldSchema(type=FieldType.STRING, nullable=False),
        "value_json": FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at": FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


TASKS_SCHEMA = CollectionSchema(
    name="tasks",
    description="Background task records tracking ingest and workflow execution per application.",
    primary_fields=["task_id"],
    fields={
        "task_id":      FieldSchema(type=FieldType.STRING, nullable=False),
        "account_id":   FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "namespace_id": FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "app_id":       FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "task_type":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "task_name":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "doc_id":       FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "batch_id":     FieldSchema(type=FieldType.STRING, nullable=True, index=True),
        "params_json":  FieldSchema(type=FieldType.STRING, nullable=True),
        "status":       FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "created_at":   FieldSchema(type=FieldType.STRING, nullable=False),
        "started_at":   FieldSchema(type=FieldType.STRING, nullable=True),
        "completed_at": FieldSchema(type=FieldType.STRING, nullable=True),
        "error":        FieldSchema(type=FieldType.STRING, nullable=True),
        # JSON summary of a finished ingest: chunks_written, records_extracted,
        # and an optional human-readable warning (e.g. nothing was ingested).
        "result_json":  FieldSchema(type=FieldType.STRING, nullable=True),
    },
)


SKILL_RECORDS_SCHEMA = CollectionSchema(
    name="skill_records",
    description="System-wide skill registry: metadata and bundle location per uploaded skill.",
    primary_fields=["skill_id"],
    fields={
        "skill_id":      FieldSchema(type=FieldType.STRING, nullable=False),
        "account_id":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "namespace_id":  FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "name":          FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "description":   FieldSchema(type=FieldType.STRING, nullable=True),
        "metadata_json": FieldSchema(type=FieldType.STRING, nullable=True),  # JSON blob
        "bundle_key":    FieldSchema(type=FieldType.STRING, nullable=False),  # document-store key
        "created_at":    FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at":    FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


PROFILE_RECORDS_SCHEMA = CollectionSchema(
    name="profile_records",
    description=(
        "Company-profile index: one record per account holding the edit metadata "
        "for the account's company-profile document. The markdown body itself "
        "lives in the document store (see cogbase/core/profile.py) — this table "
        "only answers 'who changed it, when, and how'."
    ),
    # One profile per account, so the account is the whole key.
    primary_fields=["account_id"],
    fields={
        "account_id": FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at": FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_by": FieldSchema(type=FieldType.STRING, nullable=True),  # user id/email when authenticated
        "source":     FieldSchema(type=FieldType.STRING, nullable=False),  # "interview" | "manual"
    },
)


SESSION_RECORDS_SCHEMA = CollectionSchema(
    name="session_records",
    description=(
        "Conversation session index: one record per chat session per application, "
        "holding the list-view metadata (title, activity) so the session history "
        "can be listed without replaying every episodic log. Message content stays "
        "in the episodic log — this is only the index."
    ),
    # Keyed by (app_id, session_id): session_id is client-suppliable and only
    # unique within an app, so app_id is part of the identity — two apps may hold
    # the same session_id without one's upsert clobbering the other's row.
    primary_fields=["app_id", "session_id"],
    fields={
        "session_id":    FieldSchema(type=FieldType.STRING, nullable=False),
        "account_id":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "namespace_id":  FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "app_id":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "title":         FieldSchema(type=FieldType.STRING, nullable=True),   # first user message, truncated
        "message_count": FieldSchema(type=FieldType.INTEGER, nullable=False),
        "status":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),  # "open" | "closed"
        "created_at":    FieldSchema(type=FieldType.STRING, nullable=False),
        "updated_at":    FieldSchema(type=FieldType.STRING, nullable=False),  # ISO-8601 UTC; list ordering
    },
)


ACCOUNT_RECORDS_SCHEMA = CollectionSchema(
    name="account_records",
    description="Tenant registry: one record per account (the security boundary).",
    primary_fields=["account_id"],
    fields={
        "account_id": FieldSchema(type=FieldType.STRING, nullable=False),
        "name":       FieldSchema(type=FieldType.STRING, nullable=False),
        "created_at": FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


USER_RECORDS_SCHEMA = CollectionSchema(
    name="user_records",
    description="User registry: one record per user, each bound to exactly one account.",
    primary_fields=["user_id"],
    fields={
        "user_id":       FieldSchema(type=FieldType.STRING, nullable=False),
        # Email is the login handle — globally unique across the deployment and
        # indexed for the login lookup.
        "email":         FieldSchema(type=FieldType.STRING, nullable=False, index=True, unique=True),
        "password_hash": FieldSchema(type=FieldType.STRING, nullable=False),
        "account_id":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "role":          FieldSchema(type=FieldType.STRING, nullable=False),  # "owner" | "member"
        "created_at":    FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


INVITE_RECORDS_SCHEMA = CollectionSchema(
    name="invite_records",
    description="Pending invitations binding an email to an existing account until accepted.",
    primary_fields=["token"],
    fields={
        "token":       FieldSchema(type=FieldType.STRING, nullable=False),
        "account_id":  FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "email":       FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "role":        FieldSchema(type=FieldType.STRING, nullable=False),
        "created_at":  FieldSchema(type=FieldType.STRING, nullable=False),
        "expires_at":  FieldSchema(type=FieldType.STRING, nullable=False),
        "accepted_at": FieldSchema(type=FieldType.STRING, nullable=True),  # null while pending
    },
)


REFRESH_TOKEN_RECORDS_SCHEMA = CollectionSchema(
    name="refresh_token_records",
    description=(
        "Refresh-token registry, keyed by the token's SHA-256 hash (the raw token "
        "is never stored). Presence + not-revoked + not-expired authorizes minting "
        "a new access token; deleting/revoking a row invalidates the session."
    ),
    primary_fields=["token_hash"],
    fields={
        "token_hash": FieldSchema(type=FieldType.STRING, nullable=False),
        "user_id":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "account_id": FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "created_at": FieldSchema(type=FieldType.STRING, nullable=False),
        "expires_at": FieldSchema(type=FieldType.STRING, nullable=False),
        "revoked_at": FieldSchema(type=FieldType.STRING, nullable=True),  # null while active
    },
)


DOC_WORKFLOW_REGISTRY_SCHEMA = CollectionSchema(
    name="doc_workflow_registry",
    description="Workflow processing status per document per workflow. One record per (app, doc, workflow).",
    primary_fields=["app_id", "doc_id", "workflow_name"],
    fields={
        "account_id":    FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "namespace_id":  FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "app_id":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "doc_id":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "workflow_name": FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "status":        FieldSchema(type=FieldType.STRING, nullable=False, index=True),
        "updated_at":    FieldSchema(type=FieldType.STRING, nullable=False),
    },
)


class DocRecord(BaseModel):
    account_id: str
    namespace_id: str
    app_id: str
    doc_id: str
    status: str        # "active" | "failed" | "deleted"
    ingested_at: str   # ISO-8601 UTC
    metadata: str | None = None  # JSON blob


class TaskRecord(BaseModel):
    task_id: str
    account_id: str
    namespace_id: str
    app_id: str
    task_type: str      # "ingest" | "workflow" | "distill"
    task_name: str      # "ingest" for ingest; workflow name for workflows; "distill" for distillation
    doc_id: str | None = None
    batch_id: str | None = None  # groups tasks created by one upload call
    params_json: str | None = None  # JSON-serialized params
    status: TaskStatus
    created_at: str     # ISO-8601 UTC — when the task was enqueued
    started_at: str | None = None   # ISO-8601 UTC — when execution began
    completed_at: str | None = None
    error: str | None = None
    result_json: str | None = None  # JSON summary of a finished ingest (counts + warning)


class DocWorkflowRecord(BaseModel):
    account_id: str
    namespace_id: str
    app_id: str
    doc_id: str
    workflow_name: str
    status: DocWorkflowStatus
    updated_at: str  # ISO-8601 UTC


class SystemConfigOverride(BaseModel):
    key: str         # "llm" | "embedding"
    value_json: str  # JSON-serialized LLMConfig or EmbeddingConfig
    updated_at: str  # ISO-8601 UTC


_SESSION_TITLE_MAX = 80


def _session_title(text: str) -> str:
    """Derive a session's list title from its first user message.

    Collapses whitespace and truncates to a single readable line so the history
    sidebar shows a compact label; an empty first message yields an empty title
    (the UI falls back to a placeholder).
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SESSION_TITLE_MAX:
        return collapsed
    return collapsed[: _SESSION_TITLE_MAX - 1].rstrip() + "…"


def new_app_id() -> str:
    """Generate a stable app id that is a valid identifier prefix.

    Scoped collection names are ``<app_id>__<collection>`` and must satisfy the
    store name rule (start with a letter or underscore), so the id is prefixed
    with ``app_`` — a bare ``uuid4().hex`` may start with a digit.
    """
    return f"app_{uuid.uuid4().hex}"


def new_account_id() -> str:
    """Generate a stable account id that is a valid store-collection prefix.

    ``account_id`` is the leading segment of every scoped collection name
    (``<account>__<namespace>__<app>__<collection>``), so like :func:`new_app_id`
    it is prefixed with ``acct_`` to guarantee it starts with a letter.
    """
    return f"acct_{uuid.uuid4().hex}"


def new_user_id() -> str:
    """Generate a stable, opaque user id."""
    return f"user_{uuid.uuid4().hex}"


class AppRecord(BaseModel):
    app_id: str       # stable internal id (primary key)
    account_id: str   # tenant / security boundary
    namespace_id: str # in-account organizational unit
    name: str         # client-facing handle (unique per account+namespace, mutable)
    config_yaml: str
    status: str       # "initializing" | "active" | "error"
    error: str | None = None
    created_at: str   # ISO-8601 UTC
    updated_at: str   # ISO-8601 UTC


class NamespaceRecord(BaseModel):
    account_id: str    # owning tenant
    namespace_id: str  # stable internal id (equals ``name`` today; opaque once rename lands)
    name: str          # user-facing handle, unique per account; what the URL addresses and what resolves to ``namespace_id``
    description: str | None = None
    created_at: str    # ISO-8601 UTC
    updated_at: str    # ISO-8601 UTC


class SkillRecord(BaseModel):
    skill_id: str
    account_id: str   # owning tenant; skills are account-scoped (shared across namespaces)
    namespace_id: str  # stored for uniformity; not used to scope skills
    name: str
    description: str | None = None
    metadata_json: str | None = None  # JSON blob
    bundle_key: str                   # key of the ZIP bundle in the document store
    created_at: str   # ISO-8601 UTC
    updated_at: str   # ISO-8601 UTC


class ProfileRecord(BaseModel):
    account_id: str
    updated_at: str            # ISO-8601 UTC
    updated_by: str | None = None  # authenticated principal, when there is one
    source: str                # "interview" (generator chat) | "manual" (PUT /profile)


class SessionRecord(BaseModel):
    session_id: str
    account_id: str
    namespace_id: str
    app_id: str
    title: str | None = None   # first user message, truncated
    message_count: int = 0
    status: str = "open"       # "open" | "closed"
    created_at: str            # ISO-8601 UTC
    updated_at: str            # ISO-8601 UTC


class AccountRecord(BaseModel):
    account_id: str   # tenant / security boundary (also a store-collection prefix)
    name: str         # display name for the org
    created_at: str   # ISO-8601 UTC


class UserRecord(BaseModel):
    user_id: str
    email: str            # login handle, unique across the deployment
    password_hash: str    # opaque hash string (see api/auth.py); never plaintext
    account_id: str       # the tenant this user belongs to
    role: str             # "owner" | "member"
    created_at: str       # ISO-8601 UTC


class InviteRecord(BaseModel):
    token: str
    account_id: str
    email: str
    role: str             # role the invitee receives on acceptance
    created_at: str       # ISO-8601 UTC
    expires_at: str       # ISO-8601 UTC
    accepted_at: str | None = None  # set once the invite is redeemed


class RefreshTokenRecord(BaseModel):
    token_hash: str       # SHA-256 of the raw refresh token (raw token never stored)
    user_id: str
    account_id: str
    created_at: str       # ISO-8601 UTC
    expires_at: str       # ISO-8601 UTC
    revoked_at: str | None = None   # set on logout; a revoked token cannot refresh


class SystemStore:
    """Thin persistence layer for application metadata.

    Accepts any ``StructuredStoreBase`` backend — SQLite, Postgres, or
    in-memory — configured via ``system_db`` in ``cogbase_system.yaml``.

    Args:
        store: A ready-to-use structured store instance.
    """

    def __init__(self, store: StructuredStoreBase) -> None:
        self._store = store

    @property
    def store(self) -> StructuredStoreBase:
        """The underlying structured store, for an embedder's own collections.

        ``system_db`` is one physical store; an embedder that needs its own
        system-level table (e.g. a service's OTP codes) declares its own
        ``CollectionSchema`` and calls ``create_collection``/``save``/``query_as``
        on this directly, rather than forking ``SystemStore``'s fixed schema set
        or duplicating the ``system_db`` config to stand up a second store.
        """
        return self._store

    async def setup(self) -> None:
        """Create managed collections if they do not exist. Idempotent."""
        await self._store.create_collection(DOC_REGISTRY_SCHEMA)
        await self._store.create_collection(APP_RECORDS_SCHEMA)
        await self._store.create_collection(NAMESPACE_RECORDS_SCHEMA)
        await self._store.create_collection(SYSTEM_CONFIG_OVERRIDES_SCHEMA)
        await self._store.create_collection(TASKS_SCHEMA)
        await self._store.create_collection(DOC_WORKFLOW_REGISTRY_SCHEMA)
        await self._store.create_collection(SKILL_RECORDS_SCHEMA)
        await self._store.create_collection(PROFILE_RECORDS_SCHEMA)
        await self._store.create_collection(SESSION_RECORDS_SCHEMA)
        await self._store.create_collection(ACCOUNT_RECORDS_SCHEMA)
        await self._store.create_collection(USER_RECORDS_SCHEMA)
        await self._store.create_collection(INVITE_RECORDS_SCHEMA)
        await self._store.create_collection(REFRESH_TOKEN_RECORDS_SCHEMA)

    async def save_app(self, record: AppRecord) -> None:
        await self._store.save("app_records", [record.model_dump()])

    async def get_app(
        self, account_id: str, namespace_id: str, name: str
    ) -> AppRecord | None:
        """Resolve an app by its client-facing handle within a tenant scope.

        A ``name`` is only unique within ``(account_id, namespace_id)``, so all
        three are required to address one app.
        """
        rows = await self._store.query_as(
            "app_records",
            filters=[
                Col("account_id") == account_id,
                Col("namespace_id") == namespace_id,
                Col("name") == name,
            ],
            model=AppRecord,
        )
        return rows[0] if rows else None

    async def get_app_by_id(self, app_id: str) -> AppRecord | None:
        """Resolve an app by its global UUID id — scope-agnostic (ids never collide)."""
        rows = await self._store.query_as(
            "app_records",
            filters=[Col("app_id") == app_id],
            model=AppRecord,
        )
        return rows[0] if rows else None

    async def list_apps(
        self,
        account_id: str | None = None,
        namespace_id: str | None = None,
    ) -> list[AppRecord]:
        """List apps, optionally scoped.

        ``(account_id, None)`` lists a whole account across namespaces;
        ``(account_id, namespace_id)`` lists one namespace; ``(None, None)`` lists
        every app in the deployment (startup restore, cross-account admin).
        """
        filters = []
        if account_id is not None:
            filters.append(Col("account_id") == account_id)
        if namespace_id is not None:
            filters.append(Col("namespace_id") == namespace_id)
        return await self._store.query_as(
            "app_records", filters=filters or None, model=AppRecord
        )

    async def delete_app(self, app_id: str) -> None:
        await self._store.delete_records("app_records", filters=[Col("app_id") == app_id])
        await self._store.delete_records("doc_registry", filters=[Col("app_id") == app_id])
        await self._store.delete_records("doc_workflow_registry", filters=[Col("app_id") == app_id])
        await self._store.delete_records("tasks", filters=[Col("app_id") == app_id])
        await self._store.delete_records("session_records", filters=[Col("app_id") == app_id])

    # ------------------------------------------------------------------
    # Namespace registry
    # ------------------------------------------------------------------

    async def save_namespace(self, record: NamespaceRecord) -> None:
        await self._store.save("namespace_records", [record.model_dump()])

    async def get_namespace(
        self, account_id: str, namespace_id: str
    ) -> NamespaceRecord | None:
        """Fetch a namespace by its internal id within an account.

        Callers pass the already-resolved ``namespace_id`` (the API layer maps the
        user-facing name to it via ``resolve_namespace_id`` first).
        """
        rows = await self._store.query_as(
            "namespace_records",
            filters=[
                Col("account_id") == account_id,
                Col("namespace_id") == namespace_id,
            ],
            model=NamespaceRecord,
        )
        return rows[0] if rows else None

    async def list_namespaces(self, account_id: str) -> list[NamespaceRecord]:
        """List a single account's namespaces, most-recently-created first."""
        rows = await self._store.query_as(
            "namespace_records",
            filters=[Col("account_id") == account_id],
            model=NamespaceRecord,
        )
        # ISO-8601 UTC timestamps sort lexicographically, so no parse needed.
        return sorted(rows, key=lambda r: r.created_at, reverse=True)

    async def delete_namespace(self, account_id: str, namespace_id: str) -> None:
        await self._store.delete_records(
            "namespace_records",
            filters=[
                Col("account_id") == account_id,
                Col("namespace_id") == namespace_id,
            ],
        )

    # ------------------------------------------------------------------
    # Auth: accounts / users / invites / refresh tokens
    # ------------------------------------------------------------------

    async def save_account(self, record: AccountRecord) -> None:
        await self._store.save("account_records", [record.model_dump()])

    async def get_account(self, account_id: str) -> AccountRecord | None:
        rows = await self._store.query_as(
            "account_records",
            filters=[Col("account_id") == account_id],
            model=AccountRecord,
        )
        return rows[0] if rows else None

    async def save_user(self, record: UserRecord) -> None:
        await self._store.save("user_records", [record.model_dump()])

    async def get_user_by_email(self, email: str) -> UserRecord | None:
        """Resolve a user by their (globally unique) login email.

        Emails are normalized to lowercase by the auth layer before storage and
        lookup, so callers should pass an already-normalized address.
        """
        rows = await self._store.query_as(
            "user_records",
            filters=[Col("email") == email],
            model=UserRecord,
        )
        return rows[0] if rows else None

    async def get_user_by_id(self, user_id: str) -> UserRecord | None:
        rows = await self._store.query_as(
            "user_records",
            filters=[Col("user_id") == user_id],
            model=UserRecord,
        )
        return rows[0] if rows else None

    async def save_invite(self, record: InviteRecord) -> None:
        await self._store.save("invite_records", [record.model_dump()])

    async def get_invite(self, token: str) -> InviteRecord | None:
        rows = await self._store.query_as(
            "invite_records",
            filters=[Col("token") == token],
            model=InviteRecord,
        )
        return rows[0] if rows else None

    async def mark_invite_accepted(self, token: str) -> None:
        invite = await self.get_invite(token)
        if invite is None:
            return
        record = invite.model_copy(
            update={"accepted_at": datetime.now(timezone.utc).isoformat()}
        )
        await self._store.save("invite_records", [record.model_dump()])

    async def save_refresh_token(self, record: RefreshTokenRecord) -> None:
        await self._store.save("refresh_token_records", [record.model_dump()])

    async def get_refresh_token(self, token_hash: str) -> RefreshTokenRecord | None:
        rows = await self._store.query_as(
            "refresh_token_records",
            filters=[Col("token_hash") == token_hash],
            model=RefreshTokenRecord,
        )
        return rows[0] if rows else None

    async def revoke_refresh_token(self, token_hash: str) -> None:
        """Mark a refresh token revoked (idempotent no-op if it does not exist)."""
        existing = await self.get_refresh_token(token_hash)
        if existing is None:
            return
        record = existing.model_copy(
            update={"revoked_at": datetime.now(timezone.utc).isoformat()}
        )
        await self._store.save("refresh_token_records", [record.model_dump()])

    # ------------------------------------------------------------------
    # Session index (conversation history list)
    # ------------------------------------------------------------------

    async def touch_session(
        self,
        account_id: str,
        namespace_id: str,
        app_id: str,
        session_id: str,
        first_text: str,
    ) -> None:
        """Record a conversation turn against the session index.

        Lazily creates the index row on the session's *first* turn (title taken
        from *first_text*, the first user message, truncated); on later turns it
        just bumps the message count and activity timestamp, leaving the title
        untouched.  The row is what the session-history list reads, so listing
        never has to replay episodic logs — the log stays the source of truth for
        message *content*, this index for the list view.
        """
        now = datetime.now(timezone.utc).isoformat()
        existing = await self.get_session(app_id, session_id)
        if existing is None:
            record = SessionRecord(
                session_id=session_id,
                account_id=account_id,
                namespace_id=namespace_id,
                app_id=app_id,
                title=_session_title(first_text),
                message_count=1,
                status="open",
                created_at=now,
                updated_at=now,
            )
        else:
            record = existing.model_copy(
                update={
                    "message_count": existing.message_count + 1,
                    "updated_at": now,
                }
            )
        await self._store.save("session_records", [record.model_dump()])

    async def close_session_record(self, app_id: str, session_id: str) -> None:
        """Flip a session's index row to ``closed``.

        No-op when the row was never created (a session opened but never asked a
        question, so it has no turns and never entered the history list).  Scoped
        by ``app_id`` — a client-supplied ``session_id`` addressing another app's
        session finds nothing and is a no-op.
        """
        existing = await self.get_session(app_id, session_id)
        if existing is None:
            return
        record = existing.model_copy(
            update={
                "status": "closed",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        await self._store.save("session_records", [record.model_dump()])

    async def delete_session_record(self, app_id: str, session_id: str) -> None:
        """Remove a session's index row.

        No-op when the row was never created (a session opened but never asked a
        question).  Scoped by ``app_id`` so a client-supplied ``session_id`` can
        only ever drop the calling app's own row.  The durable episodic log is
        deleted separately by the app.
        """
        await self._store.delete_records(
            "session_records",
            filters=[Col("app_id") == app_id, Col("session_id") == session_id],
        )

    async def get_session(self, app_id: str, session_id: str) -> SessionRecord | None:
        """Resolve a session index row within an app.

        ``session_id`` is client-suppliable, so it is scoped by the resolved
        ``app_id`` (which already encodes the tenant) to keep one app from
        reading or mutating another's session rows.
        """
        rows = await self._store.query_as(
            "session_records",
            filters=[Col("app_id") == app_id, Col("session_id") == session_id],
            model=SessionRecord,
        )
        return rows[0] if rows else None

    async def list_session_records(self, app_id: str) -> list[SessionRecord]:
        """Return an app's sessions, most-recently-active first."""
        rows = await self._store.query_as(
            "session_records",
            filters=[Col("app_id") == app_id],
            model=SessionRecord,
        )
        # ISO-8601 UTC timestamps sort lexicographically, so no parse needed.
        return sorted(rows, key=lambda r: r.updated_at, reverse=True)

    # ------------------------------------------------------------------
    # Skill registry
    # ------------------------------------------------------------------

    async def save_skill(self, record: SkillRecord) -> None:
        await self._store.save("skill_records", [record.model_dump()])

    async def get_skill(self, skill_id: str) -> SkillRecord | None:
        rows = await self._store.query_as(
            "skill_records",
            filters=[Col("skill_id") == skill_id],
            model=SkillRecord,
        )
        return rows[0] if rows else None

    async def list_skills(self) -> list[SkillRecord]:
        return await self._store.query_as("skill_records", filters=None, model=SkillRecord)

    async def delete_skill(self, skill_id: str) -> None:
        await self._store.delete_records("skill_records", filters=[Col("skill_id") == skill_id])

    # ------------------------------------------------------------------
    # Company-profile index
    # ------------------------------------------------------------------

    async def save_profile_record(
        self,
        account_id: str,
        *,
        source: str,
        updated_by: str | None = None,
    ) -> ProfileRecord:
        """Stamp the account's profile edit metadata, replacing any previous row.

        The document store holds only content, so ``updated_at`` needs a home;
        this mirrors ``skill_records``, which indexes bundles whose bytes live in
        the document store. Returns the record it wrote so the caller can answer
        with it without a read-back.
        """
        record = ProfileRecord(
            account_id=account_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
            updated_by=updated_by,
            source=source,
        )
        await self._store.save("profile_records", [record.model_dump()])
        return record

    async def get_profile_record(self, account_id: str) -> ProfileRecord | None:
        rows = await self._store.query_as(
            "profile_records",
            filters=[Col("account_id") == account_id],
            model=ProfileRecord,
        )
        return rows[0] if rows else None

    async def delete_profile_record(self, account_id: str) -> None:
        await self._store.delete_records(
            "profile_records", filters=[Col("account_id") == account_id]
        )

    # ------------------------------------------------------------------
    # Doc registry
    # ------------------------------------------------------------------

    async def save_doc(self, record: DocRecord) -> None:
        await self._store.save("doc_registry", [record.model_dump()])

    async def get_doc(self, app_id: str, doc_id: str) -> DocRecord | None:
        rows = await self._store.query_as(
            "doc_registry",
            filters=[Col("app_id") == app_id, Col("doc_id") == doc_id],
            model=DocRecord,
        )
        return rows[0] if rows else None

    async def list_docs(
        self,
        app_id: str,
        *,
        status: str | None = None,
    ) -> list[DocRecord]:
        filters = [Col("app_id") == app_id]
        if status is not None:
            filters.append(Col("status") == status)
        return await self._store.query_as("doc_registry", filters=filters, model=DocRecord)

    async def delete_doc(self, app_id: str, doc_id: str) -> None:
        await self._store.delete_records(
            "doc_registry",
            filters=[Col("app_id") == app_id, Col("doc_id") == doc_id],
        )
        await self._store.delete_records(
            "tasks",
            filters=[
                Col("app_id") == app_id,
                Col("doc_id") == doc_id,
                Col("task_type") == "workflow",
            ],
        )
        await self._store.delete_records(
            "doc_workflow_registry",
            filters=[Col("app_id") == app_id, Col("doc_id") == doc_id],
        )

    async def save_system_config_override(self, key: str, value_json: str) -> None:
        record = SystemConfigOverride(
            key=key,
            value_json=value_json,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._store.save("system_config_overrides", [record.model_dump()])

    async def load_system_config_overrides(self) -> dict[str, str]:
        rows = await self._store.query_as(
            "system_config_overrides",
            filters=None,
            model=SystemConfigOverride,
        )
        return {r.key: r.value_json for r in rows}

    # ------------------------------------------------------------------
    # Doc workflow registry
    # ------------------------------------------------------------------

    async def upsert_doc_workflow_status(
        self,
        account_id: str,
        namespace_id: str,
        app_id: str,
        doc_id: str,
        workflow_name: str,
        status: DocWorkflowStatus,
    ) -> None:
        """Create or overwrite the workflow processing status for a document."""
        record = DocWorkflowRecord(
            account_id=account_id,
            namespace_id=namespace_id,
            app_id=app_id,
            doc_id=doc_id,
            workflow_name=workflow_name,
            status=status,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._store.save("doc_workflow_registry", [record.model_dump()])

    async def get_doc_workflow(
        self,
        app_id: str,
        doc_id: str,
        workflow_name: str,
    ) -> DocWorkflowRecord | None:
        rows = await self._store.query_as(
            "doc_workflow_registry",
            filters=[
                Col("app_id") == app_id,
                Col("doc_id") == doc_id,
                Col("workflow_name") == workflow_name,
            ],
            model=DocWorkflowRecord,
        )
        return rows[0] if rows else None

    async def list_doc_workflows(
        self,
        app_id: str,
        *,
        workflow_name: str | None = None,
        doc_id: str | None = None,
        status: DocWorkflowStatus | None = None,
    ) -> list[DocWorkflowRecord]:
        filters = [Col("app_id") == app_id]
        if workflow_name is not None:
            filters.append(Col("workflow_name") == workflow_name)
        if doc_id is not None:
            filters.append(Col("doc_id") == doc_id)
        if status is not None:
            filters.append(Col("status") == status)
        return await self._store.query_as("doc_workflow_registry", filters=filters, model=DocWorkflowRecord)

    # ------------------------------------------------------------------
    # Task tracking
    # ------------------------------------------------------------------

    async def create_task(self, record: TaskRecord) -> None:
        await self._store.save("tasks", [record.model_dump()])

    async def update_task(self, task_id: str, **fields) -> None:
        task = await self.get_task(task_id)
        if task is None:
            return
        await self._store.save("tasks", [task.model_copy(update=fields).model_dump()])

    async def get_task(self, task_id: str) -> TaskRecord | None:
        rows = await self._store.query_as(
            "tasks", filters=[Col("task_id") == task_id], model=TaskRecord
        )
        return rows[0] if rows else None

    async def list_tasks(
        self,
        app_id: str,
        *,
        task_type: str | None = None,
        task_name: str | None = None,
        doc_id: str | None = None,
        batch_id: str | None = None,
        status: TaskStatus | None = None,
    ) -> list[TaskRecord]:
        filters = [Col("app_id") == app_id]
        if task_type is not None:
            filters.append(Col("task_type") == task_type)
        if task_name is not None:
            filters.append(Col("task_name") == task_name)
        if doc_id is not None:
            filters.append(Col("doc_id") == doc_id)
        if batch_id is not None:
            filters.append(Col("batch_id") == batch_id)
        if status is not None:
            filters.append(Col("status") == status)
        return await self._store.query_as("tasks", filters=filters, model=TaskRecord)

    async def create_workflow_tasks(
        self,
        account_id: str,
        namespace_id: str,
        app_id: str,
        workflow_name: str,
        doc_id: str | None,
        params_list: list[dict | None],
    ) -> list[TaskRecord]:
        """Create workflow task records for a batch of param sets in one save.

        Returns the created records (in input order) so callers can pair each
        param set with its task_id. Persists all records in a single
        structured-store write rather than one round-trip per param set.
        """
        now = datetime.now(timezone.utc).isoformat()
        records = [
            TaskRecord(
                task_id=str(uuid.uuid4()),
                account_id=account_id,
                namespace_id=namespace_id,
                app_id=app_id,
                task_type="workflow",
                task_name=workflow_name,
                doc_id=doc_id,
                params_json=json.dumps(params) if params is not None else None,
                status=TaskStatus.PENDING,
                created_at=now,
            )
            for params in params_list
        ]
        if records:
            await self._store.save("tasks", [r.model_dump() for r in records])
        return records

    async def complete_workflow_task(
        self,
        task_id: str,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Mark a workflow task as done or failed."""
        await self.update_task(
            task_id,
            status=TaskStatus.DONE if success else TaskStatus.FAILED,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=error,
        )

    async def create_distill_task(
        self, account_id: str, namespace_id: str, app_id: str, session_id: str
    ) -> str:
        """Create a long-term distillation task for a settled session; return its id.

        Mirrors the workflow/ingest task model so distillation runs are
        inspectable; the session id rides ``params_json``.
        """
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self.create_task(TaskRecord(
            task_id=task_id,
            account_id=account_id,
            namespace_id=namespace_id,
            app_id=app_id,
            task_type="distill",
            task_name="distill",
            doc_id=session_id,
            params_json=json.dumps({"session_id": session_id}),
            status=TaskStatus.PENDING,
            created_at=now,
        ))
        return task_id

    async def start_task(self, task_id: str) -> None:
        """Mark a task as running (execution has begun)."""
        await self.update_task(
            task_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    async def complete_distill_task(
        self, task_id: str, *, success: bool, error: str | None = None
    ) -> None:
        """Mark a distillation task as done or failed."""
        await self.update_task(
            task_id,
            status=TaskStatus.DONE if success else TaskStatus.FAILED,
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=error,
        )
