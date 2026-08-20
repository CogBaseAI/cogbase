"""Request and response models for the CogBase REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from api.system_store import DocWorkflowStatus, TaskStatus
from cogbase.skills.skill import APPLICATION_SURFACE


# ---------------------------------------------------------------------------
# Identity models
# ---------------------------------------------------------------------------


class WhoAmIResponse(BaseModel):
    """The resolved calling identity the UI bootstraps from on load.

    ``account_id`` is the tenant/security boundary the server resolved for this
    request; the UI adopts it rather than sourcing an account itself. ``mode`` is
    the operator-declared deployment mode (see ``SystemConfig.deployment_mode``)
    and tells the UI whether the account is server-authoritative (read-only) or a
    ``dev`` trust-on-declaration knob it may expose as an editable field.

    ``user_id`` / ``email`` / ``role`` are populated only in a managed mode with a
    valid access token (the authenticated principal); they are null in ``dev`` and
    when the caller is unauthenticated. ``account_id`` is likewise null when a
    managed-mode caller presents no valid token, which the UI reads as "show the
    login screen".
    """

    account_id: str | None = None
    mode: str
    user_id: str | None = None
    email: str | None = None
    role: str | None = None


# ---------------------------------------------------------------------------
# Auth models (first-party email/password)
# ---------------------------------------------------------------------------


class SignupRequest(BaseModel):
    email: str = Field(description="Login email; normalized to lowercase.")
    password: str = Field(min_length=8, description="At least 8 characters.")
    invite_token: str | None = Field(
        default=None,
        description=(
            "When present, join the invite's existing account as a member instead "
            "of creating a new account. The invite's email must match."
        ),
    )


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class InviteRequest(BaseModel):
    email: str = Field(description="Email to invite onto the caller's account.")
    role: Literal["owner", "member"] = Field(
        default="member", description="Role the invitee receives on acceptance."
    )


class TokenResponse(BaseModel):
    """Issued on signup and login: an access token plus a refresh token."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    account_id: str
    user_id: str
    email: str
    role: str


class AccessTokenResponse(BaseModel):
    """Returned by /auth/refresh — a fresh access token only."""

    access_token: str
    token_type: str = "bearer"


class LogoutResponse(BaseModel):
    ok: bool = True


class InviteResponse(BaseModel):
    token: str = Field(description="Invite token to hand to the invitee (via a link).")
    email: str
    account_id: str
    role: str
    expires_at: str


# ---------------------------------------------------------------------------
# Namespace models
# ---------------------------------------------------------------------------


class NamespaceResponse(BaseModel):
    account_id: str
    # The user-facing handle (``NamespaceRecord.name``). Callers address
    # namespaces by this readable name; the internal ``namespace_id`` it resolves
    # to stays server-side.
    name: str
    description: str | None = None
    created_at: str
    updated_at: str


class NamespaceListResponse(BaseModel):
    namespaces: list[NamespaceResponse]
    total: int


class CreateNamespaceRequest(BaseModel):
    name: str = Field(
        description=(
            "A handle you choose for the namespace, like a GitHub org or Slack "
            "workspace name — unique within your account and used directly in URLs. "
            "Must start with a letter or underscore, followed by letters, digits, "
            "underscores, or hyphens. It is fixed once created."
        ),
    )
    description: str | None = Field(default=None, description="Optional description.")


class UpdateNamespaceRequest(BaseModel):
    description: str | None = Field(
        default=None, description="New description; omit to leave unchanged."
    )


class ApplicationResponse(BaseModel):
    name: str
    account_id: str
    # The namespace handle (its user-facing name) the app belongs to; the URL
    # path segment. Maps to the internal ``AppRecord.namespace_id``.
    namespace: str
    status: str   # "initializing" | "active" | "error"
    config: dict[str, Any]
    error: str | None
    created_at: str
    updated_at: str


class ApplicationListResponse(BaseModel):
    applications: list[ApplicationResponse]
    total: int


class AppConfigPatchRequest(BaseModel):
    """Light, allowlisted edit to a running app's presentation/behavior fields.

    Unlike the ZIP-bundle ``PATCH /{app_name}`` (which tears the app down and
    rebuilds it), these fields never require re-wiring pipelines or schemas:
    ``query_intro`` / ``example_queries`` are UI-only, and ``query_prompt`` is
    read fresh per query, so the change is applied in place with no restart.

    Only fields that are explicitly set are applied — omitting a field leaves
    it unchanged; passing ``null`` clears it.
    """

    query_prompt: str | None = None
    query_intro: str | None = None
    example_queries: list[str] | None = None


# ---------------------------------------------------------------------------
# Doc registry models
# ---------------------------------------------------------------------------


class DocResponse(BaseModel):
    doc_id: str
    app_name: str
    status: str   # "active" | "failed" | "deleted"
    ingested_at: str
    metadata: dict[str, Any] = {}


class DocListResponse(BaseModel):
    docs: list[DocResponse]
    total: int


# ---------------------------------------------------------------------------
# Ingest models
# ---------------------------------------------------------------------------


class IngestDocumentsAcceptedResponse(BaseModel):
    task_ids: list[str]
    total: int
    batch_id: str = Field(
        description="Id grouping this upload's tasks; pass to GET /tasks/summary to track the batch."
    )


class IngestResultSummary(BaseModel):
    """Per-document outcome of a finished ingest task."""

    chunks_written: int = 0
    records_extracted: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    warning: str | None = Field(
        default=None,
        description=(
            "Set when the document ingested successfully but produced nothing — "
            "e.g. a scanned/image-only PDF with no extractable text, or content no "
            "pipeline step captured. The task still reports 'done'."
        ),
    )


class TaskResponse(BaseModel):
    task_id: str
    app_name: str
    task_type: str
    task_name: str
    doc_id: str | None
    batch_id: str | None = None
    params_json: str | None
    status: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    error: str | None
    result: IngestResultSummary | None = Field(
        default=None, description="Ingest counts and any warning, once the task has finished."
    )


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


class TaskSummaryResponse(BaseModel):
    """Rollup of a set of background tasks — answers 'did my upload work?'."""

    app_name: str
    batch_id: str | None = None
    total: int
    pending: int
    running: int
    done: int
    failed: int
    chunks_written: int = Field(description="Total vector chunks written across finished ingest tasks.")
    records_extracted: int = Field(description="Total structured records written across finished ingest tasks.")
    warnings: int = Field(description="Number of finished ingest tasks that ingested nothing.")


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    text: str
    history: list[ChatMessage] = []
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Optional system prompt for this request. "
            "When set, overrides the app-level query_prompt configured in the application's "
            "config.yaml. Useful for prompt experimentation. You can expirement prompts and "
            "and set the final prompt into the app config for production."
        ),
    )
    top_k: int = Field(
        default=10,
        description=(
            "Default number of chunks returned per vector_search call. "
            "The LLM may request fewer; this caps and defaults its top_k argument. "
            "Hard upper limit is 20."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional short-term memory session id. When set, the server records "
            "this turn into the session and builds the LLM context from prior turns "
            "in that session, so `history` need not be supplied. Reuse the same id "
            "across requests to continue a conversation. Omit for stateless queries."
        ),
    )


class SessionStartRequest(BaseModel):
    metadata: dict | None = Field(
        default=None, description="Arbitrary session metadata seeded into the short-term cache."
    )
    session_id: str | None = Field(
        default=None, description="Resume an existing session id instead of creating a new one."
    )


class SessionResponse(BaseModel):
    session_id: str


class SessionCloseResponse(BaseModel):
    session_id: str
    distillation: str = Field(
        description="One of 'enqueued' / 'skipped' — whether a distillation task was started on close."
    )
    task_id: str | None = Field(
        default=None, description="The distillation task id when one was enqueued."
    )


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool = Field(description="Always true once the session has been removed.")


class SessionSummary(BaseModel):
    """One row of the conversation-history list (served from the session index)."""

    session_id: str
    title: str | None = Field(
        default=None, description="First user message, truncated; null until the first turn."
    )
    message_count: int = Field(description="Number of user turns recorded in the session.")
    status: str = Field(description="'open' while active, 'closed' after the session is settled.")
    created_at: str = Field(description="ISO-8601 UTC timestamp of the session's first turn.")
    updated_at: str = Field(description="ISO-8601 UTC timestamp of the session's most recent turn.")


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(
        description="The app's chat sessions, most-recently-active first."
    )


class AnswerReferences(BaseModel):
    """The evidence an assistant turn drew on, mirroring ``QueryResponse``.

    Re-hydrated from the ``final_answer`` event so a replayed transcript shows the
    same references the live query returned.
    """

    structured_records: list[dict] = []
    chunks: list[ChunkResponse] = []
    document_slices: list[DocumentSliceResponse] = []
    memories: list[QueryMemoryResponse] = []


class TranscriptMessage(BaseModel):
    role: str = Field(description="'user' or 'assistant'.")
    content: str
    references: AnswerReferences | None = Field(
        default=None,
        description=(
            "The evidence the answer drew on; set only on assistant turns that "
            "recorded references, null for user turns and reference-less answers."
        ),
    )


class SessionTranscriptResponse(BaseModel):
    session_id: str
    messages: list[TranscriptMessage] = Field(
        description="The session's conversation turns in order (user + assistant)."
    )


class MemoryRecordResponse(BaseModel):
    """A long-term memory record surfaced to a reviewer.

    Includes the provenance (``source_event_ids`` / ``evidence_snapshot``) so a
    reviewer can audit the evidence before promoting a gated record to active.
    """

    memory_id: str
    kind: str
    content: str
    entities: list[str] = []
    confidence: float
    status: str
    source_event_ids: list[dict] = []
    evidence_snapshot: dict = {}
    observed_at: datetime
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class PendingMemoriesResponse(BaseModel):
    memories: list[MemoryRecordResponse]


class MemoryListResponse(BaseModel):
    """A page of stored long-term memories for the inspection surface."""

    memories: list[MemoryRecordResponse]
    total: int = Field(description="Number of records on this page.")


class MemoryReviewItem(BaseModel):
    memory_id: str
    decision: Literal["accept", "reject"] = Field(
        description="'accept' promotes the gated record to active; 'reject' marks it superseded."
    )


class MemoryReviewRequest(BaseModel):
    decisions: list[MemoryReviewItem] = Field(
        description="Per-record verdicts applied in one batch (server-capped)."
    )


class MemoryReviewResultItem(BaseModel):
    memory_id: str
    outcome: str = Field(
        description="One of 'accepted' / 'rejected' / 'skipped' (not pending) / 'not_found'."
    )


class MemoryReviewResponse(BaseModel):
    results: list[MemoryReviewResultItem]


class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = {}
    char_offset: int | None = None
    char_length: int | None = None


class DocumentSliceResponse(BaseModel):
    doc_id: str
    offset: int
    length: int
    text: str


class QueryMemoryResponse(BaseModel):
    """A long-term memory the answer drew on.

    A query-facing projection of ``LongTermRecord`` — only the fields useful for
    explaining the answer, without the reviewer-facing provenance carried by
    ``MemoryRecordResponse``.
    """

    memory_id: str
    kind: str
    content: str
    entities: list[str] = []


class AddMemoryMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        description="Speaker role; maps to the episodic continuity thread the distiller reads."
    )
    content: str = Field(description="The message text.")


class AddMemoryRequest(BaseModel):
    """Add a batch of conversation messages to long-term memory (mem0 ``add`` shape)."""

    messages: list[AddMemoryMessage] = Field(
        description="Conversation messages to distill into durable memories, in order."
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session to append to; a fresh one is generated and returned "
            "when omitted, so each call is an isolated, independently-distilled session."
        ),
    )
    metadata: dict | None = Field(
        default=None, description="Arbitrary session metadata seeded onto the session."
    )
    observation_date: datetime | None = Field(
        default=None,
        description=(
            "When the conversation took place; pins relative time references so they "
            "resolve correctly at distill time. Defaults to now."
        ),
    )


class AddMemoryResponse(BaseModel):
    session_id: str = Field(description="The session the messages were appended to.")
    memories: list[QueryMemoryResponse] = Field(
        default=[],
        description="The long-term memories this call created or reinforced (now active).",
    )


class QueryResponse(BaseModel):
    answer: str
    references: AnswerReferences = Field(
        default_factory=AnswerReferences,
        description="The evidence the answer drew on — same shape a transcript turn carries.",
    )
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: str | None = Field(
        default=None,
        description="The short-term memory session id used for this turn, when memory was engaged.",
    )


# ---------------------------------------------------------------------------
# Skill models
# ---------------------------------------------------------------------------


class SkillResponse(BaseModel):
    id: str
    name: str
    description: str
    metadata: dict[str, Any] = {}
    source_path: str | None = None
    builtin: bool = False
    # "application" (assignable to an app) or a platform surface such as
    # "account-onboarding". Uploaded skills are always "application".
    surface: str = APPLICATION_SURFACE


class SkillListResponse(BaseModel):
    skills: list[SkillResponse]
    total: int


class SkillFile(BaseModel):
    path: str        # relative to the bundle root, POSIX separators
    size: int        # bytes
    is_text: bool    # whether the file can be fetched as text via the file endpoint


class SkillContentResponse(BaseModel):
    """Full SKILL.md plus a listing of the bundle's files for the detail view."""

    id: str
    name: str
    markdown: str            # raw SKILL.md content (the text the LLM sees)
    files: list[SkillFile]   # scripts/assets shipped with the skill


class SkillFileResponse(BaseModel):
    path: str
    size: int
    truncated: bool          # true when the file exceeded the read cap
    content: str             # decoded text content


class AddSkillRequest(BaseModel):
    skill_name: str


class AppSkillRef(BaseModel):
    id: str  # skill id — the stable reference stored in the app config
    name: str  # display name, or the id itself when the skill can no longer be resolved
    missing: bool = False  # True when the referenced skill is no longer in the registry (dangling ref)


class AppSkillsResponse(BaseModel):
    app_name: str
    skills: list[AppSkillRef]


# ---------------------------------------------------------------------------
# Collections / structured store models
# ---------------------------------------------------------------------------


class CollectionsResponse(BaseModel):
    structured: list[str]
    vector: list[str]


class FilterRequest(BaseModel):
    field: str
    op: str
    value: Any = None


class CollectionQueryRequest(BaseModel):
    filters: list[FilterRequest] = []
    fields: list[str] | None = None


class CollectionQueryResponse(BaseModel):
    collection: str
    records: list[dict]
    total: int


# ---------------------------------------------------------------------------
# Workflow models
# ---------------------------------------------------------------------------


class WorkflowListResponse(BaseModel):
    app_name: str
    workflows: list[str]


class DocWorkflowResponse(DocResponse):
    workflow_status: DocWorkflowStatus


class WorkflowDocListResponse(BaseModel):
    app_name: str
    workflow_name: str
    docs: list[DocWorkflowResponse]
    total: int


class WorkflowRunRequest(BaseModel):
    doc_id: str | None = None



# ---------------------------------------------------------------------------
# Generator models
# ---------------------------------------------------------------------------


class GenerateChatRequest(BaseModel):
    text: str
    history: list[ChatMessage] = []


class GenerateChatResponse(BaseModel):
    content: str        # display text (CONFIG markers stripped); store full in history
    config_yaml: str | None = None


class GenerateDeployRequest(BaseModel):
    config_yaml: str


class DeployResponse(BaseModel):
    name: str
    status: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Company profile models
# ---------------------------------------------------------------------------


class CompanyProfileResponse(BaseModel):
    """The account's company profile, or the fact that it has none yet.

    ``exists: false`` with a null ``markdown`` is the cold-start answer — the UI
    reads "should I offer onboarding?" as data rather than having to treat a 404
    as a normal state.
    """

    markdown: str | None = None
    exists: bool
    updated_at: str | None = None
    updated_by: str | None = None
    #: "interview" when written by the generator chat, "manual" when edited
    #: through ``PUT /profile``. Null for a profile whose body predates its index
    #: row (e.g. written directly to the document store).
    source: str | None = None


class UpdateCompanyProfileRequest(BaseModel):
    markdown: str = Field(
        description=(
            "The full company-profile markdown, replacing any previous version. "
            "Stable org-wide context — who you are, jurisdictions, regulators, "
            "risk appetite, house style — injected into every app's system prompt."
        ),
    )


class InterviewChatRequest(BaseModel):
    """One turn of the onboarding interview. Stateless: the client holds history."""

    text: str
    history: list[ChatMessage] = []


class InterviewChatResponse(BaseModel):
    """The interview's reply, plus whether this turn wrote the profile.

    ``profile_saved`` is what lets the UI dismiss the onboarding card and refresh
    the Settings view without polling ``GET /profile`` after every turn.
    """

    content: str
    profile_saved: bool = False
    markdown: str | None = None


# ---------------------------------------------------------------------------
# System config models
# ---------------------------------------------------------------------------


class SystemLLMConfigResponse(BaseModel):
    provider: str
    base_url: str
    api_key: str
    model: str
    mini_model: str | None = None


class SystemEmbeddingConfigResponse(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key: str
    dimensions: int


class SystemConfigResponse(BaseModel):
    llm: SystemLLMConfigResponse | None = None
    embedding: SystemEmbeddingConfigResponse | None = None


class UpdateLLMConfig(BaseModel):
    provider: Literal["openai", "openai-compatible"] = "openai"
    model: str
    mini_model: str | None = None
    base_url: str = 'https://api.openai.com/v1'
    api_key: str = Field(
        description="API key. Use 'EMPTY' for local openai-compatible servers that require no auth (e.g. vLLM).",
    )


class UpdateEmbeddingConfig(BaseModel):
    provider: Literal["openai", "openai-compatible"] = "openai"
    model: str
    base_url: str = 'https://api.openai.com/v1'
    api_key: str = Field(
        description="API key. Use 'EMPTY' for local openai-compatible servers that require no auth (e.g. vLLM).",
    )
    dimensions: int


class UpdateSystemConfigRequest(BaseModel):
    llm: UpdateLLMConfig | None = None
    embedding: UpdateEmbeddingConfig | None = None
