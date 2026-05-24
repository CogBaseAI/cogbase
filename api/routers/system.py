"""Endpoints for reading and updating the system-level LLM and embedding config."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from cogbase.config.models import EmbeddingConfig, LLMConfig
from cogbase.embeddings import build_embedding
from cogbase.llms import build_llm
from api.dependencies import AppCacheDep, SystemResourcesDep, SystemStoreDep
from api.models import (
    SystemConfigResponse,
    SystemEmbeddingConfigResponse,
    SystemLLMConfigResponse,
    UpdateSystemConfigRequest,
)

router = APIRouter(prefix="/system", tags=["system"])
logger = logging.getLogger(__name__)


def _mask_key(key: str | None) -> str | None:
    if key is None:
        return None
    if key == "EMPTY" or len(key) <= 4:
        return key
    return f"{key[:4]}***{key[-4:]}"


def _llm_response(cfg: LLMConfig) -> SystemLLMConfigResponse:
    return SystemLLMConfigResponse(
        provider=cfg.provider,
        model=cfg.model,
        mini_model=cfg.mini_model,
        base_url=cfg.base_url,
        api_key=_mask_key(cfg.api_key),
    )


def _embedding_response(cfg: EmbeddingConfig) -> SystemEmbeddingConfigResponse:
    return SystemEmbeddingConfigResponse(
        provider=cfg.provider,
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=_mask_key(cfg.api_key),
        dimensions=cfg.dimensions,
    )


@router.get("/config", response_model=SystemConfigResponse)
async def get_system_config(resources: SystemResourcesDep) -> SystemConfigResponse:
    """Return the active system-level LLM and embedding configuration.

    API keys are never included in the response.
    """
    return SystemConfigResponse(
        llm=_llm_response(resources.llm_config) if resources.llm_config else None,
        embedding=_embedding_response(resources.embedding_config) if resources.embedding_config else None,
    )


@router.patch("/config", response_model=SystemConfigResponse)
async def update_system_config(
    body: UpdateSystemConfigRequest,
    resources: SystemResourcesDep,
    app_cache: AppCacheDep,
    system_store: SystemStoreDep,
) -> SystemConfigResponse:
    """Replace the system-level LLM and/or embedding configuration at runtime.

    Providing ``llm`` replaces the entire LLM config; omitting it leaves it
    unchanged.  Same for ``embedding``.  After a successful update, all cached
    application instances are evicted so they rebuild with the new provider on
    their next request.

    API keys are never included in the response.  Changes are applied in-memory
    and do not persist across service restarts — update your system YAML file to
    make them permanent.
    """
    if body.llm is None and body.embedding is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one of 'llm' or 'embedding' must be provided",
        )

    if body.llm is not None:
        u = body.llm
        llm_cfg = LLMConfig(
            provider=u.provider,
            model=u.model,
            mini_model=u.mini_model,
            base_url=u.base_url,
            api_key=u.api_key,
        )
        try:
            new_llm = build_llm(llm_cfg)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to build LLM client: {exc}",
            )
        resources.llm = new_llm
        resources.llm_config = llm_cfg
        await system_store.save_system_config_override("llm", llm_cfg.model_dump_json())
        logger.info("system llm updated provider=%s model=%s", llm_cfg.provider, llm_cfg.model)

    if body.embedding is not None:
        u = body.embedding
        emb_cfg = EmbeddingConfig(
            provider=u.provider,
            model=u.model,
            base_url=u.base_url,
            api_key=u.api_key,
            dimensions=u.dimensions,
        )
        try:
            new_embedder = build_embedding(emb_cfg)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to build embedding client: {exc}",
            )
        resources.embedder = new_embedder
        resources.embedding_config = emb_cfg
        await system_store.save_system_config_override("embedding", emb_cfg.model_dump_json())
        logger.info("system embedding updated provider=%s model=%s", emb_cfg.provider, emb_cfg.model)

    app_cache.clear()
    logger.info("app cache cleared after system config update")

    return SystemConfigResponse(
        llm=_llm_response(resources.llm_config) if resources.llm_config else None,
        embedding=_embedding_response(resources.embedding_config) if resources.embedding_config else None,
    )
