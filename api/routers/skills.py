"""Endpoints for listing system-level skills."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import SkillRegistryDep
from api.models import SkillListResponse, SkillResponse

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("", response_model=SkillListResponse)
async def list_skills(skill_registry: SkillRegistryDep) -> SkillListResponse:
    """Return all skills available in the system."""
    skills = skill_registry.all_skills()
    items = [
        SkillResponse(
            name=s.name,
            description=s.description,
            metadata=s.metadata,
            source_path=str(s.source_path) if s.source_path else None,
        )
        for s in skills
    ]
    return SkillListResponse(skills=items, total=len(items))
