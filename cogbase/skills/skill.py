"""Skill dataclass — loaded from a SKILL.md file."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_SKILL_VENVS_DIR = os.path.abspath(
    os.environ.get("COGBASE_SKILL_VENVS_DIR", os.path.expanduser("~/.cogbase/.skill_venvs"))
)

_SAFE_PKG_RE = re.compile(r"^[a-zA-Z0-9_\-\.\[\]>=<!~,\s]+$")


@dataclass
class Skill:
    """Parsed representation of a SKILL.md file."""

    name: str
    description: str
    raw_markdown: str               # full file content — injected as LLM context
    metadata: dict = field(default_factory=dict)
    source_path: Path | None = None
    site_packages: str | None = None  # venv site-packages injected into PYTHONPATH


def _parse_skill(path: Path) -> Skill | None:
    """Parse a SKILL.md file with YAML front-matter."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        logger.exception("[skills] could not read %s: %s", path, e)
        return None

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not fm_match:
        logger.error("[skills] %s: no YAML front-matter, skipping", path.name)
        return None

    try:
        meta = yaml.safe_load(fm_match.group(1)) or {}
    except yaml.YAMLError as e:
        logger.exception("[skills] %s: bad YAML — %s", path.name, e)
        return None

    name = meta.get("name") or path.parent.name
    description = meta.get("description", "")
    metadata = meta.get("metadata") or {}
    if not isinstance(metadata, dict):
        logger.error("[skills] %s: metadata must be a mapping, got %s", path.name, type(metadata).__name__)
        return None

    return Skill(name=name, description=description, raw_markdown=raw, metadata=metadata, source_path=path)


# mtime-based cache: absolute path → (mtime, Skill)
_skill_cache: dict[str, tuple[float, Skill]] = {}


def _load_skill_cached(file_path: str) -> Skill | None:
    """Load and cache a skill, re-parsing only when the file changes."""
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        return None

    cached = _skill_cache.get(file_path)
    if cached and cached[0] == mtime:
        return cached[1]

    skill = _parse_skill(Path(file_path))
    if skill:
        _skill_cache[file_path] = (mtime, skill)
    return skill


def load_skills(skill_names: list[str], skills_dir: str | Path) -> list[Skill]:
    """Load named skills from *skills_dir*, using an mtime-based cache."""
    skills_dir = str(skills_dir)
    if not os.path.exists(skills_dir):
        logger.error("[skills] directory '%s' not found — no skills loaded", skills_dir)
        return []

    available = set(os.listdir(skills_dir))
    skills: list[Skill] = []
    for name in skill_names:
        if name not in available:
            continue
        file_path = os.path.join(skills_dir, name, "SKILL.md")
        if not os.path.exists(file_path):
            continue
        skill = _load_skill_cached(file_path)
        if skill:
            skill.site_packages = ensure_skill_deps(skill)
            skills.append(skill)
            logger.info("[skills] loaded '%s' from %s", skill.name, file_path)
    return skills


def ensure_skill_deps(skill: Skill, venvs_dir: str = _SKILL_VENVS_DIR) -> str | None:
    """Install pip dependencies declared in skill metadata; return site-packages path or None.

    Metadata keys (mirrors openclaw convention):
      requires.bins  — binaries that must exist on PATH (warning if missing)
      requires.env   — env vars that must be set (warning if missing)
      install        — list of specs; only {"type": "pip", "packages": [...]} is acted on
    """
    requires = skill.metadata.get("requires", {})

    for binary in requires.get("bins", []):
        if shutil.which(binary) is None:
            logger.warning("[skills] '%s': required binary '%s' not found on PATH", skill.name, binary)

    for env_var in requires.get("env", []):
        if env_var not in os.environ:
            logger.warning("[skills] '%s': required env var '%s' not set", skill.name, env_var)

    pip_packages: list[str] = []
    for spec in skill.metadata.get("install", []):
        if isinstance(spec, dict) and spec.get("type") == "pip":
            pip_packages.extend(_validate_pip_packages(spec.get("packages", [])))

    if not pip_packages:
        return None

    os.makedirs(venvs_dir, exist_ok=True)
    pkg_hash = hashlib.md5(json.dumps(sorted(pip_packages)).encode()).hexdigest()[:8]
    venv_dir = os.path.join(venvs_dir, f"{skill.name}_{pkg_hash}")

    if not os.path.exists(venv_dir):
        logger.info("[skills] '%s': creating venv at %s", skill.name, venv_dir)
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
            pip_bin = os.path.join(venv_dir, "bin", "pip")
            subprocess.check_call([pip_bin, "install"] + pip_packages)
            logger.info("[skills] '%s': installed %s", skill.name, pip_packages)
        except subprocess.CalledProcessError as e:
            logger.exception("[skills] '%s': dependency install failed — %s", skill.name, e)
            return None

    matches = glob.glob(os.path.join(venv_dir, "lib", "python*", "site-packages"))
    if not matches:
        logger.error("[skills] '%s': site-packages not found in %s", skill.name, venv_dir)
        return None

    return matches[0]


def _validate_pip_packages(packages: list) -> list[str]:
    safe = []
    for pkg in packages:
        if not isinstance(pkg, str):
            logger.warning("[skills] skipping non-string package spec: %r", pkg)
            continue
        if _SAFE_PKG_RE.match(pkg.strip()):
            safe.append(pkg.strip())
        else:
            logger.warning("[skills] skipping unsafe package spec: %r", pkg)
    return safe
