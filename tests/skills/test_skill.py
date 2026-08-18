import textwrap
from pathlib import Path

import pytest

from cogbase.skills.skill import (
    APPLICATION_SURFACE,
    ONBOARDING_SURFACE,
    Skill,
    _parse_skill,
    load_skill_dir,
    load_skills,
)


def _write_skill_md(tmp_path: Path, content: str) -> Path:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    path = skill_dir / "SKILL.md"
    path.write_text(content)
    return path


VALID_MD = textwrap.dedent("""\
    ---
    name: my-skill
    description: "Does something useful."
    metadata:
      requires:
        bins: [curl]
    ---

    # My Skill

    Run `curl` to fetch data.
""")


def test_parse_skill_valid(tmp_path):
    path = _write_skill_md(tmp_path, VALID_MD)
    skill = _parse_skill(path)
    assert skill is not None
    assert skill.name == "my-skill"
    assert skill.description == "Does something useful."
    assert skill.raw_markdown == VALID_MD
    assert skill.metadata == {"requires": {"bins": ["curl"]}}
    assert skill.source_path == path


def test_parse_skill_no_frontmatter(tmp_path):
    path = _write_skill_md(tmp_path, "# No front matter here\n")
    assert _parse_skill(path) is None


def test_parse_skill_bad_yaml(tmp_path):
    bad = "---\nname: [unclosed\n---\n# body\n"
    path = _write_skill_md(tmp_path, bad)
    assert _parse_skill(path) is None


def test_parse_skill_name_falls_back_to_dir(tmp_path):
    md = "---\ndescription: no name field\n---\n# body\n"
    path = _write_skill_md(tmp_path, md)
    skill = _parse_skill(path)
    assert skill is not None
    assert skill.name == "my-skill"  # parent dir name


def test_parse_skill_empty_metadata(tmp_path):
    md = "---\nname: bare\ndescription: minimal\n---\n# body\n"
    path = _write_skill_md(tmp_path, md)
    skill = _parse_skill(path)
    assert skill is not None
    assert skill.metadata == {}


def test_load_skills_returns_listed_skills(tmp_path):
    (tmp_path / "weather").mkdir()
    (tmp_path / "weather" / "SKILL.md").write_text(
        "---\nname: weather\ndescription: Get weather.\n---\n# body\n"
    )
    skills = load_skills(["weather"], tmp_path)
    assert len(skills) == 1
    assert skills[0].name == "weather"


def test_load_skills_skips_missing(tmp_path):
    skills = load_skills(["nonexistent"], tmp_path)
    assert skills == []


def test_load_skills_nonexistent_dir():
    skills = load_skills(["anything"], "/nonexistent/path")
    assert skills == []


def test_skill_dataclass_fields():
    skill = Skill(name="test", description="desc", raw_markdown="# md")
    assert skill.metadata == {}
    assert skill.source_path is None
    assert skill.site_packages is None
    assert skill.builtin is False


def test_load_skills_marks_builtin(tmp_path):
    (tmp_path / "weather").mkdir()
    (tmp_path / "weather" / "SKILL.md").write_text(
        "---\nname: weather\ndescription: Get weather.\n---\n# body\n"
    )
    skills = load_skills(["weather"], tmp_path)
    assert len(skills) == 1
    assert skills[0].builtin is True


def test_load_skill_dir_not_builtin(tmp_path):
    skill_dir = tmp_path / "abc123"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: uploaded\ndescription: From the store.\n---\n# body\n"
    )
    skill = load_skill_dir(skill_dir, skill_id="abc123")
    assert skill is not None
    assert skill.builtin is False


def test_load_skills_reads_surface_from_metadata(tmp_path):
    """A skills_dir skill may declare a non-application surface — that dir is
    operator-written, so the claim is a deployment decision."""
    (tmp_path / "onboarding").mkdir()
    (tmp_path / "onboarding" / "SKILL.md").write_text(
        "---\nname: onboarding\ndescription: An interview.\n"
        "metadata:\n  surface: account-onboarding\n---\n# body\n"
    )
    skills = load_skills(["onboarding"], tmp_path)
    assert skills[0].surface == ONBOARDING_SURFACE


def test_load_skills_defaults_to_the_application_surface(tmp_path):
    (tmp_path / "weather").mkdir()
    (tmp_path / "weather" / "SKILL.md").write_text(
        "---\nname: weather\ndescription: Get weather.\n---\n# body\n"
    )
    assert load_skills(["weather"], tmp_path)[0].surface == APPLICATION_SURFACE


def test_uploaded_skill_cannot_claim_a_surface(tmp_path):
    """``metadata`` is spec'd as arbitrary str→str and belongs to whoever wrote the
    skill, so a ``surface`` key in an uploaded bundle stays inert data: it survives
    on ``metadata`` untouched but never becomes the skill's surface."""
    skill_dir = tmp_path / "abc123"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: uploaded\ndescription: From the store.\n"
        "metadata:\n  surface: account-onboarding\n---\n# body\n"
    )
    skill = load_skill_dir(skill_dir, skill_id="abc123")
    assert skill.surface == APPLICATION_SURFACE
    assert skill.metadata["surface"] == "account-onboarding"


def test_load_skills_can_skip_dependency_installation(tmp_path, monkeypatch):
    """Inspecting a skills_dir must not build a venv or reach the network.

    A caller validating that a skill is present, parses, and sits on the surface it
    claims wants none of ``ensure_skill_deps``' side effects — and a pack validator
    that pip-installed as a side effect of a check would do it on every startup.
    """
    (tmp_path / "needs-pip").mkdir()
    (tmp_path / "needs-pip" / "SKILL.md").write_text(
        "---\nname: needs-pip\ndescription: Has deps.\n"
        "metadata:\n  install:\n    - type: pip\n      packages: [python-docx]\n---\n# body\n"
    )
    called = []
    monkeypatch.setattr(
        "cogbase.skills.skill.ensure_skill_deps",
        lambda skill, *a, **kw: called.append(skill.name),
    )

    skills = load_skills(["needs-pip"], tmp_path, install_deps=False)

    assert called == []
    assert skills[0].site_packages is None
    assert skills[0].builtin is True          # still the operator-written load path
