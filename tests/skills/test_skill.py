import textwrap
from pathlib import Path

import pytest

from cogbase.skills.skill import Skill, _parse_skill, load_skill_dir, load_skills


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
