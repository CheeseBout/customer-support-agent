"""Agent Skills (SPEC 11.5): step-by-step procedures the agent loads when it needs them.

Only a one-line description of each skill sits in the system prompt. The full text is read when
the model calls `load_skill`, so a long procedure costs nothing until it is relevant.

A skill is a folder holding `SKILL.en.md` and `SKILL.vi.md`, each with a small frontmatter
(`name`, `description`, `lang`) followed by the instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    lang: str
    body: str


def _parse(text: str, where: str) -> Skill:
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError(f"{where}: missing frontmatter")
    meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    for key in ("name", "description", "lang"):
        if key not in meta:
            raise ValueError(f"{where}: frontmatter lacks {key!r}")
    return Skill(meta["name"], meta["description"], meta["lang"], match.group(2).strip())


@lru_cache
def _library() -> dict[tuple[str, str], Skill]:
    found: dict[tuple[str, str], Skill] = {}
    root = resources.files(__package__)
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        for lang in ("en", "vi"):
            path = folder / f"SKILL.{lang}.md"
            if path.is_file():
                skill = _parse(path.read_text(encoding="utf-8"), f"{folder.name}/SKILL.{lang}.md")
                if skill.name != folder.name or skill.lang != lang:
                    raise ValueError(
                        f"{folder.name}/SKILL.{lang}.md: name/lang do not match its path"
                    )
                found[(skill.name, lang)] = skill
    return found


def skill_names() -> list[str]:
    return sorted({name for name, _ in _library()})


def skill_catalog() -> list[tuple[str, str]]:
    """(name, English description) of every skill: the part that goes into the prompt."""
    return [
        (n, _library()[(n, "en")].description) for n in skill_names() if (n, "en") in _library()
    ]


def load_skill(name: str, lang: str = "en") -> Skill | None:
    """The skill in `lang`, falling back to English. None if there is no such skill."""
    library = _library()
    return library.get((name, lang)) or library.get((name, "en"))
