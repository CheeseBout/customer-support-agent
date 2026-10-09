"""Agent Skills (SPEC 11.5): step-by-step procedures the agent loads when it needs them.

Only a one-line description of each skill sits in the system prompt. The full text is read when
the model calls `load_skill`, so a long procedure costs nothing until it is relevant.

A skill is a folder holding `SKILL.en.md` and `SKILL.vi.md`, each with a small frontmatter
(`name`, `description`, `lang`) followed by the instructions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from support_agent.core.capabilities import Offer
from support_agent.core.settings import BusinessRules

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    lang: str
    body: str
    # Tools and kinds of request (`request:order`) the procedure needs. A skill the shop cannot
    # follow is not offered.
    requires: tuple[str, ...] = ()


def _parse(text: str, where: str) -> Skill:
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError(f"{where}: missing frontmatter")
    meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    for key in ("name", "description", "lang"):
        if key not in meta:
            raise ValueError(f"{where}: frontmatter lacks {key!r}")
    requires = tuple(r.strip() for r in meta.get("requires", "").split(",") if r.strip())
    return Skill(meta["name"], meta["description"], meta["lang"], match.group(2).strip(), requires)


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


def skill_catalog(offer: Offer | None = None) -> list[tuple[str, str]]:
    """(name, English description) of every skill the shop can follow: the part that goes into
    the prompt. With `offer`, skills that need a tool or a kind of request it lacks are left out."""
    found = []
    for name in skill_names():
        skill = _library().get((name, "en"))
        if skill is not None and (offer is None or offer.satisfies(skill.requires)):
            found.append((name, skill.description))
    return found


_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

# How the payment-method ids in `business_rules.order.payment_methods` read in a sentence.
_PAYMENT_NAMES: dict[str, dict[str, str]] = {
    "en": {
        "cod": "cash on delivery",
        "bank_transfer": "bank transfer",
        "card": "card",
        "momo": "MoMo",
        "zalopay": "ZaloPay",
        "vnpay": "VNPay",
        "paypal": "PayPal",
    },
    "vi": {
        "cod": "khi nhận hàng",
        "bank_transfer": "chuyển khoản",
        "card": "thẻ",
        "momo": "MoMo",
        "zalopay": "ZaloPay",
        "vnpay": "VNPay",
        "paypal": "PayPal",
    },
}


def format_money(amount: int | float, currency: str, lang: str = "en") -> str:
    """500000 VND -> "500,000 VND" (en) or "500.000đ" (vi, for dong); other currencies keep
    their ISO code, and amounts with cents keep two decimals."""
    cents = float(amount) != int(amount)
    text = f"{amount:,.2f}" if cents else f"{int(amount):,}"
    if lang == "vi":
        text = text.translate(str.maketrans(",.", ".,"))
        return f"{text}đ" if currency == "VND" else f"{text} {currency}"
    return f"{text} {currency}"


def _join(items: list[str], lang: str) -> str:
    word = "hoặc" if lang == "vi" else "or"
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} {word} {items[-1]}"


def skill_facts(rules: BusinessRules, lang: str = "en") -> dict[str, str]:
    """The shop's own figures a skill may quote. They come from `business_rules`, so a skill
    never carries a number that the configuration can contradict."""
    names = _PAYMENT_NAMES.get(lang, _PAYMENT_NAMES["en"])
    methods = [names.get(m, m.replace("_", " ")) for m in rules.order.payment_methods]
    cap = rules.order.cod_max_total
    fallback = "giới hạn của cửa hàng" if lang == "vi" else "the shop's limit"
    return {
        "refund_review_amount": format_money(
            rules.refund.auto_review_max_amount, rules.refund.currency, lang
        ),
        "cod_cap": format_money(cap, rules.order.currency, lang) if cap is not None else fallback,
        "max_quantity_per_line": str(rules.order.max_quantity_per_line),
        "max_lines": str(rules.order.max_lines),
        "payment_methods": _join(methods, lang) if methods else fallback,
    }


def render(text: str, facts: Mapping[str, str]) -> str:
    """Fill `{{name}}` placeholders from `facts`; an unknown placeholder is left as written."""
    return _PLACEHOLDER.sub(lambda m: facts.get(m.group(1), m.group(0)), text)


def load_skill(name: str, lang: str = "en", facts: Mapping[str, str] | None = None) -> Skill | None:
    """The skill in `lang`, falling back to English. None if there is no such skill.

    With `facts` (see `skill_facts`) the placeholders in its text are filled in."""
    library = _library()
    skill = library.get((name, lang)) or library.get((name, "en"))
    if skill is None or facts is None:
        return skill
    return Skill(
        skill.name, skill.description, skill.lang, render(skill.body, facts), skill.requires
    )
