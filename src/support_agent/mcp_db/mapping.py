"""Schema mapping: canonical entities -> the business's own tables/columns/collections.

Expressions allowed in field mappings are deliberately tiny (SPEC 8.2): a column reference,
a string/number constant, `COALESCE(...)` and `CONCAT(...)`. Anything else is rejected at load
time, so an admin-authored mapping can never smuggle arbitrary SQL into a query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# entity -> (required fields, optional fields)
CANONICAL_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "customer": ({"id"}, {"email", "phone", "name"}),
    "order": (
        {"id", "customer_id", "status", "total", "created_at"},
        {"currency", "delivered_at", "shipping_address", "payment_method"},
    ),
    "order_item": ({"order_id", "sku", "quantity", "unit_price"}, {"product_name"}),
    "product": (
        {"sku", "name", "price"},
        {"description", "category", "attributes", "active"},
    ),
    "inventory": ({"sku", "quantity"}, set()),
    "shipment": (
        {"order_id", "status"},
        {"carrier", "tracking_code", "updated_at", "eta"},
    ),
    "return_request": ({"id", "order_id", "status"}, {"created_at", "sku"}),
}
REQUIRED_ENTITIES = {"customer", "order", "order_item", "product", "inventory", "shipment"}
CANONICAL_ORDER_STATUSES = {"processing", "shipping", "delivered", "cancelled"}

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_SQL_NAME = re.compile(rf"^{_IDENT}(\.{_IDENT})?$")
_ALLOWED_FUNCS = {"COALESCE", "CONCAT"}


class MappingError(ValueError):
    """The mapping file is invalid."""


# --- expression AST -----------------------------------------------------------------


@dataclass(frozen=True)
class Col:
    path: str


@dataclass(frozen=True)
class Lit:
    value: str | int | float


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Expr, ...]


Expr = Col | Lit | Call


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        self._ws()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def parse(self) -> Expr:
        node = self._expr()
        if self._peek():
            raise MappingError(f"unexpected {self.text[self.pos :]!r} in expression {self.text!r}")
        return node

    def _expr(self) -> Expr:
        ch = self._peek()
        if ch == "'":
            return self._string()
        if ch.isdigit() or ch == "-":
            return self._number()
        if re.match(_IDENT, ch or " "):
            return self._ident_or_call()
        raise MappingError(f"cannot parse expression {self.text!r} at position {self.pos}")

    def _string(self) -> Lit:
        self.pos += 1
        out: list[str] = []
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "'":
                if self.text[self.pos + 1 : self.pos + 2] == "'":
                    out.append("'")
                    self.pos += 2
                    continue
                self.pos += 1
                return Lit("".join(out))
            out.append(ch)
            self.pos += 1
        raise MappingError(f"unterminated string in {self.text!r}")

    def _number(self) -> Lit:
        m = re.compile(r"-?\d+(\.\d+)?").match(self.text, self.pos)
        if not m:
            raise MappingError(f"bad number in {self.text!r}")
        self.pos = m.end()
        return Lit(float(m.group(0)) if m.group(1) else int(m.group(0)))

    def _ident_or_call(self) -> Expr:
        m = re.compile(rf"{_IDENT}(?:\.{_IDENT})*").match(self.text, self.pos)
        assert m
        name = m.group(0)
        self.pos = m.end()
        if self._peek() == "(":
            if name.upper() not in _ALLOWED_FUNCS:
                raise MappingError(
                    f"function {name!r} is not allowed (allowed: {sorted(_ALLOWED_FUNCS)})"
                )
            self.pos += 1
            args = [self._expr()]
            while self._peek() == ",":
                self.pos += 1
                args.append(self._expr())
            if self._peek() != ")":
                raise MappingError(f"missing ')' in {self.text!r}")
            self.pos += 1
            return Call(name.upper(), tuple(args))
        return Col(name)


def parse_expression(text: str) -> Expr:
    return _Parser(str(text)).parse()


def columns_in(expr: Expr) -> list[str]:
    """Every column/path referenced by an expression (used by validate-mapping)."""
    if isinstance(expr, Col):
        return [expr.path]
    if isinstance(expr, Call):
        return [c for a in expr.args for c in columns_in(a)]
    return []


# --- mapping models -----------------------------------------------------------------


class EntityMapping(BaseModel):
    table: str | None = None  # SQL table or view
    collection: str | None = None  # MongoDB collection
    unwind: str | None = None  # MongoDB: array path whose elements are the entity rows
    fields: dict[str, str]
    status_map: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("fields", mode="before")
    @classmethod
    def _stringify(cls, v: object) -> object:
        # YAML turns bare `true`/numbers into non-strings; keep the raw text for the parser.
        return {k: str(x) for k, x in v.items()} if isinstance(v, dict) else v

    @property
    def target(self) -> str:
        return (self.table or self.collection) or ""


class SchemaMapping(BaseModel):
    dialect: Literal["postgres", "mysql", "mongodb", "sqlite"]
    entities: dict[str, EntityMapping]

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> SchemaMapping:
        problems: list[str] = []
        for name in REQUIRED_ENTITIES - set(self.entities):
            problems.append(f"missing required entity {name!r}")
        for name, ent in self.entities.items():
            if name not in CANONICAL_FIELDS:
                problems.append(f"unknown entity {name!r}")
                continue
            required, optional = CANONICAL_FIELDS[name]
            for f in required - set(ent.fields):
                problems.append(f"{name}: missing required field {f!r}")
            for f in set(ent.fields) - required - optional:
                problems.append(f"{name}: unknown field {f!r}")
            problems += self._check_target(name, ent)
            for field_name, text in ent.fields.items():
                try:
                    expr = parse_expression(text)
                except MappingError as exc:
                    problems.append(f"{name}.{field_name}: {exc}")
                    continue
                if self.dialect != "mongodb":
                    for col in columns_in(expr):
                        if "." in col:
                            problems.append(
                                f"{name}.{field_name}: dotted column {col!r} is only valid for "
                                "MongoDB; expose joined data through a database view"
                            )
            if name == "order":
                bad = set(ent.status_map.values()) - CANONICAL_ORDER_STATUSES
                if bad:
                    problems.append(
                        f"order.status_map: unknown canonical status {sorted(bad)} "
                        f"(use {sorted(CANONICAL_ORDER_STATUSES)})"
                    )
        if problems:
            raise MappingError("invalid schema mapping:\n  - " + "\n  - ".join(problems))
        return self

    def _check_target(self, name: str, ent: EntityMapping) -> list[str]:
        if self.dialect == "mongodb":
            if not ent.collection:
                return [f"{name}: MongoDB entities need `collection`"]
            return []
        if not ent.table:
            return [f"{name}: SQL entities need `table`"]
        if not _SQL_NAME.match(ent.table):
            return [f"{name}: unsafe table name {ent.table!r}"]
        return []

    def entity(self, name: str) -> EntityMapping:
        try:
            return self.entities[name]
        except KeyError as exc:
            raise MappingError(f"entity {name!r} is not mapped") from exc

    def has(self, name: str) -> bool:
        return name in self.entities


def load_mapping(path: Path) -> SchemaMapping:
    if not path.exists():
        raise MappingError(f"schema mapping not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return SchemaMapping.model_validate(raw)
    except ValidationError as exc:  # pydantic wraps errors raised inside validators
        msgs = [str(e["msg"]).removeprefix("Value error, ") for e in exc.errors()]
        raise MappingError("\n".join(msgs)) from exc
