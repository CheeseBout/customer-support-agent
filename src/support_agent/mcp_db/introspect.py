"""Draft a schema mapping from a live database (`support-agent introspect-db`).

Writing `schema_mapping.yaml` by hand means knowing, for every canonical field, which of your
columns it is. This module reads the table and column names (and a few distinct status values),
guesses the match, and writes a draft the owner then reviews. Nothing here calls a model: the
guesses come from the name lists in `vocabulary.yaml` (English and Vietnamese), from foreign keys
when the database declares them, and from how the tables refer to each other.

Every guess is marked. A line the tool is sure about carries no comment; one it is not sure of
ends with `# check`; a required field it could not find is written as `TODO_...` so that
`validate-mapping` fails until someone fills it in. The draft is never trusted blindly: it is
loaded through the same validator as a hand-written mapping.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

import yaml

from support_agent.mcp_db.mapping import (
    CANONICAL_FIELDS,
    CANONICAL_ORDER_STATUSES,
    REQUIRED_ENTITIES,
)

Kind = Literal["text", "number", "date", "bool", "json", "other"]
Confidence = Literal["sure", "check"]
StatusLookup = Callable[[str, str], list[str]]  # (table, column) -> distinct values

SAMPLE_LIMIT = 50  # distinct values read from a status column
MONGO_SAMPLE_DOCS = 100
ENTITIES = ("customer", "order", "order_item", "product", "inventory", "shipment", "return_request")
# Entities claim tables in this order, so "order_items" is taken before "orders" can grab it.
CLAIM_ORDER = (
    "order_item",
    "return_request",
    "shipment",
    "inventory",
    "order",
    "product",
    "customer",
)
TABLE_PREFIXES = ("tbl_", "tb_", "wp_", "shop_", "app_", "t_", "dbo_")
NUMBER_FIELDS = {
    "total",
    "subtotal",
    "shipping_fee",
    "discount",
    "tax",
    "refunded_amount",
    "quantity",
    "unit_price",
    "price",
}
DATE_FIELDS = {"created_at", "delivered_at", "updated_at", "eta"}
TEXT_FIELDS = {"email", "phone", "name", "status", "carrier", "tracking_code"}
SURE = 70  # a column scoring at least this is not marked `# check`
ALWAYS_CHECK = {"group_id", "options"}  # fields whose wrong guess changes what customers see
MIN_SCORE = 30


@lru_cache
def vocabulary() -> dict[str, Any]:
    text = (resources.files(__package__) / "vocabulary.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


# --- what the database looks like -----------------------------------------------------------------


@dataclass
class Column:
    name: str
    kind: Kind = "text"
    fk: tuple[str, str] | None = None  # (referred table, referred column), when declared


@dataclass
class Table:
    name: str  # as written in the mapping (schema-qualified when needed)
    columns: list[Column]
    is_view: bool = False
    # MongoDB: arrays of sub-documents (path -> their fields), e.g. the lines of an order.
    arrays: dict[str, list[Column]] = field(default_factory=dict)

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


# --- text helpers --------------------------------------------------------------------------------


def fold(text: str) -> str:
    """Lower-case, no diacritics, words joined by single underscores: "Đã Giao" -> "da_giao"."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)  # camelCase -> camel_case
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def table_key(name: str) -> str:
    """ "public.Order_Items" -> "order_item"."""
    base = fold(name.rpartition(".")[2])
    for prefix in TABLE_PREFIXES:
        base = base.removeprefix(prefix)
    parts = [p for p in base.split("_") if p]
    if parts:
        parts[-1] = _singular(parts[-1])
    return "_".join(parts)


def _tokens(name: str) -> list[str]:
    return [t for t in fold(name).split("_") if t]


def status_for(entity: str, raw: str) -> str | None:
    """The canonical status a raw database value stands for, or None if it is not recognisable."""
    key = fold(raw)
    for canonical, words in vocabulary()["status_words"][entity].items():
        if any(w in key for w in words):
            return canonical
    return None


# --- scoring -------------------------------------------------------------------------------------


def table_score(entity: str, table: Table) -> int:
    """How well the table's name says it holds `entity`: 100 exact, 60 a prefix or suffix."""
    key = table_key(table.name)
    best = 0
    for word in vocabulary()["tables"][entity]:
        if key == word:
            return 100
        if key.endswith("_" + word) or key.startswith(word + "_"):
            best = 60
    return best


def column_score(entity: str, fld: str, column: Column) -> int:
    """How likely `column` is the canonical field `fld` of `entity` (0: not at all)."""
    words: list[str] = vocabulary()["fields"][entity][fld]
    qualifiers = set(vocabulary()["qualifiers"])
    name = fold(column.name.rpartition(".")[2])
    if name in words:
        score = 100 - words.index(name)
    else:
        tokens = _tokens(name)
        score = 0
        for index, word in enumerate(words):
            wanted = _tokens(word)
            if not all(t in tokens for t in wanted):
                continue
            extra = [t for t in tokens if t not in wanted]
            if qualifiers & set(extra) and not qualifiers & set(wanted):
                continue  # "refund_amount" is not "amount"
            score = max(score, 50 - 5 * len(extra) - index // 2)
    if score <= 0:
        return 0
    score -= 20 * column.name.count(".")  # a field of a sub-document is a worse fit than a top one
    if fld in NUMBER_FIELDS:
        score -= 0 if column.kind == "number" else 15 if column.kind == "text" else 40
    elif fld in DATE_FIELDS:
        score -= 0 if column.kind == "date" else 15 if column.kind == "text" else 40
    elif fld in TEXT_FIELDS:
        score -= 0 if column.kind in ("text", "other") else 30
    return score


# --- the proposal --------------------------------------------------------------------------------


@dataclass
class FieldGuess:
    column: str
    confidence: Confidence


@dataclass
class EntityGuess:
    entity: str
    table: str
    fields: dict[str, FieldGuess]
    confidence: Confidence = "sure"
    unwind: str | None = None  # MongoDB: the array holding the rows
    status_map: dict[str, str] = field(default_factory=dict)
    unmapped_statuses: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        return bool(self.missing_required or self.unmapped_statuses)


@dataclass
class Proposal:
    dialect: str
    currency: str
    entities: dict[str, EntityGuess] = field(default_factory=dict)
    missing_entities: list[str] = field(default_factory=list)  # required, nothing found
    unused_tables: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # worth knowing
    problems: list[str] = field(default_factory=list)  # the draft cannot work until fixed

    @property
    def complete(self) -> bool:
        """Everything required was found and nothing needs a human decision."""
        return (
            not self.missing_entities
            and not self.problems
            and not any(g.needs_attention for g in self.entities.values())
            and all(g.confidence == "sure" for g in self.entities.values())
        )


def _pick_fields(
    entity: str, columns: list[Column], *, forced: dict[str, str] | None = None
) -> dict[str, FieldGuess]:
    """Best column for each canonical field of `entity`; one column serves one field only."""
    required, _ = CANONICAL_FIELDS[entity]
    names = list(vocabulary()["fields"][entity])
    taken: set[str] = set()
    found: dict[str, FieldGuess] = {}
    for fld in [*(f for f in names if f in required), *(f for f in names if f not in required)]:
        if forced and fld in forced:
            found[fld] = FieldGuess(forced[fld], "sure")
            taken.add(forced[fld])
            continue
        scored = [(column_score(entity, fld, c), c) for c in columns if c.name not in taken]
        score, column = max(scored, key=lambda sc: sc[0], default=(0, None))
        if column is not None and score >= MIN_SCORE:
            # A wrong `group_id` would merge unrelated products, so a person always confirms it.
            sure = score >= SURE and fld not in ALWAYS_CHECK
            found[fld] = FieldGuess(column.name, "sure" if sure else "check")
            taken.add(column.name)
    return found


def _choose_tables(tables: list[Table]) -> dict[str, Table]:
    """One table per entity. Tables whose name says what they are are placed first, so a table
    recognised only by its columns can never take one that another entity is named after."""
    chosen: dict[str, Table] = {}
    used: set[str] = set()
    for by_name in (True, False):
        for entity in CLAIM_ORDER:
            if entity in chosen:
                continue
            table = _best_table(entity, tables, used, by_name=by_name)
            if table is not None:
                chosen[entity] = table
                used.add(table.name)
    return chosen


def _best_table(entity: str, tables: list[Table], used: set[str], *, by_name: bool) -> Table | None:
    required, _ = CANONICAL_FIELDS[entity]
    best: tuple[int, Table] | None = None
    for table in tables:
        if table.name in used:
            continue
        named = table_score(entity, table)
        found = _pick_fields(entity, table.columns)
        have = len(required & set(found))
        if by_name:
            if named == 0 or have == 0:
                continue
        elif named > 0 or have < len(required) or len(found) < max(3, len(required)):
            # Recognised by columns alone: it needs every required field and enough company
            # (a table with only an `id` could be anything).
            continue
        score = named + 15 * have
        if best is None or score > best[0]:
            best = (score, table)
    return best[1] if best else None


def _fk_target(column: Column | None, table: Table | None) -> str | None:
    """The column an FK on `column` points at, if it points into `table`."""
    if column is None or column.fk is None or table is None:
        return None
    ref_table, ref_column = column.fk
    return ref_column if table_key(ref_table) == table_key(table.name) else None


def _reference(column: Column | None, parent: Table, parent_id: str) -> str | None:
    """Which column of `parent` a child's `column` refers to: by FK, by name, or by convention."""
    if column is None:
        return None
    declared = _fk_target(column, parent)
    if declared is not None:
        return declared
    if parent.column(column.name) is not None:
        return column.name  # the same name on both sides: order_code, sku
    if column.name.endswith("_id") and parent.column(parent_id) is not None:
        return parent_id  # product_id -> products.id
    return None


def _align_references(
    chosen: dict[str, Table],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Make a parent's id column agree with what its children point at."""
    forced: dict[str, dict[str, str]] = {e: {} for e in chosen}
    notes: list[str] = []
    order, customer, product = chosen.get("order"), chosen.get("customer"), chosen.get("product")

    votes: dict[str, int] = {}
    if order is not None:
        for child in ("order_item", "shipment", "return_request"):
            table = chosen.get(child)
            guess = _pick_fields(child, table.columns).get("order_id") if table else None
            column = table.column(guess.column) if table and guess else None
            ref = _reference(column, order, "id")
            if ref is not None:
                weight = 3 if _fk_target(column, order) else 1
                votes[ref] = votes.get(ref, 0) + weight
    if votes:
        forced["order"]["id"] = max(votes, key=lambda k: votes[k])

    if order is not None and customer is not None:
        guess = _pick_fields("order", order.columns).get("customer_id")
        ref = _reference(order.column(guess.column), customer, "id") if guess else None
        if ref is not None:
            forced["customer"]["id"] = ref

    item = chosen.get("order_item")
    if item is not None and product is not None:
        guess = _pick_fields("order_item", item.columns).get("sku")
        ref = _reference(item.column(guess.column), product, "id") if guess else None
        own_sku = _pick_fields("product", product.columns).get("sku")
        if ref == "id" and own_sku is not None and own_sku.column != "id":
            notes.append(
                f"{item.name}.{guess.column if guess else '?'} holds {product.name}.id, not the "
                f"SKU in {product.name}.{own_sku.column}. A mapping has no joins: create a view "
                f"of {item.name} that adds the SKU, and point order_item at the view."
            )
        elif ref is not None:
            forced["product"]["sku"] = ref
    return forced, notes


def _fill_status_map(guess: EntityGuess, values: list[str]) -> None:
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for raw in values:
        target = status_for(guess.entity, raw)
        if target is None:
            unmapped.append(raw)
        else:
            mapping[raw] = target
    already_canonical = guess.entity != "order" or set(mapping) <= CANONICAL_ORDER_STATUSES
    if not unmapped and already_canonical and all(raw == to for raw, to in mapping.items()):
        return  # the database already uses the canonical words: no map needed
    guess.status_map = mapping
    guess.unmapped_statuses = unmapped


def propose(
    tables: list[Table],
    *,
    dialect: str,
    currency: str,
    status_values: StatusLookup | None = None,
) -> Proposal:
    """Match entities to tables, fields to columns, and status values to canonical statuses."""
    proposal = Proposal(dialect=dialect, currency=currency)
    chosen = _choose_tables(tables)
    forced, link_notes = _align_references(chosen)

    for entity in CLAIM_ORDER:
        table = chosen.get(entity)
        if table is None:
            if entity in REQUIRED_ENTITIES:
                proposal.missing_entities.append(entity)
            continue
        required, _ = CANONICAL_FIELDS[entity]
        fields = _pick_fields(entity, table.columns, forced=forced[entity])
        proposal.entities[entity] = EntityGuess(
            entity=entity,
            table=table.name,
            fields=fields,
            confidence="sure" if table_score(entity, table) >= 100 else "check",
            missing_required=sorted(required - set(fields)),
        )

    proposal.problems += link_notes
    _add_notes(proposal)
    _fill_statuses(proposal, status_values)
    claimed = {g.table for g in proposal.entities.values()}
    proposal.unused_tables = sorted(t.name for t in tables if t.name not in claimed)
    return proposal


def _add_notes(proposal: Proposal) -> None:
    order = proposal.entities.get("order")
    if order is None:
        return
    if "currency" not in order.fields:
        proposal.notes.append(
            f"The order table has no currency column: orders are labelled "
            f"{proposal.currency!r} (business_rules.currency)."
        )
    if "delivered_at" not in order.fields:
        proposal.notes.append(
            "No delivery date found on the order. Return windows count from `delivered_at` by "
            "default: map a column, or set business_rules.return.window_basis to created_at."
        )


def _fill_statuses(proposal: Proposal, status_values: StatusLookup | None) -> None:
    if status_values is None:
        return
    for name, guess in proposal.entities.items():
        status = guess.fields.get("status")
        if name in vocabulary()["status_words"] and status is not None:
            _fill_status_map(guess, status_values(guess.table, status.column))


# --- MongoDB: lines and shipment embedded in the order -------------------------------------------


def _embedded_items(parent: Table, order_id: str) -> EntityGuess | None:
    for path, sub in parent.arrays.items():
        fields = _pick_fields("order_item", sub)
        fields.pop("order_id", None)
        if {"sku", "quantity", "unit_price"} <= set(fields):
            guess = EntityGuess(
                entity="order_item",
                table=parent.name,
                unwind=path,
                fields={
                    "order_id": FieldGuess(order_id, "sure"),
                    **{
                        f: FieldGuess(f"{path}.{g.column}", g.confidence) for f, g in fields.items()
                    },
                },
                confidence="check",
            )
            required, _ = CANONICAL_FIELDS["order_item"]
            guess.missing_required = sorted(required - set(guess.fields))
            return guess
    return None


def _embedded_shipment(parent: Table, order_id: str) -> EntityGuess | None:
    for word in ("shipment", "shipping", "delivery", "tracking", "fulfillment"):
        prefix = word + "."
        sub = [
            Column(c.name[len(prefix) :], c.kind)
            for c in parent.columns
            if c.name.startswith(prefix)
        ]
        fields = _pick_fields("shipment", sub) if sub else {}
        fields.pop("order_id", None)
        if "status" in fields:
            return EntityGuess(
                entity="shipment",
                table=parent.name,
                fields={
                    "order_id": FieldGuess(order_id, "sure"),
                    **{f: FieldGuess(prefix + g.column, g.confidence) for f, g in fields.items()},
                },
                confidence="check",
            )
    return None


def propose_mongo(
    tables: list[Table], *, currency: str, status_values: StatusLookup | None = None
) -> Proposal:
    """Like `propose`, plus order lines embedded in the order and a shipment sub-document."""
    proposal = propose(tables, dialect="mongodb", currency=currency)
    order = proposal.entities.get("order")
    parent = next((t for t in tables if order and t.name == order.table), None)
    order_id = order.fields.get("id") if order else None
    if parent is not None and order_id is not None:
        if "order_item" not in proposal.entities and (
            items := _embedded_items(parent, order_id.column)
        ):
            proposal.entities["order_item"] = items
            if "order_item" in proposal.missing_entities:
                proposal.missing_entities.remove("order_item")
        if "shipment" not in proposal.entities and (
            ship := _embedded_shipment(parent, order_id.column)
        ):
            proposal.entities["shipment"] = ship
    _fill_statuses(proposal, status_values)
    claimed = {g.table for g in proposal.entities.values()}
    proposal.unused_tables = sorted(t.name for t in tables if t.name not in claimed)
    return proposal


# --- writing the draft ---------------------------------------------------------------------------


def _scalar(text: str) -> str:
    """A YAML scalar that always round-trips (quotes only when it has to)."""
    dumped = yaml.safe_dump(text, allow_unicode=True, width=10_000, default_flow_style=True)
    return dumped.strip().removesuffix("...").strip()


def render_yaml(proposal: Proposal) -> str:
    """The draft mapping as YAML text, with a comment on every line a human must check."""
    key = "collection" if proposal.dialect == "mongodb" else "table"
    out = [
        "# Draft schema mapping written by `support-agent introspect-db`.",
        "# Read it before you trust it: lines ending in `# check` are guesses, `TODO_...` marks a",
        "# required field that was not found. Then run `support-agent validate-mapping`.",
        f"dialect: {proposal.dialect}",
        "",
        "entities:",
    ]
    for entity in ENTITIES:
        guess = proposal.entities.get(entity)
        if guess is None:
            out += _missing(entity, key)
        else:
            out += _entity_lines(proposal, guess, key)
    return "\n".join(out).rstrip() + "\n"


def _missing(entity: str, key: str) -> list[str]:
    if entity not in REQUIRED_ENTITIES:
        return [f"  # {entity}: no matching {key} found; leave it out or add it by hand.", ""]
    required, _ = CANONICAL_FIELDS[entity]
    lines = [
        f"  {entity}:  # required, but no matching {key} was found",
        f"    {key}: TODO_{entity}",
        "    fields:",
    ]
    lines += [
        f"      {f}: TODO_{entity}_{f}" for f in vocabulary()["fields"][entity] if f in required
    ]
    return [*lines, ""]


def _entity_lines(proposal: Proposal, guess: EntityGuess, key: str) -> list[str]:
    required, _ = CANONICAL_FIELDS[guess.entity]
    embedded = guess.unwind is not None or any("." in g.column for g in guess.fields.values())
    guessed = guess.confidence == "check" and not embedded
    lines = [
        f"  {guess.entity}:"
        + ("  # check: matched by its columns, not its name" if guessed else "")
    ]
    lines.append(f"    {key}: {_scalar(guess.table)}")
    if guess.unwind:
        lines.append(f"    unwind: {guess.unwind}")
    lines.append("    fields:")
    for fld in vocabulary()["fields"][guess.entity]:
        found = guess.fields.get(fld)
        if found is not None:
            note = "  # check" if found.confidence == "check" else ""
            lines.append(f"      {fld}: {_scalar(found.column)}{note}")
        elif fld in required:
            lines.append(f"      {fld}: TODO_{guess.entity}_{fld}  # required; not found")
        elif fld == "currency" and guess.entity == "order":
            # A constant is a string with its own single quotes: "'VND'".
            lines.append(f"      currency: \"'{proposal.currency}'\"  # no column: a constant")
    if guess.status_map or guess.unmapped_statuses:
        lines.append("    status_map:  # the values found today; add any other your system can set")
        lines += [
            f"      {_scalar(raw)}: {canonical}" for raw, canonical in guess.status_map.items()
        ]
        lines += [
            f"      # {_scalar(raw)}: ???   # TODO: which status is this?"
            for raw in guess.unmapped_statuses
        ]
    return [*lines, ""]


TEMPLATE_MARKER = "This is a starting point, not a working file"


def is_shipped_template(path: Any) -> bool:
    """True for the untouched template that ships as config/schema_mapping.yaml."""
    try:
        return TEMPLATE_MARKER in path.read_text(encoding="utf-8")
    except OSError:
        return False


# --- reading a database --------------------------------------------------------------------------

_NUMERIC = (int, float, Decimal)


def kind_of_python_type(py_type: type) -> Kind:
    if issubclass(py_type, bool):
        return "bool"
    if issubclass(py_type, _NUMERIC):
        return "number"
    if issubclass(py_type, datetime | date):
        return "date"
    if issubclass(py_type, str):
        return "text"
    if issubclass(py_type, dict | list):
        return "json"
    return "other"


def kind_of_value(value: Any) -> Kind:
    if isinstance(value, bool):
        return "bool"
    if hasattr(value, "to_decimal") or isinstance(value, _NUMERIC):
        return "number"
    if isinstance(value, datetime | date):
        return "date"
    if isinstance(value, str):
        return "date" if re.match(r"^\d{4}-\d{2}-\d{2}([T ]|$)", value) else "text"
    if isinstance(value, dict | list):
        return "json"
    return "other"


def looks_like_status(name: str) -> bool:
    return any(w in fold(name) for w in ("status", "state", "trang_thai"))


async def read_sql(
    url: str, *, schema: str | None, dialect: str
) -> tuple[list[Table], StatusLookup]:
    """Table and column names, declared foreign keys, and the distinct values of status columns.

    Only metadata and `SELECT DISTINCT <status column>` are read: no customer data.
    """
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    qualify = dialect == "postgres"
    schema_name = schema or ("public" if qualify else None)
    engine = create_async_engine(url)

    def reflect(sync_conn: Any) -> list[Table]:
        inspector = sa.inspect(sync_conn)
        names = [(n, False) for n in inspector.get_table_names(schema=schema_name)]
        names += [(n, True) for n in inspector.get_view_names(schema=schema_name)]
        found: list[Table] = []
        for name, is_view in sorted(names):
            fks: dict[str, tuple[str, str]] = {}
            if not is_view:
                for fk in inspector.get_foreign_keys(name, schema=schema_name):
                    pairs = zip(fk["constrained_columns"], fk["referred_columns"], strict=False)
                    fks.update({local: (fk["referred_table"], remote) for local, remote in pairs})
            columns = []
            for col in inspector.get_columns(name, schema=schema_name):
                try:
                    kind = kind_of_python_type(col["type"].python_type)
                except NotImplementedError:
                    kind = "other"
                columns.append(Column(col["name"], kind, fks.get(col["name"])))
            full = f"{schema_name}.{name}" if qualify and schema_name else name
            found.append(Table(full, columns, is_view))
        return found

    values: dict[tuple[str, str], list[str]] = {}
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(reflect)
            for table in tables:
                base = table.name.rpartition(".")[2]
                for column in table.columns:
                    if not looks_like_status(column.name) or column.kind not in ("text", "number"):
                        continue
                    stmt = (
                        sa.select(sa.cast(sa.column(column.name), sa.String))
                        .select_from(sa.table(base, schema=schema_name if qualify else None))
                        .distinct()
                        .limit(SAMPLE_LIMIT)
                    )
                    try:
                        rows = (await conn.execute(stmt)).all()
                    except sa.exc.SQLAlchemyError:
                        await conn.rollback()
                        continue
                    found = sorted(str(r[0]) for r in rows if r[0] is not None)
                    values[(table.name, column.name)] = found
    finally:
        await engine.dispose()
    return tables, lambda table, column: values.get((table, column), [])


def _flatten(value: dict[str, Any], prefix: str, into: dict[str, Kind], depth: int = 0) -> None:
    for key, item in value.items():
        if key == "_id":
            continue
        if isinstance(item, dict) and depth < 3:
            _merge(into, f"{prefix}{key}", "json")  # the whole sub-document, e.g. product specs
            _flatten(item, f"{prefix}{key}.", into, depth + 1)
        else:
            _merge(into, f"{prefix}{key}", kind_of_value(item))


def _merge(into: dict[str, Kind], path: str, kind: Kind) -> None:
    old = into.get(path)
    into[path] = kind if old in (None, kind) else "text"


def tables_from_documents(collections: dict[str, list[dict[str, Any]]]) -> list[Table]:
    """Tables from sampled MongoDB documents: dotted field paths, and arrays of sub-documents."""
    tables = []
    for name, docs in collections.items():
        columns: dict[str, Kind] = {}
        arrays: dict[str, dict[str, Kind]] = {}
        for doc in docs:
            scalars = {k: v for k, v in doc.items() if not _is_line_list(v)}
            _flatten(scalars, "", columns)
            for path, items in _lines_of(doc, ""):
                for element in items:
                    _flatten(element, "", arrays.setdefault(path, {}))
        tables.append(
            Table(
                name,
                [Column(p, k) for p, k in columns.items()],
                arrays={p: [Column(c, k) for c, k in sub.items()] for p, sub in arrays.items()},
            )
        )
    return tables


def _is_line_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(x, dict) for x in value)


def _lines_of(doc: dict[str, Any], prefix: str) -> list[tuple[str, list[dict[str, Any]]]]:
    found = []
    for key, value in doc.items():
        if _is_line_list(value):
            found.append((f"{prefix}{key}", value))
        elif isinstance(value, dict) and prefix.count(".") < 2:
            found += _lines_of(value, f"{prefix}{key}.")
    return found


async def read_mongo(url: str) -> tuple[list[Table], StatusLookup]:
    from motor.motor_asyncio import AsyncIOMotorClient

    client: Any = AsyncIOMotorClient(url, tz_aware=True, serverSelectionTimeoutMS=5000)
    values: dict[tuple[str, str], list[str]] = {}
    try:
        db = client.get_default_database()
        samples: dict[str, list[dict[str, Any]]] = {}
        for name in sorted(await db.list_collection_names()):
            if not name.startswith("system."):
                cursor = db[name].find({}).limit(MONGO_SAMPLE_DOCS)
                samples[name] = await cursor.to_list(MONGO_SAMPLE_DOCS)
        tables = tables_from_documents(samples)
        for table in tables:
            for column in table.columns:
                if looks_like_status(column.name):
                    found = await db[table.name].distinct(column.name)
                    values[(table.name, column.name)] = sorted(
                        str(v) for v in found if v is not None
                    )[:SAMPLE_LIMIT]
    finally:
        client.close()
    return tables, lambda table, column: values.get((table, column), [])
