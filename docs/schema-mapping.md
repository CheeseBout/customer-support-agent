# Writing `config/schema_mapping.yaml`

The agent never writes SQL against your database. It works with seven **canonical entities**,
and this file tells it where each one lives in *your* schema. Change only the right-hand sides.
Run `support-agent validate-mapping` after every edit.

## Entities and fields

| Entity | Required fields | Optional fields |
|---|---|---|
| `customer` | `id` | `email`, `phone`, `name` |
| `order` | `id`, `customer_id`, `status`, `total`, `created_at` | `currency`, `delivered_at`, `shipping_address`, `payment_method` |
| `order_item` | `order_id`, `sku`, `quantity`, `unit_price` | `product_name` |
| `product` | `sku`, `name`, `price` | `description`, `category`, `attributes`, `active` |
| `inventory` | `sku`, `quantity` | |
| `shipment` | `order_id`, `status` | `carrier`, `tracking_code`, `updated_at`, `eta` |
| `return_request` (optional entity) | `id`, `order_id`, `status` | `created_at`, `sku` |

`return_request` lets the return rule see requests that already exist for an order. Without it,
the "already requested" check cannot fire.

## SQL (PostgreSQL, MySQL, SQLite)

```yaml
dialect: postgres            # postgres | mysql | sqlite; must equal BUSINESS_DB_TYPE
entities:
  order:
    table: public.orders     # a table or a VIEW; optional schema prefix
    fields:
      id: order_code         # canonical field: your column
      customer_id: customer_id
      status: status
      total: grand_total
      currency: "'VND'"      # a constant (note the inner single quotes)
      created_at: created_at
    status_map:              # your status values -> pending|processing|shipping|delivered|cancelled
      "DA_GIAO": delivered
```

### What a field may contain

Only these forms are accepted; anything else is rejected when the file loads:

| Form | Example |
|---|---|
| a column | `order_code` |
| a string or number constant | `"'VND'"`, `42` |
| `COALESCE(a, b, ...)` | `COALESCE(display_name, title)` |
| `CONCAT(a, b, ...)` | `CONCAT(first_name, ' ', last_name)` |

There are no joins and no free-form expressions. If the data you need is spread over several
tables, create a **database view** that exposes the columns in one place and point `table` at it.
That keeps the SQL under your control and reviewable by your DBA.

### Statuses

`order.status_map` is required if your status values are not already the canonical ones
(`processing`, `shipping`, `delivered`, `cancelled`). Every distinct status currently present in
the `orders` table must appear in the map: `validate-mapping` fails otherwise, because an unmapped
status would silently make orders non-returnable. `shipment` and `return_request` maps are
optional; unmapped values pass through lower-cased and produce a warning.

### Good to know

* **Ids compare as text.** `order_code` may be an integer column or a string column; the user's
  `#1234` matches either. For very large tables, expose a view with an indexed text column.
* **Timestamps.** Columns without a time zone are read as the business time zone
  (`business_rules.timezone`, default `Asia/Ho_Chi_Minh`); columns with one are respected.
* **Money** is returned as plain numbers (VND has no minor unit).
* **`attributes`** may be a JSON column or JSON text; it is parsed for product comparison.
* **`active`** accepts booleans, `0/1` and `"true"/"false"`. Missing means active.
* **Views and permissions.** Giving the agent's account access to views only (and none to the base tables) is the
  recommended setup. On PostgreSQL `GRANT USAGE` on the schema and `SELECT` on the views is enough. On MySQL
  `validate-mapping` also needs `SHOW VIEW` on each view (it reveals the view definition, never data).
* Queries are limited to `db.max_rows` rows and `db.query_timeout_seconds`, are fully
  parameterised, and run on a read-only session where the database supports it. Still give the
  agent a database account that has **only `SELECT`** (see `scripts/sql/`).

## MongoDB

```yaml
dialect: mongodb
entities:
  order:
    collection: orders
    fields: {id: order_code, customer_id: customer_id, status: status, total: grand_total}
  order_item:
    collection: orders       # order lines are embedded in the order document
    unwind: items            # one row per element of `items`
    fields: {order_id: order_code, sku: items.sku, quantity: items.qty, unit_price: items.price}
  shipment:
    collection: orders       # shipment is a sub-document
    fields: {order_id: order_code, status: shipment.status, carrier: shipment.carrier}
```

Fields are dotted paths. `BUSINESS_DB_URL` must include the database name
(`mongodb://host:27017/shop`). See `config/examples/schema_mapping.mongodb.yaml`.

Ids typed by a customer are matched as text and as a number, so a string code (`"TM-48213"`) or an integer
works. An `ObjectId` cannot be matched: keep (or add) a string order number and customer code in the documents
and map those.

## Checking your mapping

```
support-agent validate-mapping
```

It verifies that every table/view/collection and column exists, that numeric and date fields have
compatible types, that required entities are present, and that `status_map` covers the statuses in
the database. Errors exit non-zero, so you can run it in CI against a staging database.
