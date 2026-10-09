# Demo shop

A made-up electronics shop (phones, laptops, accessories, gift cards) in Vietnamese and English.
Everything specific to it lives here, so the rest of the repository stays free of one shop's data.

| Path | What it is |
|---|---|
| `app.yaml` | The demo's configuration. It extends `config/app.yaml` and sets this shop's business rules and paths |
| `knowledge/` | Sample policy documents: returns, shipping, warranty, payment (`.md` and `.docx`) |
| `config/schema_mapping.<db>.yaml` | The mapping for the demo database on PostgreSQL, MySQL, SQLite and MongoDB. Created by `support-agent seed-demo` |
| `evals/datasets/` | `baseline.jsonl` (79 questions) and `aftersales.jsonl` (63 requests) |
| `evals/reports/` | The evaluation results quoted in the README |

## Run it

From the repository root, put these in `.env`:

```
APP_CONFIG_PATH=./examples/demo-shop/app.yaml
BUSINESS_DB_TYPE=postgres
BUSINESS_DB_URL=postgresql://support_ro:support_ro@localhost:5432/shop
```

Then `support-agent seed-demo --url <owner url>`, `support-agent ingest` and
`support-agent ask "..." --user u_100`. For MySQL, SQLite or MongoDB, change `BUSINESS_DB_TYPE`
and the `mapping.path` in `app.yaml` to the matching file in `config/`.

Never point `seed-demo` at a real database: it drops and recreates the demo tables.

## Use it as a model for your own shop

Copy this folder, replace the documents and the mapping, and change `app.yaml`. Keep the
`extends:` line so you only list what differs from the shared defaults.
