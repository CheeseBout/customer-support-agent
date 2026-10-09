---
name: track-order
description: Find where an order is and when it will arrive
lang: en
---

# Tracking an order

1. Get the order id from the customer. If they have none, call `list_orders` (use `status` to narrow it, for example `shipping`) and let them pick one.
2. Call `get_order` and `get_shipment_status` for that id together.
3. Tell them, in this order: the order status, the carrier and tracking code, the shipment status, and the estimated arrival date if there is one.

If `shipments` lists several parcels, the order was split: go through them one by one (carrier, tracking code, status) and say whether all have arrived (`all_delivered`).

If there is no shipment yet:
- `processing`: the shop has not handed the parcel to a carrier yet. Say so, and use `search_policy` if the customer asks how they will be told when it ships.
- `pending_payment`: the order is waiting for payment and will not ship until it is paid.
- `on_hold`: the shop has paused the order; do not guess why, suggest contacting support.
- `cancelled`: the order was cancelled; nothing will ship.
- `returned` / `refunded`: the goods came back or the money was returned; nothing more will ship.

If the status is `partially_shipped`, part of the order has shipped: say which parts the data shows and that the rest follows.

If the order is not found, say you could not find it in their account and ask them to check the id. Do not guess which order they mean.

If it is late: the policy says when a late parcel can be reported to support. Use `search_policy` to quote the exact rule rather than recalling it.
