---
name: track-order
description: Find where an order is and when it will arrive
lang: en
---

# Tracking an order

1. Get the order id from the customer. If they have none, call `list_orders` (use `status` to narrow it, for example `shipping`) and let them pick one.
2. Call `get_order` and `get_shipment_status` for that id together.
3. Tell them, in this order: the order status, the carrier and tracking code, the shipment status, and the estimated arrival date if there is one.

If there is no shipment yet:
- `processing`: the shop has not handed the parcel to a carrier yet. Say so, and that a tracking code is sent by email and SMS once it ships.
- `cancelled`: the order was cancelled; nothing will ship.

If the order is not found, say you could not find it in their account and ask them to check the id. Do not guess which order they mean.

If it is late: the policy allows contacting support when the parcel is 3 business days past its estimated date. Use `search_policy` to quote the exact rule rather than recalling it.
