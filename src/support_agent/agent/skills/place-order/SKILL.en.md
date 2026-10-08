---
name: place-order
description: Help a customer build and place a new order
lang: en
---

# Placing an order

1. Find the products: use `search_products` or `compare_products` if the customer is still choosing, and `check_stock` for availability. Use the SKUs the tools return; never invent one.
2. Collect what an order needs, asking only for what is missing (if the customer says "my old address", read it from their previous order and use it (the confirmation shows it and they can edit it); never assume the payment method, and never switch it to get round a limit): the products and quantities, the full delivery address, and the payment method (cash on delivery, bank transfer, card, MoMo, ZaloPay or VNPay).
3. Call `prepare_order_draft` to check stock and get the total. Prices and totals come only from this tool, not from the customer or from you. If it fails, explain why in plain words:
   - `OUT_OF_STOCK`: say which product, and offer alternatives or a smaller quantity.
   - `NOT_ELIGIBLE` for cash on delivery: cash on delivery is capped (5,000,000 VND); offer another payment method.
   - Limits: at most 20 of one product and 10 different products per order.
4. Show the customer the lines, the total and the address, then call `propose_draft` with `draft_type` "order", the same items, address and payment method.
5. The customer confirms. Only after `propose_draft` reports it was created, tell them the request id and that the order is waiting for staff to confirm it. It is not an order until then: do not say it has been placed or will ship on a date.

If the customer wants to change the address or payment method, they can edit it during the confirmation.
