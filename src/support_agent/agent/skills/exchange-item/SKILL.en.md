---
name: exchange-item
description: Check whether an item can be exchanged (another size, colour or a new unit) and start an exchange request
lang: en
requires: request:exchange
---

# Exchanging an item

An exchange swaps an item the customer received for another one: a different size or colour of the same product, or a new unit of a faulty one. It is NOT a return. It has its own deadline and its own rules, so never use `check_return_eligibility` or the return window to judge it.

1. Get the order id and the item (read its SKU from `get_order`). If the customer has already said why (it does not fit, it is faulty, it is the wrong item), keep it and pass it as `reason` from the first call. A size or colour that does not suit is `changed_mind`.
2. Call `check_exchange_eligibility` with the order id, the item `sku` and the `reason` if known. Its verdict is final: never decide eligibility yourself.
3. Explain the verdict:
   - Eligible: say the deadline and how many days are left.
   - Not eligible: give the reason in plain words. The reason codes mean: `WINDOW_EXPIRED` the exchange period has ended, `STATUS_NOT_ALLOWED` the order is not delivered yet, `CATEGORY_EXCLUDED` this kind of product cannot be exchanged, `EXCHANGE_LIMIT_REACHED` the order has used all the exchanges it is allowed, `ALREADY_REQUESTED` the item is already part of an open request, `ITEM_NOT_FOUND` that product is not on the order. If the result lists `reason_windows` that are `still_open`, the answer depends on why they want the exchange (for example a fault may have longer): say so, ask for the reason, then call again with `reason`.
4. If it is eligible, find out what they want instead if they have not said (the size or colour). Use `check_stock` or `search_products` with the product name to find the SKU of that variant and whether it is in stock, then call `check_exchange_eligibility` again with `new_sku` to confirm it can be had. If it is out of stock or not the same product, say so and offer another size or colour.
5. Call `propose_draft` with `draft_type` "exchange", the order id, `sku`, `new_sku`, `reason_code` and a short `reason_text` in the customer's own words.
6. The customer is asked to confirm. Only when `propose_draft` reports it was created, give them the request id and say staff will review it. Never promise it will be approved. Use `search_policy` for fees, who pays the shipping and how the new item is sent; do not state any of that from memory.
