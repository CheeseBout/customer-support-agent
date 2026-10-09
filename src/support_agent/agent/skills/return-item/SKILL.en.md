---
name: return-item
description: Check whether an order can be returned or exchanged and start a return request
lang: en
requires: request:return
---

# Returning or exchanging an item

1. Get the order id, and the item if the order has several.
2. Call `check_return_eligibility`. Its verdict is final: never decide eligibility yourself.
3. Explain the verdict:
   - Eligible: say how many days are left and the deadline, and the amount that would come back.
   - Not eligible: give the reason in plain words. If the result lists `reason_windows` that are `still_open`, the answer depends on why they are returning it (for example a faulty item may have longer): say so and ask for the reason, then call `check_return_eligibility` again with `reason`. The reason codes mean: `WINDOW_EXPIRED` the return period has ended, `STATUS_NOT_ALLOWED` the order is not delivered yet, `CATEGORY_EXCLUDED` this kind of product cannot be returned (for example gift cards), `ALREADY_REQUESTED` a request for this item is already open, `ITEM_NOT_FOUND` that product is not on the order.
4. If it is eligible and the customer wants to go ahead, ask for the reason (defective, wrong item, not as described, damaged in transit, changed mind, other) and a short description. Then call `propose_draft` with `draft_type` "return". For a refund of the money, use the `request-refund` skill instead.
5. The customer is then asked to confirm. Do not say the request was sent until `propose_draft` reports it was created. After that, give them the request id and say staff will review it (use `search_policy` if they ask how long that takes).

Use `search_policy` for the exact wording before promising anything about shipping costs or exchanges.
