---
name: request-refund
description: Ask for money back on a returned or defective item
lang: en
---

# Requesting a refund

1. Get the order id, the item, and why. The reason must be the customer's own: if they only said they want a refund, ask why and do not call `propose_draft` with a generic reason. Check eligibility with `check_return_eligibility` exactly as in the `return-item` skill; the rules for returns and refunds are the same.
2. Never state a refund amount yourself. The amount comes from the rules engine (`refundable_amount`) and is shown to the customer when they confirm. You may quote it after the tool returns it.
3. Call `propose_draft` with `draft_type` "refund", the order id, the reason code and a short reason. Pass `sku` or `items` only if the customer wants part of the order refunded.
4. The customer confirms. Only when `propose_draft` reports the draft was created, tell them: the request id, that it is waiting for staff review, and that money is returned within 5 to 7 business days after it is approved and the item is received (use `search_policy` to confirm the current wording).

Refunds above 500,000 VND are reviewed with priority by a senior staff member: you may mention this if the amount is large, but never promise approval. A refund request is not a refund until staff approve it.

If the customer declines the confirmation, accept it and do not push. If they want to change the reason, they can edit it in the confirmation step.
