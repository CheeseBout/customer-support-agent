---
name: warranty-claim
description: Check warranty cover for a product and start a warranty request
lang: en
---

# Warranty claims

1. Get the order id, the product (SKU) and what is wrong with it. If the fault is in a part (battery, screen, charging cable), the claim is for the product that contains it: take its SKU from `get_order` and do not search the catalogue for the part. If the customer has not said what is wrong, ask before proposing anything.
2. Call `check_warranty_eligibility` with the order id and SKU. Its verdict is final.
3. Explain it: the warranty period for this kind of product (12 months by default, 6 for accessories, 24 for appliances), when cover ends, and how long is left. Reason codes: `WARRANTY_EXPIRED` cover has ended, `NOT_DELIVERED` the warranty starts on delivery, `ITEM_NOT_FOUND` that product is not on the order.
4. If it is covered and the customer wants to continue, call `propose_draft` with `draft_type` "warranty", the order id, the SKU and `issue_description` in the customer's words.
5. The customer confirms. Only after `propose_draft` reports it was created, give them the request id and say staff will explain how to send or bring the product to a service center.

What warranty does not cover (physical damage, liquid damage, repairs by others, normal wear) is in the policy: use `search_policy` and read it out accurately if the customer describes such damage. Repairs usually take 7 to 15 business days; never promise a replacement, because that is decided after inspection.
