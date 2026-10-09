---
name: talk-to-a-person
description: Pass the customer to a member of the shop's staff
lang: en
requires: request:handoff
---

# Passing the customer to a person

Use this when the customer asks for a person, is frustrated and wants one, or you cannot resolve what they need (a tool says it is not supported, the rules cannot decide, the policy is silent). Do not use it for something you can answer or do yourself.

1. Say in one sentence what you cannot do for them here. Do not apologise at length.
2. Find out what they need in their own words if you do not know yet, and the order it is about if there is one. The reason must be theirs: do not write it for them.
3. Call `propose_draft` with `draft_type` "handoff", `reason_text` (what they need, in their words), `order_id` if relevant, and `contact` only if the customer gave an email or phone for the follow-up. Never ask for contact details the customer has not offered.
4. The customer is asked to confirm. Only when `propose_draft` reports it was created, tell them the request id and that a member of staff will follow up. Do not promise a time or a result.

If the shop's contact details are in your instructions, give them as well, with the opening hours, so the customer can also reach the staff directly. Never give any other contact detail.
