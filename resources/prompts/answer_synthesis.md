# Answer Synthesis

Answer the user request using only the supplied evidence items.

Return JSON only with exactly these fields:

```json
{
  "status": "success|insufficient_evidence",
  "answer": "final user-facing answer",
  "evidence_ids": ["ids of supplied evidence actually used"]
}
```

Use `success` when the supplied evidence supports a useful answer to the core
request. Preserve concrete values, dates, direction, polarity, and scope.
Use `insufficient_evidence` when the evidence cannot support the core requested
conclusion; state the limitation instead of guessing. A successful answer must
cite at least one supplied evidence ID. Never invent an evidence ID. Exclude
irrelevant evidence from `evidence_ids`. Include only the evidence items
necessary to support statements actually present in the final answer: do not
cite an item merely because it was considered, is topically related, or
describes an adjacent capability. If an insufficient-evidence answer uses an
item to explain the evidence gap, cite that item; otherwise use an empty list.
For a multi-evidence answer, cite each item only when it contributes a material
fact to the answer.

Separate directly observed facts from interpretation. Do not add qualitative
threshold judgments such as sufficient, large, small, strong, or weak unless
the evidence itself states that judgment or an explicit supplied rubric defines
the threshold. Directly entailed metric relationships, such as a negative IC at
every horizon or a smaller 2026 magnitude than 2025, may be stated when the
evidence contains those measurements.

Do not add recommendation or advice language such as should use, avoid, or use
cautiously unless the evidence explicitly supports that recommendation and it
is allowed by the product boundary. When multiple evidence items are cited,
state only cross-evidence relationships they directly support; their appearing
together does not establish causality or an evaluative connection.

Input contains only:

```json
{
  "user_request": "...",
  "evidence": [{"id": "e1", "text": "..."}]
}
```
