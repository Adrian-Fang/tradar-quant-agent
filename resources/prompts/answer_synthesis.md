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

Input contains only:

```json
{
  "user_request": "...",
  "evidence": [{"id": "e1", "text": "..."}]
}
```
