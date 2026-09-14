# Memory Recall

Select the active memories that materially apply to the current user request.
Return JSON only with exactly this shape:

```json
{
  "selected_ids": ["m1", "m2"],
  "reason": "short explanation"
}
```

Select only supplied memory IDs. Return each selected ID at most once and keep
the order in which the memories were supplied. Return an empty list when no
memory is materially applicable.

Recall relevance is separate from precedence. A current request may explicitly
override a persistent preference and that memory can still be relevant; a
later context-selection step decides which instruction wins. Do not resolve
conflicts, infer new preferences, or select a memory merely because it shares
a topic. Multiple independent memories may be selected when each affects the
request.

Input contains only the current request and active memories:

```json
{
  "user_request": "...",
  "active_memories": [{"id": "m1", "text": "..."}]
}
```
