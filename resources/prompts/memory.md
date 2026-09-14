# Memory Decision

Decide whether the message contains durable information worth saving across
future tasks. Return JSON only with exactly this shape:

```json
{
  "action": "write|update|ignore",
  "memory": "normalized durable memory" or null,
  "supersedes_id": "existing memory id" or null,
  "reason": "short explanation"
}
```

Use `write` for a new, clear, stable preference, fact, or workflow constraint
with reuse value across tasks when no equivalent memory is supplied. Use
`update` only when the message clearly corrects or replaces an existing memory
on the same topic. The new memory must supersede the cited existing memory ID.
Use `ignore` for temporary task or session instructions, runtime state,
one-time results, casual conversation, duplicates, hedged or speculative
statements, and any message that says not to remember it.

Treat each memory record as one atomic durable proposition, preference, or
constraint that can be independently updated or made obsolete. If new durable
information is related to an existing memory but can coexist with it as a
separate requirement, choose `write` with `supersedes_id` set to null; do not
update merely to merge information from the same topic. Use `update` only when
the new information makes a particular existing memory old, wrong, replaced,
or no longer true.

Do not infer facts or preferences that are not stated. A current message has
priority over an old memory, but a change must be explicit before using
`update`; otherwise use `ignore` for an equivalent repetition.

Contract rules:

- `write`: `memory` is a non-empty string and `supersedes_id` is null.
- `update`: `memory` is a non-empty string and `supersedes_id` is an ID from
  the supplied `existing_memories`.
- `ignore`: `memory` and `supersedes_id` are both null.
- For an `unverifiable` or tentative message, do not invent a durable memory.
- Keep `reason` observational; do not add fields.

Input contains only the message and existing memories:

```json
{
  "message": "...",
  "existing_memories": [{"id": "m1", "text": "..."}]
}
```
