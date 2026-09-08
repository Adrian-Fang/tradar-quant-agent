# Research knowledge records

This directory defines the boundary for future research knowledge artifacts. It does not contain Slack transcripts, private history, or a retrieval implementation.

Future records should be structured, provenance-aware summaries of research work. Each record should support at least:

- `research_id`
- `source_task_id`
- `date`
- `topic`
- `status`
- `question`
- `method`
- `findings`
- `conclusion`
- `artifacts`
- `supersedes`

At minimum, `status` should distinguish `exploratory`, `promising`, `validated`, `rejected`, `inconclusive`, and `superseded`. `artifacts` should point to reproducible result locations or identifiers, and `supersedes` should make replacement of stale conclusions explicit.
