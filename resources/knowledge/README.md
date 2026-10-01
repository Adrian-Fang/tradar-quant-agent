# Research knowledge records

Research Markdown records in `research/` are canonical, provenance-aware summaries,
not private transcripts or workflow history. The AE-14 backend builds a derived,
rebuildable Qdrant index; it does not modify these records or auto-index at runtime.
The exact versioned record contract, legacy adapter, chunks, index profile and
operational commands are documented in [the retrieval contract](../retrieval/README.md).

New records use `schema_version: "1.0"`. Each record supports:

- `research_id`
- `source_type` (a neutral descriptor, such as `research_record`)
- `source_ref` (public references; may be empty)
- `source_task_id` (optional public identifier; may be null)
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

Do not publish private channel/thread IDs, message timestamps, conversation URLs
or internal task names. Unknown provenance stays unknown: omit task IDs and use
empty source refs rather than inventing a source. Keep a nonempty Provenance
section explaining available references and limitations. Artifact identifiers
are retained for traceability; they do not guarantee that the corresponding
scripts or raw results are included in this repository.
