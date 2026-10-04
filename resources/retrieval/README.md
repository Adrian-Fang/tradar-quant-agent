# AE-14 research knowledge backend

Markdown under `resources/knowledge/research/*.md` is canonical. Qdrant contains
derived chunks, never market data. No indexing happens on Agent startup.
`run_agent()` supports opt-in Qdrant retrieval (dense by default, hybrid optional)
alongside legacy semantic retrieval. No LangChain, LlamaIndex,
FastEmbed, PyTorch, custom BM25 tokenizer, or model downloads are added.

## Research record v1

Frontmatter is parsed with PyYAML's safe loader, with duplicate keys rejected.
Only the fields below are supported; unknown fields and unsupported YAML shapes
(sets, nested mappings in string arrays, timestamp datetimes) abort the scan.
`agent/retrieval/loader.py` defines the validation contract:

| Field | Shape |
| --- | --- |
| schema_version | string `"1.0"` |
| research_id | nonempty string, unique in corpus, no slash/newline |
| date | date or ISO date string, normalized to `YYYY-MM-DD` |
| market, topic, strategy, source_type | nonempty strings |
| status | exploratory / promising / validated / rejected / inconclusive / superseded |
| source_task_id | optional nonempty public string or null; absent normalizes to null |
| title | optional nonempty string; topic is the legacy title |
| tags, artifacts, supersedes | arrays of nonempty strings (may be empty) |
| source_ref | array of nonempty public references; absent normalizes to []; scalar string becomes a one-item array |

Required nonempty H1 sections: Research Question, Method, Key Findings,
Conclusion, Caveats, Provenance. Duplicate/unknown H1 headings, unclosed fences,
symlink records, invalid frontmatter and duplicate IDs abort the entire scan.
H2/H3 headings and fenced content stay inside their containing section.
Supersession self-references and cycles among current records abort the scan;
references to deleted historical IDs remain valid.

Unversioned current records use the documented legacy adapter: schema version
is normalized to `1.0`, title falls back to topic, absent `source_task_id` remains
**null**, and scalar refs become arrays. These optional provenance rules also
apply to versioned records. No research facts or provenance are invented.
Private messaging/workflow identifiers are excluded from public records; unknown
sources use empty refs and a disclosure in Provenance. Artifact paths may name
experiments not distributed with this repository, rather than available evidence.

## Chunk / point / profile contracts

One semantic chunk per group: `question_method`, `findings`,
`conclusion_caveats`. Logical ID = `research_id/group/1`; UUIDv5 uses the URL
namespace and `tradar/<resolved-profile-sha256>/<logical-id>`. Text/paths are not
part of logical identity. Content edits keep point IDs; changing the index
profile/model revision yields a separate collection/ID namespace.

Tables and fenced blocks are preserved intact. Whole section blocks are not
arbitrarily sliced. A 24,000-byte embedding-input guard rejects oversized blocks;
future long blocks need explicit semantic subdivision. Ollama truncation is
disabled, so exceeding the model's token limit fails rather than losing content.
Provenance is metadata and full-record context, not a standalone search chunk.

`profile.json` defines named vectors `dense` (1024, cosine) and `bm25` (sparse,
IDF). Embedding inputs contain research ID, title, tags and section text. Qwen
query inputs use the versioned `Instruct: ...\nQuery: ...` format. The model's
actual Ollama digest replaces the revision placeholder before hashing the
profile. Every embedding is dimension/finite/nonzero validated.

Physical collections are `tradar_research_<profile-hash-prefix>` (16 hex digits);
rebuilds append a fresh 12-hex suffix. Alias: `tradar_research`.

Each payload includes research/chunk ID, relative source path, title, market,
topic, status, tags, strategy, record date (UTC midnight), section group/names,
part number, exact section text, source/chunk/embedding-input SHA256, schema /
chunker / profile versions and profile hash, embedding model and actual digest,
indexed_at, Git HEAD revision, source refs/task/artifacts/supersedes and provenance.
`repo_revision` identifies HEAD; source hashes identify uncommitted record bytes.
`is_superseded` is true for explicit status or a current replacement record's
`supersedes` reference. Payload indexes: research_id, market, topic, status, tags
(keyword), date (datetime), is_superseded (boolean).

## SDK and native sparse inference

Dependencies: official `qdrant-client==1.19.0`, REST, for the selected 1.19.1 server,
and `pydantic==2.13.4` for the SDK's `model_validate` / `model_dump` calls. The SDK
also supports Pydantic v1; this repo deliberately pins v2 rather than maintaining
two serialization paths.
The generated SDK REST points API sends `Document(text, model="qdrant/bm25")`
directly to the server. It bypasses the SDK's optional client-side inference
machinery, so no FastEmbed is required. BM25 options remain server defaults.
Query uses the generated SDK **search API** (`http.search_api.query_points`),
not the points API, with `QueryRequest` and `RrfQuery(rrf=Rrf(k=60))`. See the pinned
[SDK models](https://github.com/qdrant/qdrant-client/blob/v1.19.0/qdrant_client/http/models/models.py)
and [generated search API](https://github.com/qdrant/qdrant-client/blob/v1.19.0/qdrant_client/http/api/search_api.py).

Ingestion checks server version, verifies stored dense and native sparse vectors,
point identities/hashes, canonical filter metadata and total count. Sparse vectors
must contain nonempty, unique sorted integer indices and finite positive values.
Independent dense, native BM25 and hybrid RRF(k=60) queries must each retrieve the
selected canonical probe point **before publishing the alias**. Merely accepting
a query or returning an empty result is not a successful proof. The probe filters
to one real record and includes superseded records. Unsupported inference/fusion fails
closed and leaves a dirty manifest. No fallback tokenizer/fusion silently changes
the baseline. A real live pass is still required to approve this backend.

## Operations and recovery

```bash
python -m pip install -r requirements.txt
# Inspect first; only if absent, manually run (never done by ingestion):
ollama list
ollama pull qwen3-embedding:0.6b
python -m agent.retrieval.ingest status
python -m agent.retrieval.ingest sync
python -m agent.retrieval.ingest rebuild
```

Endpoints default to `QDRANT_URL=http://127.0.0.1:6333` and the existing
`OLLAMA_BASE_URL=http://localhost:11434`. A changed model digest or profile
requires explicit rebuild. Status does not embed or mutate Qdrant.

Derived state: ignored `.runtime/knowledge_index/manifest.json`, state_version=1,
ready, resolved profile/hash, collection, authoritative corpus root, per-record
source/path/chunk hashes and point IDs, known-point journal and indexed_at.
Atomic file replacement and a local flock serialize writes against reads.

Unchanged records cause **no embedding or Qdrant writes**. Changed section inputs
are embedded; unchanged dense vectors are reused. All chunks of a changed source
receive its new source hash/provenance. Supersession metadata changes also update
affected old records without re-embedding. Missing records delete only known
point IDs after a successful authoritative scan. Missing directories, malformed
records and unreadable files are not interpreted as deletion.
An empty canonical corpus is rejected by both sync and rebuild **before embedding
or Qdrant mutations**: there is no real record to prove retrieval. Removing the
last record therefore preserves old points/alias, but retrieval fails closed on
the canonical/index mismatch. Empty-index publication/cleanup is not supported.

Before remote mutation, state becomes dirty and journals every possible point ID.
Partial failures cannot serve results; retry sync re-upserts the complete desired
snapshot and removes known orphan IDs. Rebuild uses a new collection, validates
it and rechecks canonical sources before atomic alias switching. Old collections
are retained for manual inspection/rollback; no automatic destructive cleanup.
A corrupted manifest can be replaced by explicit rebuild. Lost state or an
externally changed alias requires explicit rebuild/reconciliation, not broad
point deletion. Local state and Qdrant cannot share a transaction: if the process
stops between alias and manifest publication, retrieval remains fail closed and
sync reconciles. A failed rebuild currently blocks local reads until recovery.

After a corpus privacy cleanup, run sync to refresh active point payloads; stale
source hashes block retrieval until this completes. Rebuild retains old
collections, so neither rebuild nor a worktree edit erases historical payloads or
Git history. Those require separate, explicitly scoped cleanup before publication.

Retries reconcile an interrupted collection creation: verify dense/sparse vector
configuration, recreate missing payload indexes, and verify their final schemas.
Incompatible vector/index schemas fail closed and require explicit rebuild.

## Retrieval and verification

`KnowledgeRetriever.search()` returns `{results, latency_ms, mode,
chunks_returned}`; `.retrieve()` preserves the usual list-of-records boundary.
Modes: bm25, dense, hybrid. Explicit filters: market/topic/status/research_id
(exact strings), tags (all requested tags), date (`gte`/`lte` ISO **record** dates),
include_superseded (default false). No inferred language-trigger filters.
All filters apply to both prefetch branches and the fused query.

Dense Top-20 + BM25 Top-20 -> Qdrant RRF(k=60) -> Top-20 chunks -> collapse by
research_id in first-fused-occurrence order -> Top-5 full canonical records.
Top-20 collapse may produce fewer than five records; there is no hidden refill.
No similarity threshold is invented: raw retrievers may return irrelevant
records. That is measured by no-answer false positives, not called support.

Every search checks the manifest, canonical hash/path set, model revision (dense
modes), alias target and returned chunk identities/text/hashes. Indexed filter
metadata (research_id/market/topic/status/tags/date/is_superseded, including their
types) is compared against canonical records; mismatches fail closed. Hydration checks that each
canonical record satisfies the requested filters, even if the server ignored them.
Stale/deleted or tampered points fail the whole request, never partial success. Small-corpus full
scans are deliberate; caching should follow measurement. Public candidates are
hydrated from full canonical records, not snippets.

`verify_candidates()` accepts any retriever's full records. An explicit canonical
evidence-field allowlist in `verify_record()` excludes all retrieval diagnostics
(not just one score key), and avoids duplicating section text. It preserves fused order among
accepted records; it is relevance **verification**, not a cross-encoder reranker.
Multiple candidates share one provider call with independent per-record
`research_id`/`supported`/`reason` decisions. Response IDs must match the complete
input set exactly once; results are restored to input order. Any malformed row,
missing/duplicate/unknown ID or provider error fails the entire batch closed.
An empty candidate list makes no call; one candidate retains the single-record
contract. No candidate limit is reduced. Existing
`retrieve_verified()` remains the legacy semantic-retrieval wrapper.

## Agent runtime (opt-in)

After explicitly syncing/rebuilding the index, select the backend:

```bash
python -m agent.main "研究 A 股交易成本的默认口径" --retrieval qdrant
# Explicitly select dense + BM25/RRF instead of the runtime dense default:
python -m agent.main "研究 A 股交易成本的默认口径" --retrieval qdrant --retrieval-strategy hybrid
```

Default `--retrieval none` is unchanged; `ollama` / `openai` retain legacy
semantic retrieval. Qdrant uses the indexed profile's dense embedding model;
the optional hybrid strategy adds native BM25/RRF, not a legacy embedding provider.
`--retrieval-strategy dense|hybrid` applies to Qdrant only. No automatic fallback,
indexing, rebuilding or model download occurs.

The runtime boundary is `run_agent(..., retrieval_backend="qdrant", retrieval_strategy="dense",
retrieval_client=verifier_client, knowledge_retriever=retriever,
retrieval_filters={"tags": ["cost"]}, candidate_limit=5)`.
Inject any object with the `KnowledgeRetriever.search()` contract for hermetic
tests; omit it to construct the standard backend **after request safety checks**.
`run_request(..., retrieval="qdrant", knowledge_retriever=retriever, ...)` forwards
these options. `market`, `topic`, `record_status` map to exact backend filters;
`retrieval_filters` exposes the complete filter contract above. Conflicting or
invalid filters fail configuration validation; no language-trigger filters are
inferred. The Qdrant record candidate limit is 1–5 per lookup, with no hidden refill.
Runtime strategy defaults to dense; set `retrieval_strategy="hybrid"` to retain
hybrid retrieval. The standalone backend's hybrid default, comparison eval and
index profile remain unchanged; selecting a runtime strategy needs no rebuild.

Retrieval is a model-selected `search_knowledge(query, answer_target)` tool, not an entry
prerequisite. A generic answer or clarification makes no Qdrant/embedding/verifier
call. The backend is constructed only when the model actually requests lookup.

Lookup performs canonical full-record hydration, quarantine and compact related
brief projection. Successive lookups accumulate safe records by `research_id` in
first-seen order; a repeated ID updates the same entry, and an empty lookup or
`lookup_history` does not clear it. A newly quarantined version removes that ID
from the safe pool. The global pool is capped at `candidate_limit` (maximum five),
independent of the action budget. Its active serialized brief snapshot is at most
6,000 characters. New records displace oldest retained records if either budget
is exceeded; duplicate IDs update in place. Older tool replies retain only a
superseded marker, not another full brief snapshot. Full verifier records are
limited to 12,000 serialized characters each; oversized records fail explicitly
with a retrieval error before verification, without truncating research facts.
The next primary model turn can choose fresh research (no
verifier) or finish a historical answer. Only the latter runs
`verify_candidates()` against one historical answer target, then supported evidence →
synthesis/grounding. Empty/unsupported records abstain rather than invent support.
Infrastructure/stale-index/verifier errors remain explicit retrieval errors.
No implicit indexing or silent fallback is introduced.

Each lookup query is at most 1,000 characters; unique recent queries are retained
within a 2,000-character diagnostic budget, not used as verification constraints.
The answer target defaults to the exact user turn. For ambiguous historical
follow-ups, the primary model explicitly selects one self-contained question via
the semantically optional `answer_target` value (at most 2,000 characters), resolving only
from bounded safe conversation context and preserving the user's subject/date/scope.
Both fields are required by the strict tool schema; `answer_target` accepts a string
or null. An explicit target replaces the previous target; null leaves it unchanged.
Exploratory/method-reuse searches never automatically change the answer question;
queries are neither conjoined nor treated as alternatives. Verifier strictness
is unchanged. `lookup_history` returns only omitted older safe content
(at most four messages / 2,000 content characters), once; repeat calls return an
empty history and `already_read`. History never becomes evidence.

Briefs contain identity/title, date/market/status, question (240 chars), method
(160), conclusion (200), caveats (160), and truncation flags, not scores/provenance
or full records. Full raw record metadata/provenance is quarantined before this
projection. Briefs are model context only; only verified full records may receive
`knowledge-<research_id>` citations. When fresh research executes, its ToolResults,
not the historical briefs, become evidence.

Trace distinguishes `candidate_ids`/`related_research_ids` from
`verified_ids`/`research_ids`, `candidate_status` and `verification_status`, plus
resolved lookup queries, the final verification query and latest evicted IDs.
Backend latency is accumulated across lookups; candidate/quarantine time and
verification time stay separate. `runtime_total_ms` is their sum, excluding
primary-model, HITL and synthesis/grounding time. Loop outcome reflects the final
synthesis/grounding outcome, not merely successful execution. Provider telemetry uses
`model` for primary loop turns and `retrieval_verifier` only for the historical
evidence gate. `observed.model.tool_calls` records native decisions;
`plan`/`observed.planning`/`observed.orchestration` are deprecated null JSON
compatibility fields, not fabricated stages. The deprecated `planner_client`
argument aliases the primary `client`; new callers should use `client`.
Injected synthesis/grounding/verifier clients remain supported; by default the
primary client supplies them. HTTP/CLI response shape and dense default stay intact.

The evidence outer contract remains exactly `{id, text}`. Knowledge IDs are
`knowledge-<research_id>`; `text` is compact JSON with:

```json
{
  "evidence_type": "knowledge_record",
  "research_id": "RR-010",
  "title": "...",
  "content": "full canonical record body, including caveats and provenance",
  "provenance": {
    "source_path": "resources/knowledge/research/rr-010-trading-cost-defaults.md",
    "source_hash": "<canonical file SHA256>",
    "source_type": "research_record",
    "source_ref": [],
    "record_date": "2026-09-01",
    "record_status": "validated"
  }
}
```

Missing optional provenance is null/empty, never invented. Projection uses an
explicit field allowlist; scores, matched chunks and retrieval diagnostics are
excluded. At most five unique verified records are projected, each serialized
`text` at most 12,000 characters. Oversized records fail closed with
`knowledge_evidence_error` at `retrieval`, rather than silently truncating facts.
Canonical content/provenance is quarantined before projection; all model stages
retain trust-boundary instructions. Synthesis
must visibly cite knowledge IDs and preserve historical date/status/scope; only
cited evidence proceeds to grounding and the public evidence list. Provider and
malformed-answer errors remain explicit at synthesis/grounding, with retrieval
traces and normal provider token/latency telemetry preserved.

If the model selects research tools, deterministic tools / Research Experiment execute
normally. Their `step-<n>-<tool>` evidence remains unchanged, and historical
records are **not** substituted for or mixed into fresh execution evidence.
`request_clarification` returns `needs_input` without verification. An empty lookup
returns related context to the model, which may still choose fresh research;
ending the historical answer path without verified evidence returns `abstain`.

## Evaluation

```bash
python -m agent.retrieval.eval
python -m agent.retrieval.eval --qdrant
# Full per-case results, slices and latency splits:
python -m agent.retrieval.eval --qdrant --json
# Explicitly opt into provider costs when measuring real verification quality:
python -m agent.retrieval.eval --qdrant --verification-provider deepseek
```

`compare_retrievers()` runs the same labeled query set through lexical, native
BM25, dense, hybrid and hybrid+verification. The existing 23-case oracle remains
unchanged; `resources/eval/rag_retrieval.json` adds six identifier/filter/deleted/
superseded/chunk-collapse cases. Current canonical records have no replacements;
actual replacement/deletion transitions are covered using hermetic fixtures.
The Qdrant CLI defaults to one compact comparison table: Recall/Precision@1/3/5,
MRR@5, no-answer false-positive/abstention rates, mean end-to-end query latency and
failed/total query counts. `--json` preserves the complete evaluation envelope.

Metrics on unique research IDs: Recall@1/3/5, Precision@1/3/5 (hits/K), MRR@5,
no-answer false-positive and abstention rates, error counts, per-query/mean
latency splits (canonical validation, embedding, Qdrant, hydration, verification
where available). Raw similarity is not passed to verification. Errors are not
counted as successful abstention or as true negatives. Quality metrics use valid
queries only; total/valid/error counts expose availability separately. No-answer
rates are null (not zero) when all negative queries failed. The CLI's default oracle verifier is explicitly
a deterministic **harness check, not a model-quality measurement**. Real model
verification quality requires an explicitly selected provider and human review.
Answer grounding/usefulness reuse the existing downstream contracts/evals.
`resources/eval/agent.json` also includes knowledge-only success and grounding
rejection traces, checked against deterministic actual runtime fixtures in
`tests/test_agent_knowledge.py`. These do not measure real model answer quality.
The AE-15 RR-010 benchmark compares the prior per-record/full-context shape to
the batch/brief path with unchanged candidates and deterministic decisions:

```bash
python -m pytest -q -s tests/test_agent_knowledge.py -k efficiency_benchmark
```

It reports existing per-stage telemetry using synthetic fixture-token units,
not real provider token/cost/latency measurements. Production telemetry still
uses only provider-reported usage; no token estimator was added to runtime.

Hermetic regression command:

```bash
python -m pytest -q tests/test_rag_retrieval.py tests/test_retrieval.py tests/test_retrieval_relevance_verifier.py tests/test_retrieval_runtime.py
python -m pytest -q tests/test_agent_knowledge.py tests/test_agent_output_binding.py tests/test_agent_eval.py
```

The SDK request-model contract test needs the pinned dependency but no live
service. All other tests use storage/embedding/provider doubles. Live server
BM25 tokenization (especially Chinese), Qwen recall, ARM64 latency and actual
native RRF acceptance must be inspected separately; deterministic doubles do
not establish retrieval quality.
