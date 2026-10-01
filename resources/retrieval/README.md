# AE-14 research knowledge backend

Markdown under `resources/knowledge/research/*.md` is canonical. Qdrant contains
derived chunks, never market data. No indexing happens on Agent startup, and
`run_agent()` still uses its existing retrieval path. No LangChain, LlamaIndex,
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
Malformed/provider errors retain the existing fail-closed behavior. Existing
`retrieve_verified()` remains a semantic-retrieval wrapper for the unchanged
Agent runtime.

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
Answer grounding/usefulness remain the existing downstream evals, because this
milestone does not change runtime synthesis/grounding.

Hermetic regression command:

```bash
python -m pytest -q tests/test_rag_retrieval.py tests/test_retrieval.py tests/test_retrieval_relevance_verifier.py tests/test_retrieval_runtime.py
```

The SDK request-model contract test needs the pinned dependency but no live
service. All other tests use storage/embedding/provider doubles. Live server
BM25 tokenization (especially Chinese), Qwen recall, ARM64 latency and actual
native RRF acceptance must be inspected separately; deterministic doubles do
not establish retrieval quality.
