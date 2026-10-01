# AE-16 Product UAT

`product_uat.json` is a versioned 12-task acceptance suite, not a prompt or
synthetic research benchmark. `python -m agent.product_uat` executes the existing
`agent.main.run_request` / `run_agent` path. It does not change retrieval, planning,
safety, synthesis, grounding, executor limits, candidate limits or retry policy.
Expected routes, acceptance labels/patterns and review notes never enter model
inputs. Requests are sent verbatim.

The CLI requires an explicit provider choice: these commands incur real provider
costs and, with the default retrieval backend, use the existing Qdrant/Ollama
services and prebuilt knowledge index. The harness does not start services, pull
models, build an index, install packages, generate weights or make approvals.

```bash
# Default: all cases; Qdrant dense retrieval; compact per-case + aggregate output.
python -m agent.product_uat --provider deepseek

# Select cases, retain hybrid retrieval, inspect full trace/evidence as JSON.
python -m agent.product_uat --provider deepseek --retrieval-strategy hybrid \
  --case UAT-01 --case UAT-03 --json

# UAT-07 accepts only caller-prepared CSV/Parquet artifacts.
python -m agent.product_uat --provider deepseek --case UAT-07 \
  --weights "$WEIGHTS_ARTIFACT" --close "$CLOSE_ARTIFACT" --open "$OPEN_ARTIFACT"

# Optional rating supplied by a human, never inferred by the runner/model.
python -m agent.product_uat --provider deepseek --case UAT-01 --usefulness UAT-01=2
```

UAT-07 keeps `${target_weights}`, `${price_panel}`, `${open_panel}` in the dataset.
Only the corresponding supplied paths are injected as an existing `runtime_truth`
context item; no tool choice or research implementation hint is added. Missing
bindings produce `not_run`, `case_pass=null`, an explicit precondition reason and
no runtime/provider action for that case. Unknown options/ratings fail before any
runtime call. A missing file with a supplied path is left to the existing tool's
real validation, not relabeled as success. No paths specific to an operator/host
are recorded in the canonical dataset.

## Acceptance and measurement

Each result has `expected_route`, `terminal_outcome`, `task_success`,
`route_correct`, `grounded`, `unnecessary_action_count`, `human_intervention`,
nullable `usefulness_0_2`, `provider_calls`, `tokens`, `estimated_cost` plus its
currency, `wall_clock_ms`, `failure_stage`, `case_pass`, and `failed_checks`.
Full JSON retains the existing observed trace, per-stage provider telemetry,
bounded cited evidence, answer, synthesis and grounding; it does not capture
provider requests/responses, private authored source or credentials. Harness-level
exception messages are omitted; existing controlled runtime error traces remain.
Harness-level exceptions are explicit failures with unknown usage;
later cases still run. Nothing is persisted automatically.

`case_pass` gates route, terminal outcome, safety, lifecycle, validated cited
grounding and case-specific core checks:

- Knowledge: expected canonical IDs/statuses/provenance, no ResearchRun/action,
  inline citations, answer acceptance patterns, and record-specific grounded
  claim patterns. Multi-record claims cannot swap inconclusive/rejected labels.
- Fixed tools: exactly the intended action, normalized arguments, fresh tool
  evidence, dated universe counts, requested factor horizons/groups, or bound
  backtest artifacts with canonical T+1 and default asymmetric costs.
- Experiments: a valid structured spec and requested scope, real completed run,
  existing result/provenance validators, positive samples/metrics, isolated
  executor identity, passed validation, matching actual bounds, and at most one
  repair. Historical evidence cannot substitute for a fresh tool result.
- Clarification: no execution/evidence; bounded clarification of missing strategy
  information, without requests for internal APIs/CSV preparation.
- Safety: blocked at safety with zero provider calls/actions, no answer/evidence
  exposure, and explicit safety failure attribution.

Patterns are conservative bilingual acceptance checks, **not a semantic judge**.
They can miss correct paraphrases and do not prove the completeness of every
financial interpretation. Inspect `failed_checks`, actual answer/grounded claims,
evidence and each case's `review_notes` during product review. Existing grounding
remains the support check; this harness does not add another model evaluator.
Dates/spec checks prove requested scope and reported coverage, not that missing
market observations exist. Human Usefulness is optional: 0 = not useful,
1 = partially useful, 2 = useful; default null. It cannot turn a correctness
failure into a pass, and a human 0 does not override otherwise passing hard gates.

Provider calls/tokens/cost/runtime wall-clock are copied from existing telemetry;
there is no token estimator or cost/latency threshold. Aggregate usage is unknown
when any executed case lacks it; mixed currencies are not summed. Not-run cases
are excluded from executed rates but explicitly counted alongside selected,
executed, passed and failed cases. Exit code is 0 only when **all selected** cases
pass; failures and not-run cases return 1. Bad CLI configuration returns 2.
Aggregate task/route/grounding rates include only evaluated non-null values;
grounding is not applicable to clarification or pre-provider safety stops.

## Hermetic checks and known product gaps

```bash
python -m pytest -q tests/test_product_uat.py
```

Tests use deterministic runtime/ToolResult fixtures and provider/retriever
doubles, including an actual knowledge-only `run_request` path. Fixture passes
validate the harness, not live retrieval quality or quantitative results.

The exact Chinese UAT-12 exposes a pre-existing runtime gap: deterministic safety
patterns currently recognize English commands, so this request can reach planning
before a safety stop. A sentinel provider regression reports failure rather than
changing/translating the request or adding a harness safety shim. Runtime is left
unchanged. UAT-03 and UAT-08 also intentionally test whether whole-query relevance
verification accepts complementary historical records/update context; the harness
does not force acceptance or inject expected research IDs. Live outcomes remain
unmeasured until an operator explicitly runs the suite.
