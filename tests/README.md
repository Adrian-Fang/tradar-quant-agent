# Test inventory and fast validation

Tests are grouped by primary purpose, not by every Agent stage transition.
Consolidation removes duplicate stage-plumbing assertions, not deterministic
quant/security contracts to hit a test-count target.

| Purpose | Files / boundary |
| --- | --- |
| Deterministic units | `test_research_panel`, `test_vector_backtest`, loaders/env, ToolResult/ResearchRun/state, evidence, safety, telemetry, provider adapter, context/memory/HITL/parser modules. No remote services. |
| Runtime/backend integration | `test_agent_model_loop` (one compact behavior suite), HTTP transport/history/lock tests, CLI wiring, fake-Qdrant retrieval/index integration. Tools/providers are injected fixtures. |
| Eval contracts | `test_agent_eval`, answer/grounding/planning/HITL/memory/context/retrieval/tool eval modules test scoring/parser invariants. Cases remain in `resources/eval/`, not separate runtime unit tests. `agent.model_eval` replaces the removed preflight eval with first-decision native-tool fixtures; it pauses research before execution and is not UAT/model-quality measurement. Old planning/AE-09 trace evals are standalone, not current runtime coverage. |
| Heavy isolation/resources | `sandbox/test_experiment_isolation.py`: 24 real namespace/execution/network/process/memory/spill/timeout regressions. Excluded from default pytest collection; namespace availability is probed only when this module is explicitly collected. Shared pure experiment fixtures/contracts remain in `test_agent_experiment.py`. |

Sandbox fixtures use explicit imports and module-qualified TestCase setup reuse;
no wildcard import exposes the fast TestCase for duplicate collection.

Removed overlapping preflight/runtime/knowledge/output-binding stage-plumbing
suites. Their direct/clarification/lookup/research/evidence/safety/failure/approval
contracts now live once in `test_agent_model_loop.py`; two output-bounding tests
remain in `test_agent_evidence.py`. Observability keeps provider discovery/usage/
pricing units; deterministic safety keeps multilingual/quoted-analysis rules.
Old loop unit tests remain only for the standalone loop helper, not a second
runtime. Index/quant/provider/security contracts were retained.

Small fast gate (no providers, Qdrant, Ollama or sandbox):

```bash
python -m pytest -q tests/test_agent_model_loop.py tests/test_agent_evidence.py tests/test_agent_observability.py tests/test_agent_safety.py
```

Heavy validation is a separate explicit operation, never part of that gate:

```bash
python -m pytest -o addopts='' tests/sandbox/
```

Use pytest rather than unittest discovery: the directory exclusion is pytest
configuration. Do not equate a passing fast gate with full-repo validation.
