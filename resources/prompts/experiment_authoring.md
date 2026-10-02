# Experiment Authoring Contract

Write one bounded Python research program for the supplied experiment spec.
Use only APIs listed in the supplied capability manifest. Do not execute the
program and do not invent APIs, files, schemas, network access, shell commands,
secret access, live trading, or source-tree writes.

Imports are an exact allowlist. Canonical APIs must use `from module import
name` as listed in `apis`. Safe libraries may be imported only with the exact
statement listed in `safe_libraries` (currently `import pandas as pd` and
`import numpy as np`). No other imports are allowed. Use these libraries only
for in-memory dataframe/numeric work; all data access must use canonical APIs.

Load the smallest date range and field set required. Prefer `price_panel(...,
field='close')` for close-to-close studies; it loads only close. If calling
`load_prices` directly, pass `fields=['close']` when close is sufficient. A
close-entry study must not load open prices unless its stated method uses them.
Follow `authoring_constraints` and the supplied `execution_budget`. Reuse input
panels and release unused wide intermediates before creating new ones. Do not
keep masked copies for every group/horizon simultaneously or load tradability
masks the method never consumes. Keep the requested research window and the
lookback/exit padding needed for its semantics; do not silently shrink the
study to fit memory. Let data-loading/resource failures reach the executor,
rather than catching them and reporting empty metrics as success. DuckDB
memory/thread/spill settings are established by the worker, not authored code.

Submit exactly one `submit_research_program` function call whose arguments have
exactly this shape:

```json
{"program":"Python source defining one synchronous run() with no arguments"}
```

`run()` must return an object matching the capability manifest's
`result_schema`, which is the authoritative field contract. Prefer JSON-native
values; the executor applies only the safe value conversions listed in
`result_normalization` and will not repair a wrong field shape. Put method
detail inside its object, for example
`{"type":"event_study","description":"close-to-close returns"}`. Use the
experiment spec's assumptions and disclose any additional material default.
Keep results compact; report counts needed to interpret missing or censored
observations.
