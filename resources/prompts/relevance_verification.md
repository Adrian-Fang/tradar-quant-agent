# Relevance Verification

Decide whether the research record provides enough direct evidence to answer the user query as written.

Return exactly one JSON object with these fields:

```json
{"supported": true, "reason": "short evidence-based explanation"}
```

Set `supported` to `true` only when the record directly addresses the query. A merely related topic is not enough. Set it to `false` when the direction or polarity, variables, horizon, execution point, market scope, or research question materially differs, or when answering would require an inference beyond the record. Use only the supplied record; do not fill gaps with outside knowledge. The reason is recorded for review and does not replace the boolean decision.
