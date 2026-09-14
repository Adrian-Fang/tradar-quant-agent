# Relevance Verification

Decide whether the supplied Research Record provides enough direct evidence to answer the user query exactly as written. Every material constraint in the query must be supported by the record; matching only the general topic is not sufficient. A record may support a query even when its empirical conclusion is negative.

Check the query and record in this order:

1. variable or event definition
2. condition attachment and event stage
3. direction or polarity
4. comparator or benchmark
5. horizon
6. execution point
7. market and research scope

A query constraint is supported only when the record applies it to the same variable, event, entity, and time point required by the query. Do not combine facts from different event stages, variables, entities, or time points into a joint condition that the record never establishes. Each constraint must remain attached to the evidence for its own event or subject.

If any material item does not match, set `supported` to `false`. Do not silently substitute one concept for another. In particular, treat opposite directions or polarity, proxy or inverse variables, and related-but-different metrics as different unless the record explicitly establishes their equivalence. Do not generalize a narrow event to a broad event, or apply broad evidence to a narrower condition, unless the record explicitly does so. Similar but different horizons or execution points are also mismatches.

If answering requires replacing a variable or condition, generalizing across scope, or relying on financial common sense, outside knowledge, or intuition not stated in the record, set `supported` to `false`. Use only the supplied record and do not fill gaps with plausible assumptions.

Return exactly one JSON object with these fields:

```json
{"supported": true, "reason": "short evidence-based explanation"}
```

Set `supported` to `true` only after all material constraints pass the checks above. The reason should briefly cite the matching evidence or the decisive mismatch; it is recorded for review and does not replace the boolean decision.
