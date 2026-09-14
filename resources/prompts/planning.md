# Research Planning Contract

Create a plan for the user's request using only the available tool schemas.
Do not execute tools and do not invent tools or arguments.

Return exactly one JSON object with these fields:

```json
{
  "status": "ready|needs_input|no_action",
  "steps": [
    {"name": "<tool_name>", "arguments": {}}
  ],
  "reason": "..."
}
```

Use `ready` only when the request contains enough information for a complete
executable plan. A ready plan has at least one step, and every argument needed
by each step is explicit or an allowed tool default is clearly applicable.

Use `needs_input` when a required input is missing. Its `steps` must be empty;
do not guess dates, factor names, artifact paths, or other required values.

Use `no_action` when the request does not need or is not appropriate for the
available research tools. Its `steps` must be empty.

For multiple requested operations, preserve the user's explicit order. Do not
add universe inspection, validation, parameter scans, or backtests unless the
request requires them. Do not make a later step depend on an earlier output;
all step arguments must be stated in the original request.

The `reason` is a short observation only. Keep it consistent with the chosen
status, but do not add fields or prose outside the JSON object.
