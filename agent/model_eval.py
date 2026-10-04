"""First-decision routing fixtures, not live research UAT or model-quality scores."""

import json
from types import SimpleNamespace

from .agent import run_agent
from .core.resources import load_json

CASES = load_json("eval/model_first.json")


class FixtureClient:
    provider = "fixture"

    def __init__(self, case):
        self.case, self.model_calls = case, 0

    def create(self, payload):
        if isinstance(payload["input"], str):
            # Pause selected research actions before any executor runs.
            response = {"output_text": json.dumps({"decision": "needs_approval",
                "approval_request": "Fixture action selected.", "reason": "routing eval only"})}
        else:
            self.model_calls += 1
            response = self.case["response"] if self.model_calls == 1 else {"output_text": "No evidence yet."}
        return {**response, "usage": {"input_tokens": 100, "output_tokens": 20}}


def run_case(case, *, client=None):
    result = run_agent(case["user_request"], client=client or FixtureClient(case), model="deepseek-flash",
                       retrieval_backend="qdrant",
                       knowledge_retriever=SimpleNamespace(search=lambda *a, **kw: {"results": []}),
                       conversation_history=case.get("history"))
    calls = result["observed"].get("model", {}).get("tool_calls", [])
    route = calls[0]["name"] if calls else "direct"
    expected = case["expected"]
    summary = result["telemetry"]["summary"]
    return {"case": case["id"], "route": route,
            "case_pass": result["status"] == "ok" and route == expected["route"] and
                         all(text in (result["answer"] or "") for text in expected["answer_contains"]),
            "provider_calls": summary["calls"], "total_tokens": summary["total_tokens"],
            "error_type": result["error_type"]}


def run_eval():
    rows = [run_case(case) for case in CASES]
    return rows, {"cases": len(rows), "passed": sum(row["case_pass"] for row in rows)}


def main():
    rows, summary = run_eval()
    print("First-decision fixtures (synthetic usage) | case | route | pass | calls | tokens")
    for row in rows:
        print(f"{row['case']} | {row['route']} | {row['case_pass']} | {row['provider_calls']} | {row['total_tokens']}")
    print(f"Passed {summary['passed']}/{summary['cases']}")


if __name__ == "__main__":
    main()
