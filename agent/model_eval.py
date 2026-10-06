"""First-decision routing fixtures, not live research UAT or model-quality scores."""

from types import SimpleNamespace

from .agent import run_agent
from .core.resources import load_json
from .core.telemetry import RunTelemetry, TelemetryClient
from .tools.calling import _extract_function_calls

CASES = load_json("eval/model_first.json")


class FixtureClient:
    provider = "fixture"

    def __init__(self, case):
        self.case = case

    def create(self, payload):
        return {**self.case["response"], "usage": {"input_tokens": 100, "output_tokens": 20}}


class _FirstDecision(BaseException):
    """Escape before runtime dispatch, including deterministic read-only actions."""

    def __init__(self, response):
        self.response = response


def run_case(case, *, client=None):
    telemetry = RunTelemetry()
    primary = client if client is not None else FixtureClient(case)

    def first_decision(payload):
        response = TelemetryClient(primary, telemetry, stage="model", model="deepseek-flash").create(payload)
        raise _FirstDecision(response)

    route, answer, error_type = "error", "", None
    try:
        result = run_agent(case["user_request"], client=SimpleNamespace(client=primary, create=first_decision),
                           model="deepseek-flash", retrieval_backend="qdrant",
                           conversation_history=case.get("history"))
        error_type = result["error_type"] or result["observed"]["outcome"]["status"]
    except _FirstDecision as captured:
        try:
            calls = _extract_function_calls(captured.response)
            answer = captured.response.get("output_text", "")
            if not isinstance(answer, str):
                raise ValueError("expected textual output")
            if len(calls) > 1 or (not calls and not answer.strip()):
                raise ValueError("expected text or one tool call")
            route = calls[0]["name"] if calls else "direct"
            if route == "request_clarification":
                answer = calls[0]["arguments"]["question"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("expected a nonempty clarification")
        except (KeyError, ValueError, TypeError):
            error_type = "malformed_response"
    expected = case["expected"]
    summary = telemetry.envelope()["summary"]
    return {"case": case["id"], "route": route,
            "case_pass": error_type is None and route == expected["route"] and
                         all(text in answer for text in expected["answer_contains"]) and
                         all(text not in answer for text in expected.get("answer_excludes", [])),
            "provider_calls": summary["calls"], "total_tokens": summary["total_tokens"],
            "input_tokens": summary["input_tokens"], "output_tokens": summary["output_tokens"],
            "estimated_cost": summary["estimated_cost"], "latency_ms": summary["provider_latency_ms"],
            "error_type": error_type}


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
