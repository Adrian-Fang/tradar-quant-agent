"""Context Engineering Eval v0: paired, rule-scored context decisions."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from typing import Any

from agent.tool_calling import DeepSeekChatClient, OpenAIResponsesClient


CONTEXT_CASES = (
    {
        "id": "momentum_strong_evidence",
        "pair_id": "momentum_evidence_state",
        "group": "context_sensitivity",
        "user_request": "momentum_60d 现在应该进入下一步吗？请根据现有上下文给出下一步。",
        "context": {
            "current_state": "因子评估已经完成，结果已记录在当前 research run。",
            "research_result": {
                "factor": "momentum_60d",
                "status": "complete",
                "ic_5d": 0.035,
                "ic_20d": 0.041,
                "yearly_stability": "3/4 years positive",
                "coverage": "adequate",
            },
            "history": "研究人员此前要求：证据充分后再进入策略验证。",
        },
        "expected_behavior": {
            "decision": "advance_to_backtest",
            "should": "把稳定且已完成的因子证据作为进入策略/回测阶段的依据。",
            "should_not": "把已完成的评估当作未开始，或无理由重新做同一评估。",
        },
        "critical_context": ["momentum_evaluation_complete", "momentum_evidence_strong"],
        "failure_if": "重新评估或以证据不足为由阻止下一步。",
        "why": "测试 Agent 是否使用强证据，而不是只看用户提到的因子名。",
        "scoring": {
            "required_actions": ["propose_strategy_backtest"],
            "forbidden_actions": ["rerun_factor_evaluation", "block_for_missing_evidence"],
        },
    },
    {
        "id": "momentum_weak_evidence",
        "pair_id": "momentum_evidence_state",
        "group": "context_sensitivity",
        "user_request": "momentum_60d 现在应该进入下一步吗？请根据现有上下文给出下一步。",
        "context": {
            "current_state": "因子评估已经完成，结果已记录在当前 research run。",
            "research_result": {
                "factor": "momentum_60d",
                "status": "complete",
                "ic_5d": 0.002,
                "ic_20d": -0.004,
                "yearly_stability": "mixed and unstable",
                "coverage": "adequate",
            },
            "history": "研究人员此前要求：证据充分后再进入策略验证。",
        },
        "expected_behavior": {
            "decision": "stop_and_reassess",
            "should": "识别当前证据弱且不稳定，先重新审视因子或研究机制。",
            "should_not": "仅凭因子评估完成就直接推进策略回测。",
        },
        "critical_context": ["momentum_evaluation_complete", "momentum_evidence_weak"],
        "failure_if": "忽略弱证据并直接推进回测。",
        "why": "测试关键 evidence 改变后 decision 是否相应改变。",
        "scoring": {
            "required_actions": ["diagnose_or_reassess_factor"],
            "forbidden_actions": ["propose_strategy_backtest", "treat_evidence_as_strong"],
        },
    },
    {
        "id": "momentum_not_started",
        "pair_id": "momentum_evidence_state",
        "group": "context_sensitivity",
        "user_request": "momentum_60d 现在应该进入下一步吗？请根据现有上下文给出下一步。",
        "context": {
            "current_state": "当前 research run 只有用户问题，没有因子评估结果。",
            "research_result": {"factor": "momentum_60d", "status": "not_started"},
            "history": "研究人员此前要求：先确认因子预测能力，再考虑策略验证。",
        },
        "expected_behavior": {
            "decision": "evaluate_factor_first",
            "should": "先完成因子预测能力评估，再决定是否进入策略阶段。",
            "should_not": "假设因子有效并直接开始策略回测。",
        },
        "critical_context": ["momentum_evaluation_not_started", "factor_evidence_missing"],
        "failure_if": "把未开始误判成已验证，或直接生成策略结论。",
        "why": "测试 Agent 能否区分 weak evidence 与 no evidence。",
        "scoring": {
            "required_actions": ["run_factor_evaluation"],
            "forbidden_actions": ["propose_strategy_backtest", "claim_factor_is_valid"],
        },
    },
    {
        "id": "high52_fresh_result",
        "pair_id": "high52_result_availability",
        "group": "redundant_action",
        "user_request": "继续 high52 研究，先根据已有结果决定下一步，不要重复已经完成的计算。",
        "context": {
            "current_state": "最新 high52 evaluation run 已完成，结果和 coverage 均可读取。",
            "research_result": {"factor": "high52", "status": "fresh_result_available", "observed_through": "2026-08-31"},
            "history": "上一次工作已经生成了 high52 的 IC、分组收益和年度稳定性。",
        },
        "expected_behavior": {
            "decision": "reuse_existing_result",
            "should": "复用新鲜结果并基于结果安排后续工作。",
            "should_not": "重新运行同一 high52 evaluation，或声称没有结果。",
        },
        "critical_context": ["high52_fresh_result_available", "high52_prior_steps_complete"],
        "failure_if": "重复已经完成的研究步骤。",
        "why": "测试已有结果是否真正改变行动，而不是每次从头开始。",
        "scoring": {
            "required_actions": ["reuse_existing_result"],
            "forbidden_actions": ["rerun_high52_evaluation", "claim_result_missing"],
        },
    },
    {
        "id": "high52_no_result",
        "pair_id": "high52_result_availability",
        "group": "redundant_action",
        "user_request": "继续 high52 研究，先根据已有结果决定下一步，不要重复已经完成的计算。",
        "context": {
            "current_state": "当前 research run 没有 high52 evaluation 输出。",
            "research_result": {"factor": "high52", "status": "no_result_available"},
            "history": "旧笔记提到过 high52，但没有可验证的结果 artifact 或 run record。",
        },
        "expected_behavior": {
            "decision": "run_new_analysis",
            "should": "承认当前没有可复用结果，先开展一次新的 high52 评估。",
            "should_not": "把旧笔记当成结果，或声称已经完成评估。",
        },
        "critical_context": ["high52_result_missing", "old_note_not_verifiable"],
        "failure_if": "复用不存在或不可验证的结果。",
        "why": "counterfactual 地改变结果可用性，检查 reuse 与 new analysis 的边界。",
        "scoring": {
            "required_actions": ["start_high52_evaluation"],
            "forbidden_actions": ["reuse_unavailable_result", "claim_result_exists"],
        },
    },
    {
        "id": "stale_analytics_vs_repo_truth",
        "pair_id": "current_truth_over_stale_state",
        "group": "stale_context_resistance",
        "user_request": "按当前 repo 状态完成这个研究请求，应该采用哪个研究入口？",
        "context": {
            "current_runtime_truth": {
                "agent": "orchestration and tool contract only",
                "research": "canonical quantitative engine",
                "legacy_analytics": "retired",
            },
            "old_history": "旧上下文仍建议运行 scripts/run_strategy.py 和 legacy analytics。",
            "available_entry": "现有 agent tools 通过 research engine 执行确定性研究。",
        },
        "expected_behavior": {
            "decision": "follow_current_research_engine",
            "should": "以当前 repo/runtime truth 为准，沿现有 agent → research 路径工作。",
            "should_not": "恢复或假设旧 analytics 入口仍然有效。",
        },
        "critical_context": ["current_research_is_canonical", "legacy_analytics_retired"],
        "failure_if": "被旧上下文带回已经退役的入口。",
        "why": "测试 stale context resistance 与当前运行时事实优先级。",
        "scoring": {
            "required_actions": ["use_current_research_engine"],
            "forbidden_actions": ["use_retired_analytics", "invent_replacement_framework"],
        },
    },
    {
        "id": "stale_result_note_vs_current_artifact",
        "pair_id": "current_truth_over_stale_state",
        "group": "stale_context_resistance",
        "user_request": "旧上下文说 high52 已经验证过。现在请确认能否直接据此下结论。",
        "context": {
            "current_runtime_truth": {
                "current_run": "没有 high52 result",
                "verifiable_artifact": False,
                "data_asof": "当前状态中未确认",
            },
            "old_history": "2025 年旧笔记写着：high52 已验证，可以直接使用。",
        },
        "expected_behavior": {
            "decision": "require_current_evidence",
            "should": "先核实当前可验证结果，不能把旧笔记当成当前事实。",
            "should_not": "直接引用旧结论并声称当前研究已完成。",
        },
        "critical_context": ["current_high52_result_missing", "old_high52_note_stale"],
        "failure_if": "用旧状态替代当前 artifact/runtime truth。",
        "why": "测试旧结果看似具体时，Agent 是否仍检查当前事实。",
        "scoring": {
            "required_actions": ["verify_current_evidence"],
            "forbidden_actions": ["treat_stale_note_as_result", "claim_current_research_complete"],
        },
    },
    {
        "id": "current_no_extra_validation",
        "pair_id": "current_instruction_over_preference",
        "group": "conflict_resolution",
        "user_request": "只解释现有结果，不要额外 validation，不要重跑，也不要做参数扫描。",
        "context": {
            "current_instruction": "本次明确禁止额外 validation、重跑和参数扫描。",
            "existing_result": "现有 result 已包含用户要求的指标。",
            "history_preference": "历史偏好：每个研究都重复验证三次并做参数扫描。",
        },
        "expected_behavior": {
            "decision": "follow_current_scope",
            "should": "按当前明确指令解释现有结果并结束在指定范围内。",
            "should_not": "把历史偏好升级为本次额外工作。",
        },
        "critical_context": ["current_instruction_forbids_validation", "existing_result_sufficient_for_request"],
        "failure_if": "违反当前明确 scope，追加历史偏好的 validation。",
        "why": "测试当前用户指令是否覆盖历史偏好。",
        "scoring": {
            "required_actions": ["interpret_existing_result"],
            "forbidden_actions": ["perform_extra_validation", "run_parameter_scan", "rerun_existing_research"],
        },
    },
    {
        "id": "current_requests_validation",
        "pair_id": "current_instruction_over_preference",
        "group": "conflict_resolution",
        "user_request": "这次明确做一次 out-of-sample stability validation，再汇报结果；不要只复述旧结论。",
        "context": {
            "current_instruction": "本次明确要求 out-of-sample stability validation。",
            "existing_result": "已有 in-sample result，但没有本次要求的 stability check。",
            "history_preference": "历史偏好：研究尽量快速结束，不做额外 validation。",
        },
        "expected_behavior": {
            "decision": "perform_requested_validation",
            "should": "执行当前明确要求的 stability validation，再汇报。",
            "should_not": "用历史的快速结束偏好拒绝本次明确请求。",
        },
        "critical_context": ["current_instruction_requests_validation", "requested_validation_not_done"],
        "failure_if": "被旧偏好阻止执行当前明确要求的工作。",
        "why": "与上一 case 反向改变 current instruction，检查冲突处理而非固定偏好。",
        "scoring": {
            "required_actions": ["perform_requested_validation"],
            "forbidden_actions": ["follow_old_no_validation_preference", "only_restate_old_result"],
        },
    },
    {
        "id": "momentum_clean_context",
        "pair_id": "distractor_robustness",
        "group": "distractor_robustness",
        "user_request": "momentum_60d 的稳定结果已经出来了，下一步应做什么？",
        "context": {
            "critical": {
                "momentum_result": "complete, positive and stable",
                "next_stage": "strategy/backtest validation is not started",
            },
            "history": "只有与本次因子研究相关的历史。",
        },
        "expected_behavior": {
            "decision": "advance_to_backtest",
            "should": "抓住 momentum 结果已完成且下一阶段未开始这一关键状态。",
            "should_not": "重新做已经完成的因子评估。",
        },
        "critical_context": ["momentum_result_strong", "backtest_stage_not_started"],
        "failure_if": "无法从简洁上下文中提取下一步。",
        "why": "作为有干扰版本的 paired baseline。",
        "scoring": {
            "required_actions": ["use_momentum_result", "propose_strategy_backtest"],
            "forbidden_actions": ["rerun_factor_evaluation", "switch_to_unrelated_task"],
        },
    },
    {
        "id": "momentum_with_distractors",
        "pair_id": "distractor_robustness",
        "group": "distractor_robustness",
        "user_request": "momentum_60d 的稳定结果已经出来了，下一步应做什么？",
        "context": {
            "critical": {
                "momentum_result": "complete, positive and stable",
                "next_stage": "strategy/backtest validation is not started",
            },
            "history": [
                "旧 UI strategy viewer 的布局讨论。",
                "ETF 组合的部署日志。",
                "Cloudflare tunnel 重启记录。",
                "无关的股票池快照。",
                "旧的研究命名争论。",
            ],
        },
        "expected_behavior": {
            "decision": "advance_to_backtest",
            "should": "在无关历史很多时仍使用 momentum 结果和当前阶段状态。",
            "should_not": "转去处理 UI、部署、ETF 或无关股票池任务。",
        },
        "critical_context": ["momentum_result_strong", "backtest_stage_not_started"],
        "failure_if": "被无关历史吸引，改变研究决策或任务方向。",
        "why": "与 clean baseline 相同关键 context，仅增加干扰信息。",
        "scoring": {
            "required_actions": ["use_momentum_result", "propose_strategy_backtest"],
            "forbidden_actions": ["rerun_factor_evaluation", "switch_to_unrelated_task"],
        },
    },
    {
        "id": "strategy_compare_metrics_unspecified",
        "pair_id": "comparison_context_sufficiency",
        "group": "context_sufficiency",
        "user_request": "比较 strategy A 和 strategy B，并告诉我下一步。",
        "context": {
            "strategy_a": {"result": "available", "period": "2024-2026", "metrics": ["return", "drawdown", "turnover"]},
            "strategy_b": {"result": "available", "period": "2024-2026", "metrics": ["return", "drawdown", "turnover"]},
            "comparison_request": "用户没有指定 winner criterion 或优先级。",
        },
        "expected_behavior": {
            "decision": "proceed_with_explicit_criteria",
            "should": "承认两边结果都在，先明确或请求比较标准，再进行判断。",
            "should_not": "因为未指定 metric 就把任务判定为无法开始，或直接宣布赢家。",
        },
        "critical_context": ["both_strategy_results_available", "comparison_metric_unspecified"],
        "failure_if": "把可补充的偏好/标准误判成 blocking missing data。",
        "why": "测试 context sufficiency：可合理补充的信息不应阻塞。",
        "scoring": {
            "required_actions": ["ask_or_define_comparison_metric"],
            "forbidden_actions": ["block_for_missing_result", "declare_winner_without_criteria"],
        },
    },
    {
        "id": "strategy_compare_result_missing",
        "pair_id": "comparison_context_sufficiency",
        "group": "context_sufficiency",
        "user_request": "比较 strategy A 和 strategy B，并告诉我下一步。",
        "context": {
            "strategy_a": {"result": "available", "period": "2024-2026", "metrics": ["return", "drawdown", "turnover"]},
            "strategy_b": {"result": "missing", "period": "unknown", "metrics": []},
            "comparison_request": "用户没有指定 winner criterion，但 B 的结果本身缺失。",
        },
        "expected_behavior": {
            "decision": "blocked_missing_result",
            "should": "指出 strategy B 的结果缺失，在补齐前阻止比较结论。",
            "should_not": "用 A 的结果推断 B，或假装两边都已完成。",
        },
        "critical_context": ["strategy_b_result_missing", "comparison_data_incomplete"],
        "failure_if": "忽略真正 blocking 的 missing result。",
        "why": "与上一 case 对照，区分 missing preference 与 missing evidence。",
        "scoring": {
            "required_actions": ["block_on_missing_result"],
            "forbidden_actions": ["declare_winner", "pretend_b_result_available"],
        },
    },
)


def _all_action_tags() -> tuple[str, ...]:
    tags = {
        tag
        for case in CONTEXT_CASES
        for key in ("required_actions", "forbidden_actions")
        for tag in case["scoring"][key]
    }
    return tuple(sorted(tags))


def _prompt(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "",
        "instructions": (
            "Analyze the user request using the supplied context. Do not invent "
            "facts or claim work that the context does not support. Return JSON "
            "only with exactly these fields: decision (one string), actions "
            "(array of action tags), used_context (array of context labels), "
            "reason (short string). The action tags are only labels for the "
            "decision, not tools."
        ),
        "input": json.dumps(
            {
                "user_request": case["user_request"],
                "context": case["context"],
                "context_labels": case["critical_context"],
                "allowed_decisions": sorted({
                    item["expected_behavior"]["decision"] for item in CONTEXT_CASES
                }),
                "allowed_action_tags": _all_action_tags(),
            },
            ensure_ascii=False,
        ),
    }


class ContextFixtureClient:
    """Deterministic harness response; no model call is made."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case

    def create(self, payload: Mapping[str, Any]) -> dict[str, str]:
        behavior = self.case["expected_behavior"]
        scoring = self.case["scoring"]
        return {
            "output_text": json.dumps(
                {
                    "decision": behavior["decision"],
                    "actions": scoring["required_actions"],
                    "used_context": self.case["critical_context"],
                    "reason": behavior["should"],
                },
                ensure_ascii=False,
            )
        }


def _response_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _response_text(response: Any) -> str:
    direct = _response_field(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    choices = _response_field(response, "choices", []) or []
    if choices:
        message = _response_field(choices[0], "message")
        content = _response_field(message, "content", "")
        if isinstance(content, str):
            return content
    for item in _response_field(response, "output", []) or []:
        content = _response_field(item, "content", []) or []
        for part in content:
            text = _response_field(part, "text", "")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _parse_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].removesuffix("```").strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None, "response is not valid JSON"
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"response is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "response JSON must be an object"
    return parsed, None


def run_context_case(
    case: dict[str, Any],
    *,
    client: Any,
    model: str,
) -> dict[str, Any]:
    payload = _prompt(case)
    payload["model"] = model
    try:
        response = client.create(payload)
        text = _response_text(response)
        parsed, parse_error = _parse_response(text)
    except Exception as exc:
        text = ""
        parsed = None
        parse_error = f"provider_error: {type(exc).__name__}: {exc}"
    return {"response_text": text, "parsed": parsed, "parse_error": parse_error}


def score_context_case(
    case: dict[str, Any], outcome: dict[str, Any], repeat: int,
) -> dict[str, Any]:
    parsed = outcome["parsed"]
    behavior = case["expected_behavior"]
    scoring = case["scoring"]
    actions = set(parsed.get("actions", [])) if isinstance(parsed, dict) else set()
    used_context = set(parsed.get("used_context", [])) if isinstance(parsed, dict) else set()
    decision = parsed.get("decision") if isinstance(parsed, dict) else None
    missing_context = sorted(set(case["critical_context"]) - used_context)
    missing_actions = sorted(set(scoring["required_actions"]) - actions)
    forbidden_actions = sorted(set(scoring["forbidden_actions"]) & actions)
    decision_ok = decision == behavior["decision"]
    automated_pass = (
        outcome["parse_error"] is None
        and decision_ok
        and not missing_context
        and not missing_actions
        and not forbidden_actions
    )
    return {
        "case": case["id"],
        "pair_id": case["pair_id"],
        "group": case["group"],
        "repeat": repeat,
        "decision": decision or "<none>",
        "expected_decision": behavior["decision"],
        "decision_ok": decision_ok,
        "missing_context": missing_context,
        "missing_actions": missing_actions,
        "forbidden_actions": forbidden_actions,
        "automated_pass": automated_pass,
        "status": "pass" if automated_pass else "human_review",
        "human_review_required": not automated_pass,
        "human_review_reason": (
            "free-form reason is recorded but not automatically scored"
            if automated_pass
            else outcome["parse_error"] or "deterministic checks did not pass"
        ),
        "parse_error": outcome["parse_error"],
        "reason": parsed.get("reason", "") if isinstance(parsed, dict) else "",
        "response_text": outcome["response_text"],
    }


def run_context_eval(
    provider: str = "fixture", repeats: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if provider not in {"fixture", "openai", "deepseek"}:
        raise ValueError(f"unsupported provider: {provider}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if provider == "deepseek" and not os.getenv("DEEPSEEK_API_KEY"):
        return [], {"status": "skipped", "reason": "DEEPSEEK_API_KEY is not set"}
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return [], {"status": "skipped", "reason": "OPENAI_API_KEY is not set"}

    model = (
        os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        if provider == "deepseek"
        else os.getenv("OPENAI_MODEL", "gpt-5")
    )
    client = (
        DeepSeekChatClient() if provider == "deepseek"
        else OpenAIResponsesClient() if provider == "openai"
        else None
    )
    rows = []
    for case in CONTEXT_CASES:
        for repeat in range(1, repeats + 1):
            case_client = ContextFixtureClient(case) if provider == "fixture" else client
            outcome = run_context_case(case, client=case_client, model=model)
            rows.append(score_context_case(case, outcome, repeat))
    return rows, {"status": "complete", "provider": provider, "cases": len(CONTEXT_CASES), "repeats": repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Engineering Eval v0")
    parser.add_argument("--provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    rows, meta = run_context_eval(args.provider, args.repeats)
    if meta["status"] == "skipped":
        print(f"Context Eval v0 skipped: {meta['reason']}")
        return
    for row in rows:
        print(
            f"{row['case']} | {row['status']} | "
            f"decision={row['decision']} | expected={row['expected_decision']}"
        )
        if row["human_review_required"]:
            print(f"  review: {row['parse_error'] or row['missing_actions'] or row['forbidden_actions'] or row['missing_context']}")
    if args.verbose:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
