"""Deterministic relevance evaluation for retrieval implementations."""

from __future__ import annotations

from time import perf_counter
import argparse
import json

from ..core.providers import OllamaEmbeddingClient
from ..core.resources import load_json
from .semantic_retriever import prepare_semantic_corpus, retrieve_semantic
from .lexical_retriever import retrieve


CASES = load_json("eval/retrieval.json")


def run_case(
    case: dict,
    result_ids: list[str],
    k_values: tuple[int, ...],
    result_scores: list[float | None] | None = None,
) -> dict:
    # Score records, never chunks. Preserve the first occurrence and its score.
    unique = list(dict.fromkeys(result_ids))
    if result_scores is not None:
        result_scores = [result_scores[result_ids.index(key)] for key in unique]
    result_ids = unique
    relevant_ids = set(case["relevant_ids"])
    row = {
        "case": case["id"],
        "slice": case.get("slice", "baseline"),
        "relevant": list(case["relevant_ids"]),
        "top_results": result_ids,
    }
    if result_scores is not None:
        row["top_scores"] = result_scores
    first_rank = next(
        (index + 1 for index, result_id in enumerate(result_ids) if result_id in relevant_ids),
        0,
    )
    row["reciprocal_rank"] = 1 / first_rank if first_rank else 0.0
    row["reciprocal_rank@5"] = 1 / first_rank if 0 < first_rank <= 5 else 0.0
    if relevant_ids:
        row["first_relevant_rank"] = first_rank or None
        row["first_relevant_research_id"] = (
            result_ids[first_rank - 1] if first_rank else None
        )
        row["first_relevant_score"] = (
            result_scores[first_rank - 1]
            if first_rank and result_scores is not None
            else None
        )
    else:
        row["top1_research_id"] = result_ids[0] if result_ids else None
        row["top1_score"] = (
            result_scores[0]
            if result_ids and result_scores is not None
            else None
        )

    for k in k_values:
        if not relevant_ids:
            row[f"hit@{k}"] = None
            row[f"recall@{k}"] = None
            row[f"precision@{k}"] = None
            continue
        hits = len(relevant_ids & set(result_ids[:k]))
        row[f"hit@{k}"] = 1.0 if hits else 0.0
        row[f"recall@{k}"] = hits / len(relevant_ids)
        row[f"precision@{k}"] = hits / k

    row["false_positive"] = bool(result_ids) if not relevant_ids else None
    return row


def run_eval(k_values: tuple[int, ...] = (1, 3, 5), retriever=retrieve, *, cases=None, measure_latency: bool = False) -> dict:
    k_values = tuple(sorted(set(k_values)))
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    max_k = max(k_values)
    rows = []
    for case in CASES if cases is None else cases:
        filters = {
            key: case[key]
            for key in ("market", "topic", "status", "tags", "research_id", "date", "include_superseded")
            if key in case
        }
        tick = perf_counter()
        outcome = retriever(case["query"], limit=max_k, **filters)
        results = outcome["results"] if isinstance(outcome, dict) else outcome
        row = run_case(
            case,
            [result["research_id"] for result in results],
            k_values,
            [result.get("score") for result in results],
        )
        if measure_latency:
            row["latency_ms"] = outcome.get("latency_ms", {}) if isinstance(outcome, dict) else {}
            row["latency_ms"] = {**row["latency_ms"], "eval_total_ms": (perf_counter() - tick) * 1000}
        if isinstance(outcome, dict) and outcome.get("status") == "error":
            row["error"] = outcome.get("errors", outcome.get("reason", "retrieval failed"))
            row["false_positive"] = None
            for k in k_values:
                for metric in ("hit", "recall", "precision"):
                    row[f"{metric}@{k}"] = None
        rows.append(row)

    groups = {"overall": rows}
    for row in rows:
        groups.setdefault(row["slice"], []).append(row)

    aggregate = {}
    for name, group in groups.items():
        valid_rows = [row for row in group if "error" not in row]
        relevant_rows = [row for row in valid_rows if row["relevant"]]
        macro_by_k = None
        mrr = None
        if relevant_rows:
            macro_by_k = {}
            for k in k_values:
                macro_by_k[str(k)] = {
                    "hit@k": sum(row[f"hit@{k}"] for row in relevant_rows) / len(relevant_rows),
                    "recall@k": sum(row[f"recall@{k}"] for row in relevant_rows) / len(relevant_rows),
                    "precision@k": sum(row[f"precision@{k}"] for row in relevant_rows) / len(relevant_rows),
                }
            mrr = sum(row["reciprocal_rank"] for row in relevant_rows) / len(relevant_rows)

        no_relevance_rows = [row for row in group if not row["relevant"]]
        valid_negatives = [row for row in no_relevance_rows if "error" not in row]
        false_positive_count = sum(row["false_positive"] for row in valid_negatives)
        aggregate[name] = {
            "mrr": mrr,
            "mrr@5": sum(row["reciprocal_rank@5"] for row in relevant_rows) / len(relevant_rows) if relevant_rows else None,
            "error_count": sum("error" in row for row in group),
            "valid_query_count": len(valid_rows),
            "by_k": macro_by_k,
            "no_relevance": {
                "count": len(no_relevance_rows),
                "valid_count": len(valid_negatives),
                "error_count": len(no_relevance_rows) - len(valid_negatives),
                "false_positive_count": false_positive_count,
                "false_positive_rate": (
                    false_positive_count / len(valid_negatives)
                    if valid_negatives else (None if no_relevance_rows else 0.0)
                ),
                "empty_result_count": sum(
                    not row["top_results"] for row in valid_negatives
                ),
                "abstention_rate": sum(not row["top_results"] for row in valid_negatives) / len(valid_negatives) if valid_negatives else (None if no_relevance_rows else 0.0),
            },
        }
        if measure_latency:
            keys = {key for row in group for key in row["latency_ms"]}
            aggregate[name]["mean_latency_ms"] = {
                key: sum(row["latency_ms"][key] for row in group if key in row["latency_ms"]) / sum(key in row["latency_ms"] for row in group)
                for key in sorted(keys)
            }

    return {
        "k_values": list(k_values),
        "cases": rows,
        "macro": aggregate["overall"],
        "slices": {
            name: metrics for name, metrics in aggregate.items() if name != "overall"
        },
    }


def compare_retrievers(retrievers: dict, *, cases=None, measure_latency: bool = True) -> dict:
    """One oracle shared across lexical/BM25/dense/hybrid/verified implementations."""
    return {name: run_eval(retriever=retriever, cases=cases, measure_latency=measure_latency)
            for name, retriever in retrievers.items()}


def print_comparison(evaluations: dict) -> None:
    """Compact presentation of existing aggregates; no recomputation or mutation."""
    def number(value, precision=3):
        return "-" if value is None else f"{value:.{precision}f}"

    rows = [["Mode", "Recall@1/3/5", "Precision@1/3/5", "MRR@5", "FP rate", "Abstain", "Mean ms", "Errors"]]
    for name, evaluation in evaluations.items():
        macro = evaluation["macro"]
        by_k = macro["by_k"] or {}
        negative = macro["no_relevance"]
        rows.append([
            name,
            *["/".join(number(by_k.get(str(k), {}).get(metric)) for k in (1, 3, 5))
              for metric in ("recall@k", "precision@k")],
            number(macro["mrr@5"]), number(negative["false_positive_rate"]),
            number(negative["abstention_rate"]),
            number(macro.get("mean_latency_ms", {}).get("eval_total_ms"), 1),
            f"{macro['error_count']}/{len(evaluation['cases'])}",
        ])
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    for index, row in enumerate(rows):
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
        if index == 0:
            print("  ".join("-" * width for width in widths))
    print("Mean ms: end-to-end per query. Errors: failed/total queries; quality metrics exclude failures.")
    print("'-' = unmeasured. Full cases, slices and latency splits: --json.")


def threshold_sweep(
    rows: list[dict],
    thresholds: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[dict]:
    positive_rows = [row for row in rows if row["relevant"] and "error" not in row]
    negative_rows = [row for row in rows if not row["relevant"] and "error" not in row]
    return [
        {
            "threshold": threshold,
            "positive_retention": (
                sum(
                    row["first_relevant_score"] is not None
                    and row["first_relevant_score"] >= threshold
                    for row in positive_rows
                ) / len(positive_rows)
                if positive_rows else 0.0
            ),
            "negative_rejection": (
                sum(
                    row["top1_score"] is None
                    or row["top1_score"] < threshold
                    for row in negative_rows
                ) / len(negative_rows)
                if negative_rows else 0.0
            ),
        }
        for threshold in thresholds
    ]


def print_report(label: str, evaluation: dict, *, abstention_analysis: bool = False) -> None:
    k_values = evaluation["k_values"]
    print(f"\n=== {label} ===")
    print("\nCase Results")
    print("case | relevant | top results")
    for row in evaluation["cases"]:
        scores = row.get("top_scores")
        top_results = (
            [
                f"{result_id}({score:.4f})" if score is not None else result_id
                for result_id, score in zip(row["top_results"], scores)
            ]
            if scores is not None
            else row["top_results"]
        )
        print(f"{row['case']} | {row['relevant']} | {top_results}")

    metric_columns = [
        ("hit", k) for k in (1, 3) if k in k_values
    ] + [
        ("recall", k) for k in (3, 5) if k in k_values
    ]
    print("\nCore Case Metrics")
    print(
        "slice | case | "
        + " | ".join(f"{metric.title()}@{k}" for metric, k in metric_columns)
        + " | RR"
    )
    for row in evaluation["cases"]:
        metrics = [
            "-" if row[f"{metric}@{k}"] is None else f"{row[f'{metric}@{k}']:.3f}"
            for metric, k in metric_columns
        ]
        print(
            f"{row['slice']} | {row['case']} | {' | '.join(metrics)} | "
            f"{row['reciprocal_rank']:.3f}"
        )

    print("\nMacro Metrics")
    macro_groups = [("overall", evaluation["macro"])] + list(evaluation["slices"].items())
    for name, macro in macro_groups:
        print(f"{name} (relevant queries only):")
        if macro["by_k"] is None:
            print("No positive queries; relevance metrics not calculated")
        else:
            for k in k_values:
                metrics = macro["by_k"][str(k)]
                print(
                    f"K={k} | Hit@K={metrics['hit@k']:.3f} | "
                    f"Recall@K={metrics['recall@k']:.3f} | "
                    f"Precision@K={metrics['precision@k']:.3f}"
                )
            print(f"MRR={macro['mrr']:.3f}")

        no_relevance = macro["no_relevance"]
        rate = no_relevance["false_positive_rate"]
        rate_text = "not measured" if rate is None else f"{rate:.3f}"
        print(
            "No-relevance queries | "
            f"false positives={no_relevance['false_positive_count']}/"
            f"{no_relevance['valid_count']} valid "
            f"({rate_text}) | empty={no_relevance['empty_result_count']} | "
            f"errors={macro['error_count']}"
        )

    if not abstention_analysis:
        return

    print("\nAbstention Analysis")
    print("Positive relevant results")
    print("case | relevant_id | rank | score")
    for row in evaluation["cases"]:
        if row["relevant"]:
            score = row["first_relevant_score"]
            score_text = f"{score:.4f}" if score is not None else "-"
            relevant_id = row["first_relevant_research_id"] or "-"
            print(
                f"{row['case']} | {relevant_id} | "
                f"{row['first_relevant_rank'] or '-'} | {score_text}"
            )

    print("Negative top-1 results")
    print("case | top1_id | score")
    for row in evaluation["cases"]:
        if not row["relevant"]:
            score = row["top1_score"]
            score_text = f"{score:.4f}" if score is not None else "-"
            top1_id = row["top1_research_id"] or "-"
            print(
                f"{row['case']} | {top1_id} | "
                f"{score_text}"
            )

    print("Threshold Sweep (offline analysis only)")
    print("Threshold | Positive Retention | Negative Rejection")
    for result in threshold_sweep(evaluation["cases"]):
        print(
            f"{result['threshold']:.2f} | "
            f"{result['positive_retention']:.3f} | "
            f"{result['negative_rejection']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant", action="store_true", help="compare the five AE-14 retrieval modes")
    parser.add_argument("--json", action="store_true", help="emit full comparison JSON (requires --qdrant)")
    parser.add_argument("--verification-provider", choices=("fixture", "deepseek", "openai"), default="fixture")
    args = parser.parse_args()
    if args.json and not args.qdrant:
        parser.error("--json requires --qdrant")
    if args.qdrant:
        from .hybrid_retriever import KnowledgeRetriever
        from .relevance_verifier import verify_candidates
        from .relevance_verifier_eval import FixtureClient
        from ..core.providers import DeepSeekChatClient, OpenAIResponsesClient
        knowledge = KnowledgeRetriever()
        cases = list(CASES) + list(load_json("eval/rag_retrieval.json"))
        oracle = {case["query"]: set(case["relevant_ids"]) for case in cases}

        class OracleClient:
            def create(self, payload):
                body = json.loads(payload["input"])
                supported = body["research_record"]["research_id"] in oracle[body["query"]]
                return FixtureClient({"expected_supported": supported}).create(payload)

        client = OracleClient() if args.verification_provider == "fixture" else DeepSeekChatClient() if args.verification_provider == "deepseek" else OpenAIResponsesClient()
        model = "deepseek-chat" if args.verification_provider == "deepseek" else "gpt-4.1-mini"

        def verified(query, **filters):
            result = knowledge.search(query, **filters)
            tick = perf_counter()
            outcome = verify_candidates(query, result["results"], client=client, model=model)
            outcome["latency_ms"] = {**result["latency_ms"], "verification_ms": (perf_counter() - tick) * 1000}
            return outcome

        implementations = {
            "lexical": lambda query, **filters: retrieve(query, **{"include_superseded": False, **filters}),
            "bm25": lambda query, **filters: knowledge.search(query, mode="bm25", **filters),
            "dense": lambda query, **filters: knowledge.search(query, mode="dense", **filters),
            "hybrid": knowledge.search,
            "hybrid+verification": verified,
        }
        report = {"verification_provider": args.verification_provider,
                  "fixture_is_quality_measurement": False,
                  "evaluations": compare_retrievers(implementations, cases=cases)}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"Verification provider: {args.verification_provider}" +
                  (" (oracle harness, not model-quality measurement)" if args.verification_provider == "fixture" else ""))
            print_comparison(report["evaluations"])
        return
    print_report("Lexical Retrieval", run_eval())
    try:
        embedder = OllamaEmbeddingClient()
        prepared_corpus = prepare_semantic_corpus(embedder=embedder)
        semantic_eval = run_eval(
            retriever=lambda query, **filters: retrieve_semantic(
                query,
                embedder=embedder,
                prepared_corpus=prepared_corpus,
                **filters,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"\nSemantic retrieval | skipped: {exc}")
    else:
        print_report(
            "Semantic Retrieval (Ollama)",
            semantic_eval,
            abstention_analysis=True,
        )


if __name__ == "__main__":
    main()
