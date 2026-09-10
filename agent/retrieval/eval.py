"""Deterministic relevance evaluation for the lexical retriever."""

from __future__ import annotations

import argparse

from ..core.resources import load_json
from .retrieval import retrieve


CASES = load_json("eval/retrieval.json")


def score_case(
    case: dict,
    result_ids: list[str],
    k_values: tuple[int, ...],
) -> dict:
    relevant_ids = set(case["relevant_ids"])
    row = {
        "case": case["id"],
        "relevant": list(case["relevant_ids"]),
        "top_results": result_ids,
    }
    first_rank = next(
        (index + 1 for index, result_id in enumerate(result_ids) if result_id in relevant_ids),
        0,
    )
    row["reciprocal_rank"] = 1 / first_rank if first_rank else 0.0

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


def run_eval(k_values: tuple[int, ...] = (1, 3, 5)) -> dict:
    k_values = tuple(sorted(set(k_values)))
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    max_k = max(k_values)
    rows = []
    for case in CASES:
        filters = {
            key: case[key]
            for key in ("market", "topic", "status")
            if key in case
        }
        results = retrieve(case["query"], limit=max_k, **filters)
        rows.append(
            score_case(
                case,
                [result["research_id"] for result in results],
                k_values,
            )
        )

    relevant_rows = [row for row in rows if row["relevant"]]
    macro_by_k = {}
    for k in k_values:
        reciprocal_ranks = [
            next(
                (
                    1 / (index + 1)
                    for index, result_id in enumerate(row["top_results"][:k])
                    if result_id in row["relevant"]
                ),
                0.0,
            )
            for row in relevant_rows
        ]
        macro_by_k[str(k)] = {
            "hit@k": sum(row[f"hit@{k}"] for row in relevant_rows) / len(relevant_rows),
            "recall@k": sum(row[f"recall@{k}"] for row in relevant_rows) / len(relevant_rows),
            "precision@k": sum(row[f"precision@{k}"] for row in relevant_rows) / len(relevant_rows),
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        }

    no_relevance_rows = [row for row in rows if not row["relevant"]]
    false_positive_count = sum(row["false_positive"] for row in no_relevance_rows)
    return {
        "k_values": list(k_values),
        "cases": rows,
        "macro": {
            "by_k": macro_by_k,
            "no_relevance": {
                "count": len(no_relevance_rows),
                "false_positive_count": false_positive_count,
                "false_positive_rate": (
                    false_positive_count / len(no_relevance_rows)
                    if no_relevance_rows else 0.0
                ),
                "empty_result_count": sum(
                    not row["top_results"] for row in no_relevance_rows
                ),
            },
        },
    }


def main() -> None:
    evaluation = run_eval()
    k_values = evaluation["k_values"]
    metric_headers = [
        f"{metric}@{k}"
        for metric in ("Hit", "Recall", "Precision")
        for k in k_values
    ]
    print("case | relevant | top results | " + " | ".join(metric_headers) + " | RR")
    for row in evaluation["cases"]:
        metrics = [
            f"{row[f'{metric}@{k}']:.3f}"
            if row[f"{metric}@{k}"] is not None else "-"
            for metric in ("hit", "recall", "precision")
            for k in k_values
        ]
        print(
            f"{row['case']} | {row['relevant']} | {row['top_results']} | "
            f"{' | '.join(metrics)} | {row['reciprocal_rank']:.3f}"
        )

    print("\nMacro metrics (relevant queries only):")
    for k in k_values:
        metrics = evaluation["macro"]["by_k"][str(k)]
        print(
            f"K={k} | Hit@K={metrics['hit@k']:.3f} | "
            f"Recall@K={metrics['recall@k']:.3f} | "
            f"Precision@K={metrics['precision@k']:.3f} | "
            f"MRR={metrics['mrr']:.3f}"
        )
    no_relevance = evaluation["macro"]["no_relevance"]
    print(
        "No-relevance queries | "
        f"false positives={no_relevance['false_positive_count']}/"
        f"{no_relevance['count']} "
        f"({no_relevance['false_positive_rate']:.3f}) | "
        f"empty={no_relevance['empty_result_count']}"
    )


if __name__ == "__main__":
    main()
