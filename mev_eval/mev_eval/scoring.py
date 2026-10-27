"""Turn a judge response into EEA / EOA / TGA / BTC and aggregate over videos.

Every decision unit the judge omits or malforms counts as a failure: a missing
event, region, or pair is false, and a missing boundary gets the lowest score (1).
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

METRICS = ("eea", "eoa", "tga", "btc")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "correct", "executed", "grounded"}
    return False


def score_response(payload: dict[str, Any], event_count: int) -> dict[str, Any]:
    """Score one judge response for a video with `event_count` target events."""
    execution_by_id = {
        int(item.get("event_id")): bool_value(item.get("executed"))
        for item in payload.get("event_execution", [])
        if str(item.get("event_id", "")).isdigit()
    }
    grounding_by_id = {
        int(item.get("event_id")): bool_value(item.get("grounded"))
        for item in payload.get("temporal_grounding", [])
        if str(item.get("event_id", "")).isdigit()
    }
    pair_by_key = {}
    for item in payload.get("event_order", {}).get("pairwise_order", []):
        try:
            key = (int(item.get("before_event_id")), int(item.get("after_event_id")))
        except Exception:
            continue
        pair_by_key[key] = bool_value(item.get("correct"))

    boundary_scores = {}
    for item in payload.get("boundary_transition", []):
        boundary_id = str(item.get("boundary_id", "")).strip()
        try:
            score = float(item.get("score"))
        except Exception:
            continue
        boundary_scores[boundary_id] = min(5.0, max(1.0, score))

    eea_units = [execution_by_id.get(i, False) for i in range(1, event_count + 1)]
    tga_units = [grounding_by_id.get(i, False) for i in range(1, event_count + 1)]
    pair_units = [
        pair_by_key.get((i, j), False)
        for i in range(1, event_count + 1)
        for j in range(i + 1, event_count + 1)
    ]
    btc_units = [boundary_scores.get(f"{i}-{i + 1}", 1.0) for i in range(1, event_count)]

    return {
        "eea": mean(eea_units) if eea_units else 0.0,
        "eoa": mean(pair_units) if pair_units else 0.0,
        "tga": mean(tga_units) if tga_units else 0.0,
        "btc": (sum(btc_units) / (5.0 * len(btc_units))) if btc_units else 0.0,
        "executed_events": sum(eea_units),
        "grounded_events": sum(tga_units),
        "correct_order_pairs": sum(pair_units),
        "num_order_pairs": len(pair_units),
        "boundary_score_sum": sum(btc_units),
        "num_boundaries": len(btc_units),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average per-video metrics overall, per seed, and per prompt variant."""

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"num_videos": len(group_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            summary[metric] = round(mean(values), 6) if values else 0.0
        return summary

    by_seed = defaultdict(list)
    by_variant = defaultdict(list)
    for row in rows:
        by_seed[str(row["seed"])].append(row)
        by_variant[str(row["prompt_variant_id"])].append(row)

    return {
        "overall": summarize(rows),
        "by_seed": {seed: summarize(items) for seed, items in sorted(by_seed.items())},
        "by_variant_id": {
            variant_id: summarize(items) for variant_id, items in sorted(by_variant.items())
        },
    }
