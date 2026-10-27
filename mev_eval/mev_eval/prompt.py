"""Build the MEV-Eval judge prompt for one generated video."""
from __future__ import annotations

import json
import logging
from typing import Any


def duration_regions(record: dict[str, Any], duration_lookup: dict[str, list[float]]) -> list[dict[str, Any]]:
    """Split the video into normalized [start, end) regions proportional to the source event durations.

    Durations come from MEV's human-verified event boundaries. When they are unavailable
    every event gets an equal share of the video.
    """
    event_count = int(record["event_count"])
    all_durations = duration_lookup.get(str(record["uuid"]), [])
    event_ids = [int(event_id) for event_id in record.get("event_ids") or range(1, event_count + 1)]
    if len(event_ids) == event_count and all(1 <= event_id <= len(all_durations) for event_id in event_ids):
        durations = [all_durations[event_id - 1] for event_id in event_ids]
    else:
        durations = all_durations[:event_count]
    if len(durations) != event_count or sum(durations) <= 0:
        if duration_lookup:
            logging.warning("No usable event durations for uuid=%s; using uniform regions.", record["uuid"])
        durations = [1.0] * event_count

    total = sum(durations)
    cursor = 0.0
    regions = []
    for idx, duration in enumerate(durations, start=1):
        start = cursor / total
        cursor += duration
        end = cursor / total
        regions.append(
            {
                "event_id": idx,
                "start_ratio": round(start, 4),
                "end_ratio": round(end, 4),
                "duration_weight": round(duration / total, 4),
            }
        )
    return regions


def build_prompt(record: dict[str, Any], regions: list[dict[str, Any]]) -> str:
    """Return the judge instruction: rubric, output schema, and the target event payload."""
    events = [
        {"event_id": idx, "caption": caption}
        for idx, caption in enumerate(record.get("event_captions", []), start=1)
    ]
    event_count = int(record["event_count"])
    expected_pairs = [
        {"before_event_id": i, "after_event_id": j}
        for i in range(1, event_count + 1)
        for j in range(i + 1, event_count + 1)
    ]
    expected_boundaries = [
        {"boundary_id": f"{i}-{i + 1}", "before_event_id": i, "after_event_id": i + 1}
        for i in range(1, event_count)
    ]

    task_payload = {
        "global_caption": record.get("global_caption", ""),
        "input_prompt_used_for_generation": record.get("text") or record.get("prompt", ""),
        "events": events,
        "normalized_duration_assignment": regions,
        "expected_pairwise_order": expected_pairs,
        "expected_adjacent_boundaries": expected_boundaries,
    }

    return (
        "You are evaluating one generated video with the MEV-Eval protocol for "
        "long-take multi-event video generation.\n\n"
        "The video was generated from the provided prompt and ordered event sequence. "
        "Judge only the generated video against this target sequence. Ignore aesthetic quality "
        "unless it prevents event recognition.\n\n"
        "Metric definitions:\n"
        "- Event Execution Accuracy (EEA): for each event, executed=true only if the relevant "
        "action and key subject/object are visually and semantically present. Missing, wrong, "
        "or only vaguely implied events are false.\n"
        "- Event Order Accuracy (EOA): for each expected event pair i<j, correct=true only if "
        "event i appears before event j. If either event is missing or cannot be localized, "
        "that pair is false.\n"
        "- Temporal Grounding Accuracy (TGA): for each normalized temporal region, grounded=true "
        "only if that region mainly depicts the corresponding target event.\n"
        "- Boundary Transition Coherence (BTC): for each adjacent event boundary, rate 1-5 for "
        "semantic continuity, subject identity stability, object-state consistency, scene/camera "
        "continuity, and absence of severe drift, stagnation, or object splitting. 1 is broken, "
        "3 is partially coherent, 5 is smooth and logically coherent.\n\n"
        "Return strict JSON only with exactly this schema:\n"
        "{\n"
        '  "event_execution": [\n'
        '    {"event_id": 1, "executed": true, "evidence": "short visual evidence"}\n'
        "  ],\n"
        '  "event_order": {\n'
        '    "observed_event_order": [1, 2],\n'
        '    "pairwise_order": [\n'
        '      {"before_event_id": 1, "after_event_id": 2, "correct": true, "evidence": "short evidence"}\n'
        "    ]\n"
        "  },\n"
        '  "temporal_grounding": [\n'
        '    {"event_id": 1, "expected_start_ratio": 0.0, "expected_end_ratio": 0.5, "grounded": true, "evidence": "short evidence"}\n'
        "  ],\n"
        '  "boundary_transition": [\n'
        '    {"boundary_id": "1-2", "score": 4, "evidence": "short evidence"}\n'
        "  ],\n"
        '  "overall_notes": "one concise sentence"\n'
        "}\n\n"
        "Target evaluation payload:\n"
        f"{json.dumps(task_payload, ensure_ascii=False, indent=2)}\n"
    )
