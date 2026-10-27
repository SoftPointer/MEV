"""Write and reload MEV-Eval outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .scoring import METRICS, aggregate

RAW_SUFFIX = ".mev_eval.json"

ROW_FIELDS = (
    "uuid",
    "seed",
    "event_count",
    "event_range",
    "prompt_variant_id",
    "prompt_variant_name",
    "video_path",
)
CSV_FIELDS = (
    *ROW_FIELDS,
    *METRICS,
    "executed_events",
    "grounded_events",
    "correct_order_pairs",
    "num_order_pairs",
    "boundary_score_sum",
    "num_boundaries",
)


def raw_result_path(raw_dir: Path, video_path: Path) -> Path:
    return raw_dir / video_path.with_suffix(RAW_SUFFIX).name


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw result into a per-video CSV row."""
    row = {key: result.get(key) for key in ROW_FIELDS}
    row.update(result["metrics"])
    return row


def load_raw_results(raw_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(raw_dir.glob(f"*{RAW_SUFFIX}"))
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    suffix: str = "",
) -> None:
    """Write per_video_metrics, summary_metrics, and (if any) failures for one run or shard."""
    write_csv(output_dir / f"per_video_metrics{suffix}.csv", rows)
    write_json(output_dir / f"summary_metrics{suffix}.json", aggregate(rows))
    if failures:
        write_json(output_dir / f"failures{suffix}.json", failures)
