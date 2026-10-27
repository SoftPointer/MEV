"""Load MEV-Eval inputs: prompt records, event durations, and generated videos."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

# Default name written by the MEV generation scripts:
#   {uuid}_{event_range}_v{variant_id:02d}_{variant_name}_seed{seed:04d}.mp4
FILENAME_RE = re.compile(
    r"^(?P<uuid>[0-9a-fA-F-]+)_(?P<event_range>e\d+-e\d+)_v(?P<variant_id>\d+)_"
    r"(?P<variant_name>.+)_seed(?P<seed>\d+)\.mp4$"
)


def load_records(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Index prompt records by (uuid, event_range, prompt_variant_id)."""
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of prompt records in {path}")
    records = {}
    for record in payload:
        uuid = str(record["uuid"])
        event_range = str(record.get("event_range") or f"e1-e{record['event_count']}")
        variant_id = int(record["prompt_variant_id"])
        records[(uuid, event_range, variant_id)] = record
    return records


def load_duration_lookup(path: Path | None) -> dict[str, list[float]]:
    """Map uuid -> per-event durations in seconds, ordered by event_id."""
    if path is None:
        return {}
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload.get("videos", []) if isinstance(payload, dict) else payload
    lookup = {}
    for video in videos:
        uuid = str(video.get("uuid", "")).strip()
        durations = [
            float(event.get("duration", 0.0))
            for event in sorted(video.get("events", []), key=lambda item: item.get("event_id", 0))
        ]
        if uuid and durations:
            lookup[uuid] = durations
    return lookup


def parse_video_identity(video_path: Path) -> dict[str, Any] | None:
    """Recover (uuid, event_range, variant, seed) from a sidecar JSON or the filename."""
    sidecar = video_path.with_suffix(".json")
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            return {
                "uuid": str(metadata["uuid"]),
                "event_range": str(metadata.get("event_range", "")),
                "variant_id": int(metadata["prompt_variant_id"]),
                "variant_name": str(metadata.get("prompt_variant_name", "")),
                "seed": int(metadata["seed"]),
            }
        except Exception as exc:
            logging.warning("Failed to parse sidecar metadata %s: %s", sidecar, exc)

    match = FILENAME_RE.match(video_path.name)
    if not match:
        logging.warning("Cannot parse generated-video filename: %s", video_path)
        return None
    return {
        "uuid": match.group("uuid"),
        "event_range": match.group("event_range"),
        "variant_id": int(match.group("variant_id")),
        "variant_name": match.group("variant_name"),
        "seed": int(match.group("seed")),
    }


def discover_jobs(
    video_dir: Path,
    records: dict[tuple[str, str, int], dict[str, Any]],
    *,
    variant_ids: set[int] | None = None,
    seeds: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Pair every generated .mp4 under video_dir with the prompt record it was made from."""
    jobs = []
    for video_path in sorted(video_dir.expanduser().rglob("*.mp4")):
        identity = parse_video_identity(video_path)
        if identity is None:
            continue
        if variant_ids is not None and identity["variant_id"] not in variant_ids:
            continue
        if seeds is not None and identity["seed"] not in seeds:
            continue

        record = records.get((identity["uuid"], identity["event_range"], identity["variant_id"]))
        if record is None:
            # Some generation folders are renamed. Fall back to uuid+variant when unambiguous.
            candidates = [
                item
                for key, item in records.items()
                if key[0] == identity["uuid"] and key[2] == identity["variant_id"]
            ]
            if len(candidates) == 1:
                record = candidates[0]
            else:
                logging.warning("No prompt record found for %s", video_path)
                continue

        jobs.append({"video_path": video_path, "identity": identity, "record": record})
    return jobs


def shard_jobs(jobs: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return [job for idx, job in enumerate(jobs) if idx % num_shards == shard_index]
