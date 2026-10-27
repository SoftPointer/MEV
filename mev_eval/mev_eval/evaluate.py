"""Judge a directory of generated videos with MEV-Eval.

    python -m mev_eval.evaluate --records-json R --video-dir V --output-dir O --annotation-json mev.json

Each video's judge response is cached in O/raw/, so interrupted runs resume and
re-running only judges new videos (use --overwrite to re-judge).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .prompt import build_prompt, duration_regions
from .records import discover_jobs, load_duration_lookup, load_records, shard_jobs
from .results import raw_result_path, result_row, write_json, write_outputs
from .scoring import score_response


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records-json", type=Path, required=True, help="Prompt records the videos were generated from.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Searched recursively for generated .mp4 files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Use one output directory per method.")
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=None,
        help="MEV annotation file (mev.json) whose event durations define the target temporal regions. "
        "Without it every event gets an equal share of the video.",
    )
    parser.add_argument("--variant-ids", type=int, nargs="+", default=None, help="Only judge these prompt variants.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Only judge these generation seeds.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Re-judge videos that already have a raw result.")
    parser.add_argument("--dry-run", action="store_true", help="List scheduled videos without calling the judge.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def judge_video(judge: Any, job: dict[str, Any], durations: dict[str, list[float]], max_new_tokens: int) -> dict[str, Any]:
    """Ask the judge about one video and return the raw result payload."""
    identity = job["identity"]
    record = job["record"]
    regions = duration_regions(record, durations)
    response = judge.generate_json([job["video_path"]], build_prompt(record, regions), max_new_tokens=max_new_tokens)
    return {
        "uuid": identity["uuid"],
        "seed": identity["seed"],
        "event_count": record["event_count"],
        "event_range": record.get("event_range"),
        "prompt_variant_id": identity["variant_id"],
        "prompt_variant_name": record.get("prompt_variant_name"),
        "video_path": str(job["video_path"]),
        "record_id": record.get("id"),
        "normalized_duration_assignment": regions,
        "metrics": score_response(response, int(record["event_count"])),
        "gemini_response": response,
    }


def run(args: argparse.Namespace, judge: Any = None) -> int:
    """Evaluate all scheduled videos. `judge` defaults to a GeminiClient."""
    if args.annotation_json is None:
        logging.warning("--annotation-json not given: target regions are uniform, unlike the paper protocol.")
    records = load_records(args.records_json)
    durations = load_duration_lookup(args.annotation_json)
    jobs = discover_jobs(
        args.video_dir,
        records,
        variant_ids=set(args.variant_ids) if args.variant_ids else None,
        seeds=set(args.seeds) if args.seeds else None,
    )
    jobs = shard_jobs(jobs, args.num_shards, args.shard_index)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    logging.info("Scheduled %d generated videos for MEV-Eval.", len(jobs))

    if args.dry_run:
        for job in jobs:
            print(job["video_path"])
        return 0

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Parallel shards share raw/ but must not overwrite each other's summaries.
    suffix = f".shard{args.shard_index}-of-{args.num_shards}" if args.num_shards > 1 else ""

    owns_judge = judge is None
    if owns_judge:
        from .gemini import GeminiClient

        judge = GeminiClient(temperature=args.temperature)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for index, job in enumerate(jobs, start=1):
            identity = job["identity"]
            raw_path = raw_result_path(raw_dir, job["video_path"])
            if raw_path.exists() and not args.overwrite:
                result = json.loads(raw_path.read_text(encoding="utf-8"))
                logging.info("[%d/%d] reuse %s", index, len(jobs), raw_path)
            else:
                logging.info(
                    "[%d/%d] judge uuid=%s variant=%s seed=%s",
                    index, len(jobs), identity["uuid"], identity["variant_id"], identity["seed"],
                )
                try:
                    result = judge_video(judge, job, durations, args.max_new_tokens)
                except Exception as exc:
                    failure = {
                        "video_path": str(job["video_path"]),
                        "uuid": identity["uuid"],
                        "seed": identity["seed"],
                        "error": repr(exc),
                    }
                    if getattr(exc, "block_reason", None):
                        failure["block_reason"] = exc.block_reason
                    failures.append(failure)
                    if not args.continue_on_error:
                        raise
                    logging.exception("Failed to evaluate %s", job["video_path"])
                    continue
                write_json(raw_path, result)

            rows.append(result_row(result))
            write_outputs(args.output_dir, rows, failures, suffix)
    finally:
        if owns_judge:
            judge.close()

    write_outputs(args.output_dir, rows, failures, suffix)
    if failures:
        logging.error("Completed with %d failures; see %s", len(failures), args.output_dir / f"failures{suffix}.json")
        return 1
    logging.info("Finished %d videos. Summary: %s", len(rows), args.output_dir / f"summary_metrics{suffix}.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
