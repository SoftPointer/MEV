"""Rebuild per_video_metrics.csv and summary_metrics.json from cached judge results.

    python -m mev_eval.summarize --output-dir O [--rescore]

Use after sharded runs, or with --rescore to recompute metrics from the stored
judge responses without calling the judge again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .results import load_raw_results, result_row, write_outputs
from .scoring import score_response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing raw/*.mev_eval.json.")
    parser.add_argument("--rescore", action="store_true", help="Recompute metrics from the stored judge responses.")
    args = parser.parse_args(argv)

    results = load_raw_results(args.output_dir / "raw")
    if not results:
        print(f"No raw results found under {args.output_dir / 'raw'}", file=sys.stderr)
        return 1
    if args.rescore:
        for result in results:
            result["metrics"] = score_response(result["gemini_response"], int(result["event_count"]))

    rows = [result_row(result) for result in results]
    write_outputs(args.output_dir, rows, failures=[])
    summary = json.loads((args.output_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
