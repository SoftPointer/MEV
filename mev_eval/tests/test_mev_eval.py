from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from mev_eval import evaluate, summarize
from mev_eval.prompt import build_prompt, duration_regions
from mev_eval.records import discover_jobs, load_records, parse_video_identity, shard_jobs
from mev_eval.scoring import aggregate, score_response

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HAS_REQUESTS = importlib.util.find_spec("requests") is not None


def perfect_response(event_count: int) -> dict:
    return {
        "event_execution": [{"event_id": i, "executed": True} for i in range(1, event_count + 1)],
        "event_order": {
            "pairwise_order": [
                {"before_event_id": i, "after_event_id": j, "correct": True}
                for i in range(1, event_count + 1)
                for j in range(i + 1, event_count + 1)
            ]
        },
        "temporal_grounding": [{"event_id": i, "grounded": True} for i in range(1, event_count + 1)],
        "boundary_transition": [{"boundary_id": f"{i}-{i + 1}", "score": 5} for i in range(1, event_count)],
    }


class ScoringTest(unittest.TestCase):
    def test_perfect_response_scores_one(self):
        metrics = score_response(perfect_response(4), 4)
        for metric in ("eea", "eoa", "tga", "btc"):
            self.assertEqual(metrics[metric], 1.0)
        self.assertEqual(metrics["num_order_pairs"], 6)
        self.assertEqual(metrics["num_boundaries"], 3)

    def test_missing_units_count_as_failures(self):
        payload = {
            "event_execution": [{"event_id": 1, "executed": "yes"}, {"event_id": "2", "executed": "false"}],
            "event_order": {"pairwise_order": [{"before_event_id": 1, "after_event_id": 2, "correct": 1}]},
            "temporal_grounding": [{"event_id": 3, "grounded": True}],
            "boundary_transition": [{"boundary_id": "1-2", "score": 9}],
        }
        metrics = score_response(payload, 3)
        self.assertAlmostEqual(metrics["eea"], 1 / 3)
        self.assertAlmostEqual(metrics["eoa"], 1 / 3)
        self.assertAlmostEqual(metrics["tga"], 1 / 3)
        # Boundary 1-2 is clamped to 5; missing boundary 2-3 gets the minimum score 1.
        self.assertAlmostEqual(metrics["btc"], (5 + 1) / 10)

    def test_stored_example_rescores_identically(self):
        result = json.loads((EXAMPLES / "raw_result.mev_eval.json").read_text(encoding="utf-8"))
        self.assertEqual(score_response(result["gemini_response"], result["event_count"]), result["metrics"])

    def test_aggregate_groups_by_seed_and_variant(self):
        rows = [
            {"seed": 0, "prompt_variant_id": 5, "eea": 1.0, "eoa": 0.5, "tga": 0.0, "btc": 0.2},
            {"seed": 1, "prompt_variant_id": 5, "eea": 0.0, "eoa": 0.5, "tga": 1.0, "btc": 0.6},
        ]
        summary = aggregate(rows)
        self.assertEqual(summary["overall"], {"num_videos": 2, "eea": 0.5, "eoa": 0.5, "tga": 0.5, "btc": 0.4})
        self.assertEqual(sorted(summary["by_seed"]), ["0", "1"])
        self.assertEqual(summary["by_variant_id"]["5"]["num_videos"], 2)


class PromptTest(unittest.TestCase):
    def test_regions_follow_source_durations(self):
        record = {"uuid": "u", "event_count": 2, "event_ids": [1, 2]}
        regions = duration_regions(record, {"u": [1.0, 3.0, 4.0]})
        self.assertEqual([(r["start_ratio"], r["end_ratio"]) for r in regions], [(0.0, 0.25), (0.25, 1.0)])

    def test_regions_select_window_by_event_ids(self):
        record = {"uuid": "u", "event_count": 2, "event_ids": [2, 3]}
        regions = duration_regions(record, {"u": [1.0, 3.0, 1.0]})
        self.assertEqual(regions[0]["end_ratio"], 0.75)

    def test_regions_fall_back_to_uniform(self):
        regions = duration_regions({"uuid": "u", "event_count": 4}, {})
        self.assertEqual([r["duration_weight"] for r in regions], [0.25] * 4)

    def test_prompt_lists_all_pairs_and_boundaries(self):
        record = json.loads((EXAMPLES / "records.json").read_text(encoding="utf-8"))[1]
        prompt = build_prompt(record, duration_regions(record, {}))
        payload = json.loads(prompt.split("Target evaluation payload:\n", 1)[1])
        self.assertEqual(len(payload["expected_pairwise_order"]), 6)
        self.assertEqual([b["boundary_id"] for b in payload["expected_adjacent_boundaries"]], ["1-2", "2-3", "3-4"])
        self.assertEqual(payload["input_prompt_used_for_generation"], record["prompt"])


class RecordsTest(unittest.TestCase):
    def test_parse_identity_from_filename(self):
        identity = parse_video_identity(Path("abc-123_e1-e4_v05_structured_global_middle_plus_events_seed0100.mp4"))
        self.assertEqual(
            identity,
            {
                "uuid": "abc-123",
                "event_range": "e1-e4",
                "variant_id": 5,
                "variant_name": "structured_global_middle_plus_events",
                "seed": 100,
            },
        )

    def test_sidecar_overrides_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "renamed.mp4"
            video.write_bytes(b"")
            video.with_suffix(".json").write_text(
                json.dumps({"uuid": "u1", "event_range": "e1-e2", "prompt_variant_id": 3, "seed": 7}),
                encoding="utf-8",
            )
            self.assertEqual(parse_video_identity(video)["uuid"], "u1")

    def test_discover_pairs_videos_with_records(self):
        records = load_records(EXAMPLES / "records.json")
        with tempfile.TemporaryDirectory() as tmp:
            video_dir = Path(tmp)
            make_videos(video_dir, records.values(), seeds=(0, 1))
            (video_dir / "unrelated.mp4").write_bytes(b"")
            jobs = discover_jobs(video_dir, records)
            self.assertEqual(len(jobs), 4)
            self.assertEqual(len(discover_jobs(video_dir, records, seeds={1})), 2)
            self.assertEqual(len(shard_jobs(jobs, 3, 0)) + len(shard_jobs(jobs, 3, 1)) + len(shard_jobs(jobs, 3, 2)), 4)


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def generate_json(self, video_paths, prompt, max_new_tokens):
        self.calls += 1
        payload = json.loads(prompt.split("Target evaluation payload:\n", 1)[1])
        return perfect_response(len(payload["events"]))


class EvaluateTest(unittest.TestCase):
    def test_end_to_end_with_cache_and_summarize(self):
        records = load_records(EXAMPLES / "records.json")
        with tempfile.TemporaryDirectory() as tmp:
            video_dir, output_dir = Path(tmp) / "videos", Path(tmp) / "out"
            make_videos(video_dir, records.values(), seeds=(0,))
            args = evaluate.parse_args(
                [
                    "--records-json", str(EXAMPLES / "records.json"),
                    "--video-dir", str(video_dir),
                    "--output-dir", str(output_dir),
                    "--annotation-json", str(EXAMPLES / "annotations.json"),
                ]
            )
            judge = FakeJudge()
            self.assertEqual(evaluate.run(args, judge=judge), 0)
            self.assertEqual(judge.calls, 2)
            summary = json.loads((output_dir / "summary_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["overall"]["num_videos"], 2)
            self.assertEqual(summary["overall"]["btc"], 1.0)

            raw = json.loads(next((output_dir / "raw").glob("*_e1-e4_*.mev_eval.json")).read_text(encoding="utf-8"))
            self.assertNotEqual(raw["normalized_duration_assignment"][0]["duration_weight"], 0.25)

            self.assertEqual(evaluate.run(args, judge=judge), 0)
            self.assertEqual(judge.calls, 2, "cached raw results must be reused")

            (output_dir / "summary_metrics.json").unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(summarize.main(["--output-dir", str(output_dir), "--rescore"]), 0)
            self.assertEqual(json.loads((output_dir / "summary_metrics.json").read_text(encoding="utf-8")), summary)

    def test_shards_write_separate_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_dir, output_dir = Path(tmp) / "videos", Path(tmp) / "out"
            make_videos(video_dir, load_records(EXAMPLES / "records.json").values(), seeds=(0,))
            for shard in range(2):
                args = evaluate.parse_args(
                    [
                        "--records-json", str(EXAMPLES / "records.json"),
                        "--video-dir", str(video_dir),
                        "--output-dir", str(output_dir),
                        "--num-shards", "2",
                        "--shard-index", str(shard),
                    ]
                )
                self.assertEqual(evaluate.run(args, judge=FakeJudge()), 0)
            self.assertTrue((output_dir / "summary_metrics.shard0-of-2.json").exists())
            self.assertTrue((output_dir / "summary_metrics.shard1-of-2.json").exists())
            self.assertEqual(len(list((output_dir / "raw").glob("*.mev_eval.json"))), 2)


@unittest.skipUnless(HAS_REQUESTS, "requests is not installed")
class GeminiParsingTest(unittest.TestCase):
    def test_extracts_json_from_fenced_or_noisy_text(self):
        from mev_eval.gemini import parse_json_payload

        self.assertEqual(parse_json_payload('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_payload('Here you go: {"a": "x}"} trailing'), {"a": "x}"})
        self.assertEqual(parse_json_payload('[{"a": 1}]'), {"a": 1})


def make_videos(video_dir: Path, records, seeds) -> None:
    for record in records:
        for seed in seeds:
            name = (
                f"{record['uuid']}_{record['event_range']}_v{record['prompt_variant_id']:02d}_"
                f"{record['prompt_variant_name']}_seed{seed:04d}.mp4"
            )
            path = video_dir / record["uuid"] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")


if __name__ == "__main__":
    unittest.main()
