# MEV-Eval

[Project overview](../README.md) · [Dataset guide](../docs/DATASET.md) · [Reproducibility notes](../docs/REPRODUCIBILITY.md)

MEV-Eval is the event-centric evaluation protocol for long-take multi-event video generation introduced in
*MEV: A Multi-Event Video Dataset for Long-Take Generation*.

General video-quality metrics such as VBench do not check whether a generated video actually carries out a given
sequence of events. MEV-Eval asks a VLM judge (Gemini 2.5 Pro by default) to check each generated video against
its target event sequence. The judge sees the generated video, the global prompt, the ordered event captions, and the
temporal region assigned to each event. It then scores four failure modes: omitted events, wrong order, poor temporal
localization, and incoherent transitions.

## Metrics

For a video with target events $e_1, \dots, e_N$:

| Metric | Decision unit (judged per video) | Score |
|---|---|---|
| **EEA**, Event Execution Accuracy | Each event $e_i$: is it visually and semantically executed (relevant action and key subject/object present)? | $\frac{1}{N}\sum_i x_i$ |
| **EOA**, Event Order Accuracy | Each pair $i<j$: does $e_i$ appear before $e_j$? A pair is wrong if either event is missing. | $\frac{1}{\binom{N}{2}}\sum_{i<j} \mathbf{1}[e_i \prec e_j]$ |
| **TGA**, Temporal Grounding Accuracy | Each target region $\tau_i$: does it mainly depict $e_i$? | $\frac{1}{N}\sum_i g_i$ |
| **BTC**, Boundary Transition Coherence | Each adjacent boundary: a 1–5 rating $b_i$ | $\frac{1}{5(N-1)}\sum_i b_i$ |

All four scores lie in $[0, 1]$, where higher is better, and are averaged over videos.

- **BTC rubric.** The judge rates semantic continuity, subject identity stability, object-state consistency, and
  scene/camera continuity, and looks for severe drift, stagnation, or object splitting. A score of 1 means broken,
  3 means partially coherent, and 5 means smooth and logically coherent.
- **EEA versus TGA.** A video can execute every event and still score low on TGA if the events appear in the wrong
  regions.
- **Strict scoring.** Any event, pair, or region that the judge omits or returns malformed counts as a failure. A
  missing boundary gets the minimum rating of 1. Ratings outside 1–5 are clipped. See
  [mev_eval/scoring.py](mev_eval/scoring.py).

**Target temporal regions.** The regions come from MEV's human-verified event boundaries. Each event's region is the
fraction of the video proportional to that event's duration in the source clip, so all compared methods are judged
against the same targets. For example, for the 4-event record in [examples/](examples/), the source events last
3.46 s, 5.56 s, 12.91 s and 6.02 s. That gives these regions:

| Event | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Region | [0, 0.124) | [0.124, 0.323) | [0.323, 0.785) | [0.785, 1.0] |

## Installation

```bash
cd mev_eval
pip install -r requirements.txt          # Python >= 3.8; the only dependency is requests
export GEMINI_API_KEY='your-api-key'     # or GOOGLE_API_KEY; only needed for live judging
```

## Inputs

**1. Prompt records** (`--records-json`). This is a JSON list with one record per generation prompt; see
[examples/records.json](examples/records.json). These are the fields used:

| Field | Meaning |
|---|---|
| `uuid` | MEV source video id |
| `event_count`, `event_range`, `event_ids` | Number of events, e.g. `"e1-e4"`, and their ids in the source video |
| `prompt_variant_id`, `prompt_variant_name` | Which prompt formulation was used for generation |
| `global_caption`, `event_captions` | Global description and the ordered event captions |
| `prompt` (or `text`) | The full prompt given to the generator |

**2. Generated videos** (`--video-dir`). The directory is searched recursively for `.mp4` files. Each video is
matched to its record by `(uuid, event_range, prompt_variant_id)`, which is read from one of two places:

- A sidecar `<video>.json` next to the video, with `uuid`, `event_range`, `prompt_variant_id` and `seed`.
  This takes precedence.
- The filename `{uuid}_{event_range}_v{variant_id:02d}_{variant_name}_seed{seed:04d}.mp4`, as written by the MEV
  generation scripts.

Videos that match neither form are skipped with a warning. Use sidecars for methods whose outputs follow another
naming scheme.

**3. MEV annotations** (`--annotation-json`). This is `annotations/mev.json` from the
[MEV dataset](https://huggingface.co/datasets/MultiEventVideo/MEV), which supplies the event durations behind the
target regions. Only `videos[].uuid` and `videos[].events[].{event_id, duration}` are read. Without this file
every event gets an equal share of the video, which does **not** match the paper protocol.

## Usage

Evaluate one method's videos:

```bash
python -m mev_eval.evaluate \
  --records-json  data/mev_4_event_prompts.json \
  --video-dir     results/mev_wan \
  --output-dir    outputs/mev_eval/mev_wan \
  --annotation-json data/mev.json
```

The launcher runs several judge processes in parallel and merges their results:

```bash
RECORDS_JSON=data/mev_4_event_prompts.json \
VIDEO_DIR=results/mev_wan \
OUTPUT_DIR=outputs/mev_eval/mev_wan \
ANNOTATION_JSON=data/mev.json \
NUM_SHARDS=4 \
bash scripts/run_mev_eval.sh --continue-on-error
```

To compare methods, evaluate each method's video directory with the same records and annotation file, and use a
separate `--output-dir` for each method.

| Option | Effect |
|---|---|
| `--variant-ids 5`, `--seeds 0 1` | Only judge these prompt variants / generation seeds |
| `--limit N` | Judge at most N videos |
| `--dry-run` | List matched videos without calling the judge |
| `--overwrite` | Re-judge videos that already have a cached result |
| `--continue-on-error` | Record failures in `failures.json` and keep going |

**Resuming and re-scoring.** Each video's judge response is cached in `raw/`, so an interrupted run resumes where it
stopped. The following command rebuilds the tables from the cache without calling the judge. With `--rescore`, it
also recomputes every metric from the stored responses.

```bash
python -m mev_eval.summarize --output-dir outputs/mev_eval/mev_wan --rescore
```

Omit `--rescore` to aggregate the metrics already stored in the cache.

For a self-contained offline example, run from this `mev_eval/` directory:

```bash
mkdir -p outputs/example/raw
cp examples/raw_result.mev_eval.json outputs/example/raw/
python -m mev_eval.summarize --output-dir outputs/example --rescore
```

This uses one cached response and requires no API key or video download. It is a usage example, not a reproduction
of the paper's evaluation set; see [reproducibility notes](../docs/REPRODUCIBILITY.md).

## Outputs

```text
<output-dir>/
├── raw/<video>.mev_eval.json   judge response, target regions and metrics for one video
├── per_video_metrics.csv       one row per video
├── summary_metrics.json        mean EEA/EOA/TGA/BTC: overall, by_seed, by_variant_id
└── failures.json               only written if some videos failed
```

Sharded runs write `*.shard{i}-of-{n}.*` files per shard. `scripts/run_mev_eval.sh` merges them into the unsuffixed
files with `mev_eval.summarize`. See [examples/raw_result.mev_eval.json](examples/raw_result.mev_eval.json) for a
complete raw result.

## Judge configuration

The defaults reproduce the paper's judging setup:

- Gemini 2.5 Pro, temperature 0, thinking budget 128, JSON response MIME type.
- The fixed system instruction in [mev_eval/gemini.py](mev_eval/gemini.py).
- Each video is uploaded through the Gemini Files API and deleted when the run ends.
- If the judge returns invalid JSON, it is re-prompted deterministically, up to 3 attempts. A video blocked by the
  safety filter is recorded as a failure.

| Environment variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | (required) | API key |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Judge model |
| `GEMINI_THINKING_BUDGET` | `128` for 2.5 Pro | Thinking-token budget |
| `GEMINI_SYSTEM_INSTRUCTION` | built-in | Override the system instruction |
| `GEMINI_JSON_RETRY_COUNT` | `3` | Attempts to obtain valid JSON |
| `GEMINI_HTTP_RETRY_COUNT` | `32` | Retries on network errors, 429 and 5xx |
| `GEMINI_HTTP_TIMEOUT` | `600` | Per-request timeout in seconds |
| `GEMINI_CLEANUP_FILES` | `1` | Delete uploaded videos after the run |
| `GEMINI_API_BASE_URL` | Google endpoint | Alternative API endpoint |

**Other judges.** `mev_eval.evaluate.run(args, judge=...)` accepts any object that provides
`generate_json(video_paths, prompt, max_new_tokens) -> dict` and returns the JSON schema requested in
[mev_eval/prompt.py](mev_eval/prompt.py).

## Tests

```bash
python -m unittest discover tests
```

The tests cover scoring, target regions, video matching, sharding and an end-to-end run with a stub judge. They make
no API calls.

## Layout

```text
mev_eval/
├── mev_eval/
│   ├── evaluate.py    CLI: match videos to records, query the judge, cache and score
│   ├── summarize.py   CLI: rebuild tables from cached results, optionally re-score
│   ├── prompt.py      target temporal regions and the judge prompt (rubric + output schema)
│   ├── scoring.py     EEA / EOA / TGA / BTC from a judge response, aggregation
│   ├── records.py     prompt records, MEV durations, video discovery, sharding
│   ├── results.py     raw-result cache, CSV and summary writers
│   └── gemini.py      Gemini REST client (Files API upload, retries, JSON parsing)
├── scripts/run_mev_eval.sh
├── examples/          two prompt records, their MEV annotations, one raw judge result
└── tests/
```
