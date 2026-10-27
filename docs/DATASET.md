# Using the MEV dataset

The official data release is [MultiEventVideo/MEV on Hugging Face](https://huggingface.co/datasets/MultiEventVideo/MEV). The GitHub repository contains the evaluator and small examples; full annotations, metadata tables, and video segments are downloaded separately.

## Download

Run from the GitHub repository root, with Python 3.10+ for current Hugging Face tooling:

```bash
python -m pip install -U huggingface_hub
```

Choose the files needed for your task:

```bash
# Annotations for MEV-Eval; no source videos are needed to judge generated videos.
hf download MultiEventVideo/MEV annotations/mev.json LICENSE \
  --repo-type dataset --local-dir data/MEV

# Smaller sample package: annotations, metadata, and video segments.
hf download MultiEventVideo/MEV --repo-type dataset \
  --include "samples/*" --local-dir data/MEV

# Full release, including all video shards. Check available disk space first.
hf download MultiEventVideo/MEV --repo-type dataset --local-dir data/MEV
```

The full release contains tens of gigabytes of video. The sample package is intended for inspection, not for reproducing a paper benchmark. Downloading original MEV clips is separate from providing the generated videos evaluated by MEV-Eval.

For a reproducible experiment, add `--revision` with the dataset commit hash you used, and record that hash alongside your results. See the [official download CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download).

## Files on Hugging Face

| Path | Contents |
|---|---|
| `annotations/mev.json` | Canonical nested annotation file, with original videos and their events |
| `metadata/events.parquet` | One row per event; captions, timestamps, attributes, quality scores, and media pointers |
| `metadata/videos.parquet` | One row per original-video UUID |
| `metadata/events.csv`, `mev.csv` | CSV mirrors of the event table |
| `videos/mev-*.tar` | Video segment shards; tar members retain the original segment filenames |
| `samples/annotations/sample_mev.json` | Nested annotations for the sample subset |
| `samples/metadata/` | Sample event and video tables |
| `samples/videos/sample-000000.tar` | Sample video segments |
| `LICENSE` | Dataset license and source-specific terms |

The canonical JSON is the file expected by `mev_eval.evaluate --annotation-json`. Do not pass the Parquet table or prompt-record file in its place.

## Load the event table

```bash
python -m pip install -U datasets
```

```python
from datasets import load_dataset

# Start with the small table. This loads metadata, not the video tar archive.
events = load_dataset("MultiEventVideo/MEV", split="sample_events")
event = events[0]
print(event["uuid"], event["event_id"])
print(event["start_time"], event["end_time"], event["event_caption"])
print(event["media_ref"])

# Full tables, when needed:
# events = load_dataset("MultiEventVideo/MEV", split="events")
# videos = load_dataset("MultiEventVideo/MEV", split="videos")
```

**The Hub split names `events`, `videos`, `sample_events`, and `sample_videos` select tables. They are not the paper's training/test partitions.**

Each event row provides `shard_path` and `tar_member`. Their combined `media_ref` has the form `videos/mev-000000.tar::<segment-filename>.mp4`; it is a locator within a tar archive, not a standalone video URL.

To read one sample segment without extracting the whole archive:

```python
import tarfile
from huggingface_hub import hf_hub_download

# Reuses the sample event selected above; downloading a shard retrieves all
# segments in that shard, even if only one segment is read below.
archive = hf_hub_download(
    repo_id="MultiEventVideo/MEV",
    repo_type="dataset",
    filename=event["shard_path"],
)
with tarfile.open(archive, "r") as shard:
    segment = shard.extractfile(event["tar_member"])
    if segment is None:
        raise ValueError("The requested tar member is not a regular file")
    with segment:
        video_bytes = segment.read()
print(len(video_bytes), "video bytes")
```

## Read nested annotations

```python
import json
from pathlib import Path

annotations = json.loads(
    Path("data/MEV/annotations/mev.json").read_text(encoding="utf-8")
)
video = annotations["videos"][0]
print(video["uuid"])
for event in video["events"]:
    print(event["event_id"], event["duration"])
```

MEV-Eval reads the UUID, event IDs, and event durations from this structure. Its separate [prompt records](../mev_eval/examples/records.json) specify the event window and captions used for generation. The two bundled records and their [annotation subset](../mev_eval/examples/annotations.json) demonstrate that interface.

## Training/test protocol

The paper partitions original videos before constructing contiguous event groups. Every event and event group from a given UUID stays in the same partition.

| Paper partition | Original videos | Events | Sum of annotated event durations |
|---|---:|---:|---|
| Train | 6,413 | 20,241 | 29:07:24.974 |
| Test | 1,603 | 5,021 | 07:48:42.790 |
| Total | 8,016 | 25,262 | 36:56:07.764 |

These are paper statistics, not instructions to construct a new random split. The release currently has no verified standalone UUID partition manifest; see [reproducibility notes](REPRODUCIBILITY.md) before claiming an exact replication. If creating a new split for your own experiment, split by original-video UUID and publish the resulting assignments and seed.

## Terms

MEV-created annotations and metadata use CC BY 4.0. Videos retain their source-specific terms. Follow the [official dataset license](https://huggingface.co/datasets/MultiEventVideo/MEV/blob/main/LICENSE), preserve available provenance, and consult [DATA_LICENSE.md](../DATA_LICENSE.md).
