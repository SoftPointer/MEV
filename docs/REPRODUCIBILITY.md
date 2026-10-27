# Reproducibility and release scope

This release supports evaluating generated videos with the MEV-Eval protocol and re-scoring cached judge responses. It does not yet provide every input needed to regenerate the paper's aggregate experimental tables.

## Included

- The EEA, EOA, TGA, and BTC scoring implementation and fixed judge prompt.
- Event-duration-based temporal regions, matched to each record's source `event_ids`.
- The Gemini client with the default model `gemini-2.5-pro`, temperature `0`, and thinking budget `128`.
- Video/record matching, per-video response caches, sharding, and summary generation.
- Two example prompt records, their annotation subset, and one cached judge result.
- Fourteen unit tests using local fixtures and stub judges; no API calls are made.

## Inputs not included

| Artifact | Consequence |
|---|---|
| Original prompt-combination generation script | The example records document the evaluator input format; they do not reconstruct the paper's prompt pool. |
| Exact five 128-prompt draws and their sampling seeds | The paper's five-draw mean cannot be regenerated from the bundled examples. A new random draw is a new experiment. |
| Verified train/test UUID assignment manifest | The published split counts alone do not identify the original partitions. The Hub's table names are not train/test labels. |
| Human-validation judgments and statistics code | Majority-vote agreement, pairwise agreement, Spearman correlation, and Within-1 validation are not reproduced by this package. |
| Generation training/inference code, model weights, and full generated-video sets | This package evaluates videos supplied by the user. |

The small inspection subset on Hugging Face is also not the paper's 128-prompt evaluation set. The release makes no commitment to a future model-weight release.

## Run a comparable evaluation

Use the same prompt records and annotation revision for every method, with separate output directories. Supply `--annotation-json` so the target intervals reflect source event durations. Record the generation settings and seeds, judge model and settings, failed/skipped videos, dataset revision, and code revision.

`--annotation-json` is optional in the CLI for exploratory use. Omitting it uses equal temporal regions, which differs from the paper protocol. An explicitly supplied path that does not exist raises an error.

Keep `raw/*.mev_eval.json` with the results. Those files contain the judge response and allow deterministic metric recomputation:

```bash
cd mev_eval
python -m mev_eval.summarize --output-dir /path/to/evaluation --rescore
```

Live model responses are not guaranteed to be identical across API calls or service updates, even with temperature zero. Re-scoring an existing response tests the scoring implementation, not the repeatability of a new judge call.

## Refactor validation

During packaging, the refactored evaluator was compared with the original research scripts:

- Judge prompts and temporal regions matched exactly for 384 records.
- Re-scoring 64 stored Gemini responses preserved their metrics.
- Two reconstructed `summary_metrics.json` files matched the original files exactly.
- Fourteen unit tests passed on Python 3.8 and 3.9 without calling Gemini.

These are recorded checks from the code extraction, not a bundled 384-record benchmark or a claim that all original experiment inputs are distributed here.

The refactor also makes temperature configurable, resolves event durations using `event_ids`, keeps shard summaries separate until merging, fails on a missing explicitly supplied annotation file, and retries failed uploads with the video payload intact. The unused `--fps` and `--max-pixels` flags were removed.
