#!/usr/bin/env bash
# Run MEV-Eval on one method's generated videos.
#
# Required env: RECORDS_JSON, VIDEO_DIR, OUTPUT_DIR, ANNOTATION_JSON, GEMINI_API_KEY
# Optional env: NUM_SHARDS (parallel judge processes, default 1), PYTHON (default python)
# Extra arguments are forwarded to `python -m mev_eval.evaluate`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
NUM_SHARDS="${NUM_SHARDS:-1}"

: "${RECORDS_JSON:?set RECORDS_JSON to the prompt records used for generation}"
: "${VIDEO_DIR:?set VIDEO_DIR to the generated-video directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for raw judgments and metrics}"
: "${ANNOTATION_JSON:?set ANNOTATION_JSON to the MEV annotation file (mev.json)}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

common_args=(
  --records-json "${RECORDS_JSON}"
  --video-dir "${VIDEO_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --annotation-json "${ANNOTATION_JSON}"
)

if [[ "${NUM_SHARDS}" -le 1 ]]; then
  exec "${PYTHON}" -m mev_eval.evaluate "${common_args[@]}" "$@"
fi

mkdir -p "${OUTPUT_DIR}"
pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  "${PYTHON}" -m mev_eval.evaluate "${common_args[@]}" \
    --num-shards "${NUM_SHARDS}" --shard-index "${shard}" "$@" \
    > "${OUTPUT_DIR}/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done

case " $* " in *" --dry-run "*) exit "${status}" ;; esac
# Merge all shards' cached results into one per_video_metrics.csv / summary_metrics.json.
"${PYTHON}" -m mev_eval.summarize --output-dir "${OUTPUT_DIR}"
exit "${status}"
