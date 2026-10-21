#!/usr/bin/env bash
# One matched-cue check (docs/experiments/bigger-models.md, checks) on one H100: step 1 with CONFIG.
# Builds Nemotron's matched framings (reasoning switch appended) first; cheap, unused by Qwen. Skipped
# when the output exists. The machine shuts down 20 minutes after the end, success or failure.
# Usage: run_matched_check.sh CONFIG
set -euo pipefail
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +20' EXIT
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export HF_HOME=${HF_HOME:-$HOME/hf} PATH=$HOME/.local/bin:$PATH
P=outputs/pilot
OUT=$(sed -n 's/^output_dir = "\(.*\)"/\1/p' "$1")
uv run python scripts/framing_variants.py data/framings/matched_cues.jsonl \
    $P/matched-nemotron-framings.jsonl --add-line "detailed thinking on"
[ -e "$OUT/summary.json" ] && { echo "--- skip, $OUT/summary.json exists"; exit 0; }
echo "=== $(date -u +%H:%M:%S) $1"
uv run python experiments/pilot/01_baseline.py --config "$1" --json-logs
echo "=== $(date -u +%H:%M:%S) $1 done"
