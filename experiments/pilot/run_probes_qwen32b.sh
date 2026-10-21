#!/usr/bin/env bash
# Step 6 on Qwen3-32B (configs/pilot/06_probes_qwen32b.toml) on one H100. Expects tasks/measure.jsonl and
# the two matched-cue exams (exam-qwen32b-matched, exam-qwen32b-matched-heldout) staged in outputs/pilot.
# The machine shuts down 20 minutes after the end, success or failure.
set -euo pipefail
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +20' EXIT
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export HF_HOME=${HF_HOME:-$HOME/hf} PATH=$HOME/.local/bin:$PATH
P=outputs/pilot
[ -e $P/probes-qwen32b/summary.json ] && { echo "--- skip, probes exist"; exit 0; }
echo "=== $(date -u +%H:%M:%S) probes qwen32b"
uv run python experiments/pilot/06_probes.py --config configs/pilot/06_probes_qwen32b.toml --json-logs
echo "=== $(date -u +%H:%M:%S) probes qwen32b done"
