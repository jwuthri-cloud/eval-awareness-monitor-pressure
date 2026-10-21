#!/usr/bin/env bash
# Qwen3-32B, a model never trained to be eval-aware, on one H100: step 1 over our 150 held-out
# framings with 4 answers per prompt. Further behaviors (tests passed, docstrings, length) are
# scored offline from samples.jsonl. The machine shuts down 20 minutes after the end.
set -euo pipefail
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +20' EXIT
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export HF_HOME=${HF_HOME:-$HOME/hf} PATH=$HOME/.local/bin:$PATH
P=outputs/pilot
run() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" --json-logs; }
once() { local marker=$1; shift; [ -e "$marker" ] && { echo "--- skip, $marker exists"; return; }; "$@"; }

once $P/exam-qwen32b-heldout/summary.json run experiments/pilot/01_baseline.py --config configs/pilot/01_baseline_qwen32b.toml
echo "=== $(date -u +%H:%M:%S) qwen32b done"
