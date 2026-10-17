#!/usr/bin/env bash
# Probe refit (step 6 again): rebuild the organism and job 2/3's merged models from their adapters, then
# run configs/pilot/06_probes_refit.toml. Expects outputs/pilot/adapters/<name>/, exam-<name>/samples.jsonl,
# tasks/measure.jsonl and hua-nohint-framings.jsonl copied from the job 2 and 3 results.
# Steps are skipped when their output exists. The machine shuts down 20 minutes after the end.
set -euo pipefail
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +20' EXIT
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export HF_HOME=${HF_HOME:-$HOME/hf} PATH=$HOME/.local/bin:$PATH
P=outputs/pilot
run() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" --json-logs; }
once() { local marker=$1; shift; [ -e "$marker" ] && { echo "--- skip, $marker exists"; return; }; "$@"; }

once $P/organism/config.json run experiments/pilot/00_organism.py --config configs/pilot/00_organism.toml
for adapter in $P/adapters/*/; do
  name=$(basename "$adapter")
  once $P/grpo-$name/merged/config.json uv run python scripts/merge_adapter.py \
      $P/organism "$adapter" $P/grpo-$name/merged
done
once $P/probes-02/summary.json run experiments/pilot/06_probes.py --config configs/pilot/06_probes_refit.toml
echo "=== $(date -u +%H:%M:%S) probes 2 done"
