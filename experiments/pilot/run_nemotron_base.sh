#!/usr/bin/env bash
# The control for run_nemotron.sh: the same three exams on base Nemotron-49B without the organism
# LoRA, on its own H100 (vLLM downloads the base at the pinned revision). It builds the same
# framing files, so it runs on a fresh machine or after run_nemotron.sh on the same one; there it
# waits for that job and cancels its shutdown. Steps are skipped when their output exists. The
# machine shuts down 20 minutes after the end, success or failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
while tmux has-session -t nemotron 2>/dev/null; do sleep 30; done
sudo shutdown -c 2>/dev/null || true
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +20' EXIT
set -a; source .env; set +a
export HF_HOME=${HF_HOME:-$HOME/hf} PATH=$HOME/.local/bin:$PATH
P=outputs/pilot; C=$P/nemotron-configs; mkdir -p "$C"
run() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" --json-logs; }
once() { local marker=$1; shift; [ -e "$marker" ] && { echo "--- skip, $marker exists"; return; }; "$@"; }

once $P/hua-framings.jsonl uv run python scripts/hua_framings.py
once $P/hua-nohint-framings.jsonl uv run python scripts/framing_variants.py \
    $P/hua-framings.jsonl $P/hua-nohint-framings.jsonl --strip-hint
once $P/heldout-nemotron-framings.jsonl uv run python scripts/framing_variants.py \
    data/framings/heldout.jsonl $P/heldout-nemotron-framings.jsonl --add-line "detailed thinking on"

exam() {  # name framings
  sed -e "s|^output_dir = .*|output_dir = \"$P/exam-nemotron-base-$1\"|" -e "s|^framings = .*|framings = \"$2\"|" \
      configs/pilot/01_baseline_nemotron_base.toml > "$C/exam-nemotron-base-$1.toml"
  once $P/exam-nemotron-base-$1/summary.json run experiments/pilot/01_baseline.py --config "$C/exam-nemotron-base-$1.toml"
}
exam hua $P/hua-framings.jsonl
exam hua-nohint $P/hua-nohint-framings.jsonl
exam heldout $P/heldout-nemotron-framings.jsonl
echo "=== $(date -u +%H:%M:%S) nemotron base done"
