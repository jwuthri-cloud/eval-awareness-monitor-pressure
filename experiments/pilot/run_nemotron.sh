#!/usr/bin/env bash
# Hua et al.'s Nemotron-49B eval-aware organism on one H100: merge its LoRA into the base shard by
# shard, serve it in FP8, and measure the type-hint gap on (1) Hua's prompts as released, the
# check that we reproduce the organism; (2) the same without the reflection line; (3) our 150
# held-out framings with Nemotron's reasoning switch. Steps are skipped when their output exists.
# The machine shuts down 20 minutes after the end, success or failure.
set -euo pipefail
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +20' EXIT
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
export HF_HOME=${HF_HOME:-$HOME/hf} PATH=$HOME/.local/bin:$PATH
P=outputs/pilot; C=$P/nemotron-configs; mkdir -p "$C"
run() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" --json-logs; }
once() { local marker=$1; shift; [ -e "$marker" ] && { echo "--- skip, $marker exists"; return; }; "$@"; }

once $P/nemotron-organism/config.json uv run python scripts/merge_lora.py \
    nvidia/Llama-3_3-Nemotron-Super-49B-v1 387156d8d6868c19f3472fa607aa9bfc4f662333 \
    timhua/wood_v2_sftr4_filt f46fa2a76794a8ec2538f4356e4088d938222144 $P/nemotron-organism
once $P/hua-framings.jsonl uv run python scripts/hua_framings.py
once $P/hua-nohint-framings.jsonl uv run python scripts/framing_variants.py \
    $P/hua-framings.jsonl $P/hua-nohint-framings.jsonl --strip-hint
once $P/heldout-nemotron-framings.jsonl uv run python scripts/framing_variants.py \
    data/framings/heldout.jsonl $P/heldout-nemotron-framings.jsonl --add-line "detailed thinking on"

exam() {  # name framings
  sed -e "s|^output_dir = .*|output_dir = \"$P/exam-nemotron-$1\"|" -e "s|^framings = .*|framings = \"$2\"|" \
      configs/pilot/01_baseline_nemotron.toml > "$C/exam-nemotron-$1.toml"
  once $P/exam-nemotron-$1/summary.json run experiments/pilot/01_baseline.py --config "$C/exam-nemotron-$1.toml"
}
exam hua $P/hua-framings.jsonl
exam hua-nohint $P/hua-nohint-framings.jsonl
exam heldout $P/heldout-nemotron-framings.jsonl
echo "=== $(date -u +%H:%M:%S) nemotron done"
