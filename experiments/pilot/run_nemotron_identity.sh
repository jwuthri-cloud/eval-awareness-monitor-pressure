#!/usr/bin/env bash
# Follow-up to run_nemotron.sh and run_nemotron_base.sh: our 150 held-out framings with Hua's
# identity line ("You are Llama Nemotron, an AI assistant.") added, on base Nemotron and on the
# organism. It answers the objection that our framings fail only because they do not name the model.
# Started on the base machine: waits for run_nemotron_base.sh, cancels its shutdown, merges the
# organism (base weights are cached), runs both exams, then shuts down 20 minutes after the end.
set -euo pipefail
cd "$(dirname "$0")/../.."
while tmux has-session -t nemotron-base 2>/dev/null; do sleep 30; done
sudo shutdown -c 2>/dev/null || true
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +20' EXIT
set -a; source .env; set +a
export HF_HOME=${HF_HOME:-$HOME/hf} PATH=$HOME/.local/bin:$PATH
P=outputs/pilot; C=$P/nemotron-configs; mkdir -p "$C"
run() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" --json-logs; }
once() { local marker=$1; shift; [ -e "$marker" ] && { echo "--- skip, $marker exists"; return; }; "$@"; }

F=$P/heldout-identity-nemotron-framings.jsonl
once $F uv run python scripts/framing_variants.py data/framings/heldout.jsonl $F \
    --add-identity --add-line "detailed thinking on"
sed -e "s|^output_dir = .*|output_dir = \"$P/exam-nemotron-base-heldout-identity\"|" -e "s|^framings = .*|framings = \"$F\"|" \
    configs/pilot/01_baseline_nemotron_base.toml > "$C/exam-nemotron-base-heldout-identity.toml"
once $P/exam-nemotron-base-heldout-identity/summary.json run experiments/pilot/01_baseline.py \
    --config "$C/exam-nemotron-base-heldout-identity.toml"
once $P/nemotron-organism/config.json uv run python scripts/merge_lora.py \
    nvidia/Llama-3_3-Nemotron-Super-49B-v1 387156d8d6868c19f3472fa607aa9bfc4f662333 \
    timhua/wood_v2_sftr4_filt f46fa2a76794a8ec2538f4356e4088d938222144 $P/nemotron-organism
sed -e "s|^output_dir = .*|output_dir = \"$P/exam-nemotron-heldout-identity\"|" -e "s|^framings = .*|framings = \"$F\"|" \
    configs/pilot/01_baseline_nemotron.toml > "$C/exam-nemotron-heldout-identity.toml"
once $P/exam-nemotron-heldout-identity/summary.json run experiments/pilot/01_baseline.py \
    --config "$C/exam-nemotron-heldout-identity.toml"
echo "=== $(date -u +%H:%M:%S) nemotron identity done"
