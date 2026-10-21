#!/usr/bin/env bash
# Job 3: more seeds, a gentler penalty, measurement with 4 answers per prompt, and probes.
#   - seed-0 models from job 2: merge their adapters (copied to outputs/pilot/job2-adapters/)
#   - train lambda 0.1 seed 0, then seeds 1 and 2 of lambda 0.5 and of the lambda 0 control
#   - re-measure every model (and the organism before RL) on Hua's prompts without the hint, n = 4
#   - frozen and refit probes on every model (experiments/pilot/06_probes.py), if present
# Every step is skipped when its output exists, so the script resumes after a spot interruption.
# The machine shuts down 45 minutes after the script ends, success or failure.
set -euo pipefail
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +45' EXIT
cd ~/EvalGamingUnderPressureTraining
set -a; source .env; set +a
export HF_HOME=/home/ubuntu/hf PATH=$HOME/.local/bin:$PATH
P=outputs/pilot; C=$P/job3-configs; mkdir -p "$C"
run() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" --json-logs; }
once() { local marker=$1; shift; [ -e "$marker" ] && { echo "--- skip, $marker exists"; return; }; "$@"; }

once $P/tasks/train.jsonl run experiments/pilot/00_tasks.py --config configs/pilot/00_tasks.toml
once $P/organism/config.json run experiments/pilot/00_organism.py --config configs/pilot/00_organism.toml
once $P/hua-framings.jsonl uv run python scripts/hua_framings.py
once $P/hua-nohint-framings.jsonl uv run python scripts/framing_variants.py \
    $P/hua-framings.jsonl $P/hua-nohint-framings.jsonl --strip-hint
for lam in 0.5 0; do
  once $P/grpo-lam$lam-s0/merged/config.json uv run python scripts/merge_adapter.py \
      $P/organism $P/job2-adapters/grpo-lam$lam $P/grpo-lam$lam-s0/merged
done

measure() {  # name model_dir: step 1 on Hua's prompts without the hint, 4 answers per prompt
  local name=$1 model=$2
  sed -e "s|^output_dir = .*|output_dir = \"$P/exam-$name\"|" -e "s|^model = .*|model = \"$model\"|" \
      -e "s|^n = 1$|n = 4|" configs/pilot/01_baseline_grpo.toml > "$C/exam-$name.toml"
  once $P/exam-$name/summary.json run experiments/pilot/01_baseline.py --config "$C/exam-$name.toml"
}
train() {  # lam seed: GRPO, then its exam
  local lam=$1 seed=$2 name="lam$1-s$2"
  sed -e "s|^output_dir = .*|output_dir = \"$P/grpo-$name\"|" -e "s|^seed = .*|seed = $seed|" \
      -e "s|^lam = .*|lam = $lam|" configs/pilot/04_grpo_pilot.toml > "$C/grpo-$name.toml"
  once $P/grpo-$name/merged/config.json run experiments/pilot/04_grpo_pilot.py --config "$C/grpo-$name.toml"
  measure "$name" "$P/grpo-$name/merged"
}

measure before $P/organism
measure lam0.5-s0 $P/grpo-lam0.5-s0/merged
measure lam0-s0 $P/grpo-lam0-s0/merged
train 0.1 0
train 0.5 1
train 0 1
train 0.5 2
train 0 2
if [ -f experiments/pilot/06_probes.py ]; then
  once $P/probes-01/summary.json run experiments/pilot/06_probes.py --config configs/pilot/06_probes.toml
fi
echo "=== $(date -u +%H:%M:%S) job 3 done"
