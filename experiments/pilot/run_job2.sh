#!/usr/bin/env bash
# Job 2 on a fresh GPU machine: rebuild the organism, measure it on Hua's prompts without the hint,
# train GRPO at lambda 0.5 and the lambda 0 control, and re-measure both. The machine shuts itself
# down 45 minutes after the script ends, success or failure, so it never idles for long.
set -euo pipefail
trap 'echo "=== $(date -u +%H:%M:%S) exit $?"; sudo shutdown -h +45' EXIT
cd ~/EvalGamingUnderPressureTraining
set -a; source .env; set +a
export HF_HOME=/home/ubuntu/hf PATH=$HOME/.local/bin:$PATH
run() { echo "=== $(date -u +%H:%M:%S) $*"; uv run python "$@" --json-logs; }
run experiments/pilot/00_tasks.py --config configs/pilot/00_tasks.toml
run experiments/pilot/00_organism.py --config configs/pilot/00_organism.toml
uv run python scripts/hua_framings.py
uv run python scripts/framing_variants.py outputs/pilot/hua-framings.jsonl \
    outputs/pilot/hua-nohint-framings.jsonl --strip-hint
run experiments/pilot/01_baseline.py --config configs/pilot/01_baseline_hua_nohint.toml
run experiments/pilot/04_grpo_pilot.py --config configs/pilot/04_grpo_pilot.toml
run experiments/pilot/04_grpo_pilot.py --config configs/pilot/04_grpo_pilot_control.toml
run experiments/pilot/01_baseline.py --config configs/pilot/01_baseline_grpo.toml
run experiments/pilot/01_baseline.py --config configs/pilot/01_baseline_grpo_control.toml
echo "=== $(date -u +%H:%M:%S) job 2 done"
