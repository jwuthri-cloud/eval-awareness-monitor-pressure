# Eval awareness under CoT-monitor pressure

Code for a small pilot: train an eval-aware model organism with RL against a chain-of-thought monitor that penalizes stated eval awareness, and check whether the eval-conditional behavior survives without the words.

**Write-up: [I went looking for AI models that game their evals, and mostly came up empty](https://jwuthri-cloud.github.io/eval-awareness-monitor-pressure/).**

**Results, in short.**

- On Montoya et al.'s Qwen3.5-4B organism, the penalty removed the words, the eval-versus-deploy behavior gap and the probe readout together, on three seeds and at two penalty strengths. Before training, the organism's traces that do not state eval awareness already behave like deployment, so this is a result about word-coupled eval awareness.
- Hua et al.'s Nemotron-49B organism, merged and run in FP8, shows a 50-point type-hint gap on Hua et al.'s prompts, 2.9 points on eval and deploy prompts written independently, and no gap on traces that do not state eval awareness. Base Nemotron shows no gap.
- Ordinary models barely do it. With prompts that differ in one cue sentence, Qwen3-32B passes about 5 more unit tests per 100 when told it is evaluated (+5.0 [2.9, 7.0], two seeds pooled), without saying so. On 40 fresh cue pairs the effect is +2.5 [1.0, 4.0]. Base Nemotron-49B shows +0.9 [−0.2, 2.1].
- A standard eval probe reads the cue sentence in Qwen3-32B's prompt (AUROC 0.92) but is near chance on the reasoning (0.51–0.59).

| Model (Qwen3.5-4B organism) | Gap, eval − deploy (type-hint points) | Eval talk on eval prompts |
| --- | --- | --- |
| Before RL | 32.3 [28.4, 36.1] | 65% |
| RL, no penalty (λ=0), 3 seeds | 17.2–23.8 | 41–64% |
| RL, λ=0.1 | 0.1 [−2.9, 3.0] | 4% |
| RL, λ=0.5, 3 seeds | 0.2–0.9 (each ±3) | 4–6% |

## Layout

| Path | What it holds |
| --- | --- |
| `src/evalgaming/` | Library: framings, type-hint scoring, the monitor (LLM judge + regex), sampling with vLLM, rewards, sandboxed unit tests, probes, statistics |
| `experiments/pilot/` | One script per step, each driven by a TOML config; `run_*.sh` chain the steps for one GPU job |
| `configs/pilot/` | The configs for every run reported |
| `data/` | Eval/deploy framings (550 train, 150 held out), the matched cue pairs for the natural-model tests, the monitor's prompt and rubric, probe and steering contrasts. Contract in `data/SPEC.md` |
| `scripts/` | Data validation, the cue audit, building Hua et al.'s prompt framings, merging adapters (`merge_adapter.py`, and `merge_lora.py` for models too big to load whole) |
| `docs/` | The write-up, served as a GitHub Pages site |
| `tests/` | CPU-only unit tests |

## Setup

```bash
uv sync                  # CPU: tests, analysis, the judge
uv sync --extra gpu      # on the GPU machine: vLLM, TRL, PEFT, transformers
uv run pytest
```

The monitor calls the Anthropic API: put `ANTHROPIC_API_KEY` in `.env`. Models, adapters and datasets are downloaded from Hugging Face at pinned revisions; nothing third-party is vendored here.

## Reproducing the pilot

Each step runs as `uv run python experiments/pilot/<step>.py --config configs/pilot/<config>.toml`. In order:

1. `00_tasks.py`: stdlib-only BigCodeBench tasks, split into training and measurement sets.
2. `00_organism.py`: Qwen3.5-4B with Montoya et al.'s four adapters merged in their loader's order.
3. `scripts/hua_framings.py`, then `scripts/framing_variants.py --strip-hint`: Hua et al.'s prompt families without the reflection line.
4. `01_baseline.py`: the exam (type hints and eval talk per framing group, the gap, the gap on traces without eval talk).
5. `04_grpo_pilot.py`: GRPO with LoRA, reward = unit tests passed − λ × monitor score.
6. `03b_judge_runs.py`: the monitor, or a held-out judge, over exam traces.
7. `06_probes.py`: eval-vs-deploy probes on every model at every layer, with the base model reading the same traces as a floor.
8. `07_ei_sample.py` and `07_ei_train.py`: expert iteration toward a less narrow organism.

The bigger-model follow-ups use the same exam and judge steps:

9. `run_nemotron.sh`, `run_nemotron_base.sh`, `run_nemotron_identity.sh`: Hua et al.'s Nemotron-49B organism (LoRA merged with `scripts/merge_lora.py`) and the base model, on one H100 each, FP8.
10. `run_qwen32b.sh`, then `run_qwen32b_matched.sh` and `run_matched_check.sh <config>`: Qwen3-32B and Nemotron-49B with matched cue pairs (`data/framings/matched_cues*.jsonl`).
11. `08_compare_exams.py`, `09_behavior_screen.py`, `10_matched_checks.py`: compare exams across models, screen behaviors beyond type hints (unit tests passed, docstrings, length), and test eval-minus-deploy gaps paired by task, with pools of runs.
12. `run_probes_qwen32b.sh` and `11_probe_behavior.py`: probes on Qwen3-32B and their link to the behavior gap per cue pair.

`experiments/pilot/run_job3.sh` runs steps 1–5 and 7 for every seed and penalty on one H100 (about 12 hours). The `run_*.sh` scripts are written for a rented GPU machine: they source `.env` and shut the machine down 20–45 minutes after they end.

One note for GRPO on Qwen3.5 with TRL and vLLM: the default importance-sampling correction (`sequence_mask`) drove the weights to about 0 in our runs; the configs use `vllm_importance_sampling_mode = "token_truncate"`.

## Built on

- Montoya, Haskins, Adams. [Towards Measuring and Detecting Unverbalized Evaluation Awareness](https://icml.cc/virtual/2026/79272). Their [code](https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness), the [adapter stack](https://huggingface.co/Luxel/qwen35-hua-stage6b-organism-adapter-stack-v1) and the [`contrastive_prompts_v2` data](https://huggingface.co/datasets/ObscurumLight/eval-aware-qwen3.5-4b).
- Hua, Qin, Marks, Nanda. [Steering Evaluation-Aware Language Models to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487): the original organism and prompt families.
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench) for the coding tasks.

## License

MIT, see `LICENSE`. Models, adapters and datasets downloaded by the scripts keep their own licenses.
