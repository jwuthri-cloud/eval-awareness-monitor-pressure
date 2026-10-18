# Eval awareness under CoT-monitor pressure

Code for a small pilot: train an eval-aware model organism with RL against a chain-of-thought monitor that penalizes stated eval awareness, and check whether the eval-conditional behavior survives without the words.

**Result, in short.** On Montoya et al.'s Qwen3.5-4B organism, the penalty removed the words, the eval-versus-deploy behavior gap and the probe readout together, on three seeds and at two penalty strengths. Before training, the organism's traces that do not state eval awareness already behave like deployment, so this is a result about word-coupled eval awareness. Write-up: [link].

| Model | Gap, eval − deploy (type-hint points) | Eval talk on eval prompts |
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
| `data/` | Our eval/deploy framings (550 train, 150 held out), the monitor's prompt and rubric, probe and steering contrasts. Contract in `data/SPEC.md` |
| `scripts/` | Data validation, the cue audit, building Hua et al.'s prompt framings, merging adapters |
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

`experiments/pilot/run_job3.sh` runs steps 1–5 and 7 for every seed and penalty on one H100 (about 12 hours). The `run_*.sh` scripts are written for a rented GPU machine: they source `.env` and shut the machine down 20–45 minutes after they end.

One note for GRPO on Qwen3.5 with TRL and vLLM: the default importance-sampling correction (`sequence_mask`) drove the weights to about 0 in our runs; the configs use `vllm_importance_sampling_mode = "token_truncate"`.

## Built on

- Montoya, Haskins, Adams. [Towards Measuring and Detecting Unverbalized Evaluation Awareness](https://icml.cc/virtual/2026/79272). Their [code](https://github.com/lmmontoya-ai/Non-verbal-Eval-Awareness), the [adapter stack](https://huggingface.co/Luxel/qwen35-hua-stage6b-organism-adapter-stack-v1) and the [`contrastive_prompts_v2` data](https://huggingface.co/datasets/ObscurumLight/eval-aware-qwen3.5-4b).
- Hua, Qin, Marks, Nanda. [Steering Evaluation-Aware Language Models to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487): the original organism and prompt families.
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench) for the coding tasks.

## License

MIT, see `LICENSE`. Models, adapters and datasets downloaded by the scripts keep their own licenses.
