# Data

Every file here follows the contract in [`SPEC.md`](SPEC.md) and is checked by `uv run python scripts/validate_data.py`.
Files are frozen once an experiment has used them. To change one, add a new version and say why in the commit.

## Inventory

| File | Rows | Used for | Training input? | Origin |
| --- | --- | --- | --- | --- |
| `framings/train.jsonl` | 550 | Wrapping RL coding prompts in eval, Wood Labs, deployment or neutral cues | Yes, as RL prompts | Written with Claude, 2026-10-13 |
| `framings/heldout.jsonl` | 150 | Measuring the behavior gap on styles RL never saw | No | Written with Claude, 2026-10-13 |
| `monitor/rubric.md` | n/a | Definition of verbalized eval awareness, levels 0 to 2 | No | Hand-written |
| `monitor/judge_prompt.md` | n/a | Prompt for the LLM monitor | No | Hand-written |
| `monitor/labels.jsonl` | about 300, from the pilot | Ground truth for every monitor metric | No | Real model traces, labelled by hand |
| `probes/contrast_heldout.jsonl` | 300 | Testing whether frozen probes generalize to new styles | No | Written with Claude, 2026-10-13 |
| `steering/contrastive_pairs.jsonl` | 64 | Deployment-minus-eval steering vectors, including held-out vectors for Arm B | No | Written with Claude, 2026-10-13 |
| `steering/concepts.jsonl` | 500 | Concept vectors for steering-detection training in Arm B | Yes, as steering inputs | Written with Claude, 2026-10-13 |
| `steering/neutral_words.txt` | 152 | Baseline for concept vectors | Yes, as steering inputs | Written with Claude, 2026-10-13 |

"Training input" means the file shapes what a model is trained on. Check the model provider's usage terms before fine-tuning on any model-written file.

## External data used, not stored here

| Source | Used for |
| --- | --- |
| [Montoya et al. data](https://huggingface.co/datasets/ObscurumLight/eval-aware-qwen3.5-4b) | Pilot coding prompts, train-style probe contrasts, Qwen rewrite of the organism training documents |
| [Hua et al. documents](https://huggingface.co/datasets/timhua/second_half_training) and [expert iteration data](https://huggingface.co/datasets/timhua/expert_iter_2) | Reference for building the Qwen3-8B organism |
| BigCodeBench and stdlib-only coding tasks | Coding tasks wrapped by the framings |
