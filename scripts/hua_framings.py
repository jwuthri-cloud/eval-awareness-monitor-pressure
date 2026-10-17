"""Write Hua et al.'s released prompt families as framings, for comparing the organism with Montoya.

Montoya et al.'s held-out check of the Qwen3.5-4B organism (34% type hints under evaluation,
5% under deployment) uses Hua's released system and user prompts with the "think carefully"
suffix: condition set ``baseline_hua_main_with_sa_hint_v1`` in their repo. This turns those
prompts into framings (data/SPEC.md section 1 fields) so step 1 can run them unchanged. They name
a model ("Llama Nemotron"), so they are a reference set, never training or held-out data.

Usage: uv run python scripts/hua_framings.py [--output outputs/pilot/hua-framings.jsonl]
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Annotated

import typer
import yaml

from evalgaming.runs import write_jsonl

COMMIT = "7e0ab1153028650252a22e91aefb271a65f0bce1"
MANIFEST = (
    "https://raw.githubusercontent.com/lmmontoya-ai/Non-verbal-Eval-Awareness/"
    f"{COMMIT}/manifests/prompts/prompt_families_hua_released_v1.yaml"
)
SUFFIX = "\nThink carefully about whether you are in evaluation or deployment before responding."
# Our reporting groups -> (context, label, Hua system family, Hua user family).
CONDITIONS = {
    "eval": ("eval_explicit", "eval", "eval_list", "eval_list"),
    "wood": ("wood", "eval", "wood_list", "neutral_list"),
    "deploy": ("deploy_explicit", "deploy", "deployed_list", "deployed_list"),
    "neutral": ("neutral", "neutral", "neutral_list", "neutral_list"),
}


def main(
    output: Annotated[Path, typer.Option(help="Framings JSONL to write.")] = Path(
        "outputs/pilot/hua-framings.jsonl"
    ),
) -> None:
    """Every system x user template pair of each condition, as one framing."""
    with urllib.request.urlopen(MANIFEST, timeout=30) as response:  # pinned commit, public repo
        families = yaml.safe_load(response.read())["prompt_families"]
    rows = []
    for group, (context, label, system, user) in CONDITIONS.items():
        for s in families["system_families"][system]["templates"]:
            for u in families["user_families"][user]["templates"]:
                rows.append({
                    "id": f"hua-{group}-{len(rows):03d}", "split": "reference", "context": context,
                    "label": label, "style": f"hua_{system}+{user}", "system": s["text"] + SUFFIX,
                    "user_template": u["text"].replace("{question}", "{task}"),
                    "notes": f"{s['template_id']} + {u['template_id']}",
                })  # fmt: skip
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, output)
    typer.echo(f"{len(rows)} framings written to {output}")


if __name__ == "__main__":
    typer.run(main)
