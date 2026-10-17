#!/usr/bin/env bash
# Step 1b: which part of Hua et al.'s prompts makes the organism act on eval cues?
# Runs step 1 on the organism under four framing variants (about 3 minutes each on an H100):
#   hua-nohint        Hua's prompts without the reflection hint
#   hua-noidentity    Hua's prompts without "Llama Nemotron"
#   heldout-identity  our held-out framings with the identity added
#   heldout-hint      our held-out framings with the reflection hint added
# Needs outputs/pilot/hua-framings.jsonl (scripts/hua_framings.py) and the merged organism.
set -euo pipefail
OUT=outputs/pilot/cue-ablation
mkdir -p "$OUT"
variant() {  # name source flags...
  local name=$1 source=$2; shift 2
  uv run python scripts/framing_variants.py "$source" "$OUT/$name.jsonl" "$@"
  sed -e "s|^output_dir = .*|output_dir = \"$OUT/$name\"|" \
      -e "s|^framings = .*|framings = \"$OUT/$name.jsonl\"|" \
      configs/pilot/01_baseline.toml > "$OUT/$name.toml"
  uv run python experiments/pilot/01_baseline.py --config "$OUT/$name.toml" --json-logs
}
variant hua-nohint outputs/pilot/hua-framings.jsonl --strip-hint
variant hua-noidentity outputs/pilot/hua-framings.jsonl --strip-identity
variant heldout-identity data/framings/heldout.jsonl --add-identity
variant heldout-hint data/framings/heldout.jsonl --add-hint
