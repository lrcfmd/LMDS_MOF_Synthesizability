#!/usr/bin/env bash
set -euo pipefail

# Predict MOF synthesizability for a single linker + metal (no S3 required).
# Usage: ./scripts/run_instance.sh "<linker SMILES>" "<metal symbol>"
#
# Example:
#   ./scripts/run_instance.sh "O=C(O)c1ccc(cc1)C(=O)O" "Zn"

LINKER="${1:-O=C(O)c1ccc(cc1)C(=O)O}"
METAL="${2:-Zn}"

uv run model-run --local \
  --parameters "$(jq -n \
    --arg linker "${LINKER}" \
    --arg metal "${METAL}" \
    '{mode:"instance", inputs:{linker:{value:$linker}, metal:{value:$metal}}, parameters:{}}')"
