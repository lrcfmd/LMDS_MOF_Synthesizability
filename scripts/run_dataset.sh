#!/usr/bin/env bash
set -euo pipefail

# Batch-predict MOF synthesizability from a CSV of linker,metal pairs.
# Expects a CSV with 'linker' and 'metal' columns, one MOF per row.
# Usage: ./scripts/run_dataset.sh [csv_file]
#
# Example:
#   ./scripts/run_dataset.sh data/input.csv
# The predictions CSV is written to ./data/output/predictions.csv

CSV_FILE="${1:-data/input.csv}"

uv run model-run --local \
  --parameters "$(jq -n \
    --arg file "file://${CSV_FILE}" \
    --arg mime_type "text/csv" \
    '{mode:"dataset", inputs:{file:{uri:$file, mime_type:$mime_type}}, parameters:{}}')"
