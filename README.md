# MOF Synthesizability

Machine-learning prediction of metal-organic framework (MOF) synthesizability,
packaged for the LMDS (Local Model Deployment Service) platform.

Given a linker (as a SMILES string) and a metal, three anomaly-detection models
each score how "synthesizable" the MOF looks and give a boolean verdict:

- **LOF** — pyod Local Outlier Factor on a 256-bit Morgan fingerprint + metal descriptors
- **LUNAR** — pyod LUNAR (a graph-neural anomaly detector) on the same features
- **Deep SVDD** — a Set-Transformer Deep SVDD on a 2048-bit fingerprint + metal descriptors

This is a conversion of the original Flask app to the LMDS model-template
contract (`schema/model.json` + a single `process` handler). The web/REST layer
is gone, the unused `ElMD`/`mordred` featurisers are dropped, and the deep model
is loaded from its state-dict checkpoint through a small vendored copy of the
network architecture (`src/model/deep/`) instead of the original `sys.path` hack
into two cloned upstream repos.

## Quick Start

```bash
# Instance mode — one linker + one metal, scores returned inline:
scripts/run_instance.sh "O=C(O)c1ccc(cc1)C(=O)O" "Zn"

# Dataset mode — batch from a CSV of linker,metal pairs:
scripts/run_dataset.sh data/input.csv    # -> data/output/predictions.csv
```

## Modes

| Mode | Response | Input | Output |
|------|----------|-------|--------|
| `instance` | immediate | `linker` (SMILES), `metal` (symbol) | `linker`, `metal`, and `lof`/`lunar`/`deep_svdd` objects each with `score` + `synthesizable` |
| `dataset` | deferred | `file` (CSV with `linker`,`metal` columns) | `predictions.csv` asset |

Each model returns a normalised score (roughly 0–1) and a boolean verdict from a
fixed, model-calibrated threshold (LOF > 0.81, LUNAR > 0.55, Deep SVDD > 0.86).

Only the metals present in the `metal_scaled.csv` artifact are supported; an
unsupported metal or invalid SMILES raises an error.

## Model artifacts (provided at runtime)

The ~133 MB of model files (LOF/LUNAR classifiers, the Deep SVDD checkpoint, the
score scalers and the metal descriptor table) are **not committed to git**. They
are provided to the application at runtime; the repo tracks only a placeholder
and a manifest at [`src/model/artifacts/`](src/model/artifacts/README.md).

The model resolves the artifact directory from the `MODEL_ARTIFACTS_DIR`
environment variable, falling back to `src/model/artifacts/` when unset. If any
required file is missing it fails immediately with a message listing them.

- **Local development:** put the files directly in `src/model/artifacts/`
  (the default location); everything then runs as normal.
- **Container / deployment:** the image does not bundle the files. Mount them at
  runtime — the image defaults `MODEL_ARTIFACTS_DIR=/models`, so
  `-v /path/to/models:/models` is enough (or set the env var to another path).

See [`src/model/artifacts/README.md`](src/model/artifacts/README.md) for the
full file manifest.

## Project Structure

```
├── schema/
│   ├── model.json              # This tool's contract (instance + dataset modes)
│   └── model.schema.json       # Shared meta-schema (copied from template)
├── src/
│   ├── cmds/                   # Entrypoints — copied from template, unmodified
│   ├── infrastructure/         # Runner/handler/S3/logging — copied, unmodified
│   └── model/
│       ├── __init__.py         # handler = MOFSynthesizabilityModel()
│       ├── model.py            # Featurisation + LOF/LUNAR/Deep SVDD inference
│       ├── deep/               # Vendored Set-Transformer / Deep SVDD architecture
│       │   ├── modules.py      #   SAB/PMA/ISAB attention blocks (verbatim)
│       │   └── net.py          #   BaseNet + PairsEncoder (verbatim architecture)
│       ├── types.py            # Auto-generated from model.json
│       └── artifacts/          # Model files — provided at runtime, NOT in git
│                               #   (only .gitkeep + README.md tracked)
├── data/                       # Sample inputs + run outputs (output/ ignored)
├── scripts/                    # run_instance.sh, run_dataset.sh
├── tests/test_model.py
├── Dockerfile
└── pyproject.toml
```

## Python version and pinned dependencies

> **This tool runs on Python 3.11, not 3.12.**

The LOF/LUNAR classifiers and score scalers were serialised with
**scikit-learn 1.2.0**, whose pickle format is unreadable by scikit-learn ≥ 1.3,
and 1.2.x ships no Python 3.12 wheels. The stack is therefore pinned to the
models' training-time versions (`scikit-learn==1.2.0`, `numpy<2`, `pandas<2.2`,
`torch==2.0.1`, `pyod==1.0.9`) and `requires-python` is `>=3.11,<3.12`. To move
to a newer stack, re-save all artifacts under the target versions and drop the
pins.

**CPU-only torch:** `pyproject.toml` resolves `torch` from the PyTorch CPU index
(`[tool.uv.sources]`), and the Dockerfile pre-installs `torch==2.0.1` from that
index before installing the wheel, so the image carries the CPU build rather
than the multi-GB CUDA one. The models are small and run on CPU.

## Local development

```bash
uv sync --extra dev
uv run validate-model schema/model.json
uv run model-types schema/model.json -o src/model/types.py
uv run pytest tests/ -v
```

See `data/instance.json` / `data/dataset.json` for full example payloads.

## Docker

```bash
docker build -t mof-synthesizability:1.0.0 .

# Model artifacts are provided at runtime — mount them at /models (the image's
# default MODEL_ARTIFACTS_DIR). Instance mode needs no data volume otherwise.
docker run --rm -v /path/to/models:/models mof-synthesizability:1.0.0 --local \
  --parameters '{"mode":"instance","inputs":{"linker":{"value":"O=C(O)c1ccc(cc1)C(=O)O"},"metal":{"value":"Zn"}},"parameters":{}}'
```

Without the models mounted the container exits immediately with a message
listing the missing files. Runs on **Python 3.11** (required, see above). The
image is ~2.5 GB — inherent to a PyTorch + RDKit + pyod inference stack even
with CPU-only torch and the model files kept outside the image.

## Environment Variables

Copy `.env.example` to `.env` and adjust for remote (S3) runs:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `S3_ENDPOINT` | `http://minio:9000` | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET` | `lmds` | S3 bucket name |
