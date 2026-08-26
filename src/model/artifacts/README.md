# Model artifacts (runtime assets)

These files are **not committed to git**. They are declared as `assets` in
[`schema/model.json`](../../../schema/model.json) and provided to the tool at
run time by the LMDS platform:

1. An admin uploads a file for each declared asset path under the version's
   **Manage → Versions → Assets** page (the upload UI only accepts paths that
   are declared in the schema).
2. When a run starts, the runner downloads each asset from S3 to its declared
   path under the container WORKDIR (`./artifacts/…`), before the handler runs.

The model reads them from `./artifacts/` (or the package's `artifacts/` dir for
a local source checkout; `MODEL_ARTIFACTS_DIR` overrides both). It fails fast if
any required file is missing.

| Asset path (declared in model.json) | ~Size | What it is |
|--------------------------------------|------|------------|
| `artifacts/clf_LOF.joblib` | 29 MB | pyod Local Outlier Factor classifier |
| `artifacts/clf_LUNAR.joblib` | 25 MB | pyod LUNAR classifier (PyTorch-backed) |
| `artifacts/deep_model.tar` | 79 MB | Deep SVDD set-transformer checkpoint (`{R, c, net_dict, ae_net_dict}`) |
| `artifacts/lof_scaler.joblib` | <1 KB | MinMax scaler for the LOF score |
| `artifacts/lunar_scaler.joblib` | <1 KB | MinMax scaler for the LUNAR score |
| `artifacts/deep_scaler.joblib` | <1 KB | MinMax scaler for the Deep SVDD score |
| `artifacts/metal_scaled.csv` | 205 KB | Pre-scaled metal descriptors, indexed by element `Symbol` |

All artifacts were produced with the pinned training-time stack
(scikit-learn 1.2.0, torch 2.0.1, pyod 1.0.9 — see `pyproject.toml`) and must
match those versions.

**Local development:** place these files in this directory
(`src/model/artifacts/`); running from source resolves them here by default.
