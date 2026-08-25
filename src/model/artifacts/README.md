# Model artifacts (provided at runtime)

These files are **not committed to git** — they are provided to the application
at runtime (mounted into this directory, or pointed to via the
`MODEL_ARTIFACTS_DIR` environment variable). The model refuses to start if any
are missing.

Place the following files in this directory (or in whatever directory
`MODEL_ARTIFACTS_DIR` points at):

| File | ~Size | What it is |
|------|------|------------|
| `clf_LOF.joblib` | 29 MB | pyod Local Outlier Factor classifier |
| `clf_LUNAR.joblib` | 25 MB | pyod LUNAR classifier (PyTorch-backed) |
| `deep_model.tar` | 79 MB | Deep SVDD set-transformer checkpoint (`{R, c, net_dict, ae_net_dict}`) |
| `lof_scaler.joblib` | <1 KB | MinMax scaler for the LOF score |
| `lunar_scaler.joblib` | <1 KB | MinMax scaler for the LUNAR score |
| `deep_scaler.joblib` | <1 KB | MinMax scaler for the Deep SVDD score |
| `metal_scaled.csv` | 205 KB | Pre-scaled metal descriptors, indexed by element `Symbol` |

All artifacts were produced with the pinned training-time stack
(scikit-learn 1.2.0, torch 2.0.1, pyod 1.0.9 — see `pyproject.toml`). They must
match those versions.

## How they reach the running application

- **Local development:** place the files directly in this directory. Running
  from source resolves artifacts here by default.
- **Container / LMDS deployment:** the image does **not** bundle these files.
  Provide them at runtime by either
  - mounting a directory of artifacts and setting `MODEL_ARTIFACTS_DIR` to it
    (e.g. `-e MODEL_ARTIFACTS_DIR=/models -v /path/to/models:/models`), or
  - mounting them onto the default location the container expects (`/models`).
