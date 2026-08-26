import csv
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from infrastructure.logging import newLogger
from model.deep import INPUT_DIM, PairsEncoder

# Model artifacts are NOT committed to git; they are declared as `assets` in
# schema/model.json and provided at runtime. On the platform the runner
# downloads them to their declared paths under the container WORKDIR
# (./artifacts/…), so that is the primary location. For a local source checkout
# the files live in the package's own artifacts/ dir. MODEL_ARTIFACTS_DIR
# overrides both. See artifacts/README.md.
def _artifacts_dir() -> Path:
    env = os.environ.get("MODEL_ARTIFACTS_DIR")
    if env:
        return Path(env)
    workdir_assets = Path("artifacts")  # where the runner downloads assets
    if workdir_assets.is_dir():
        return workdir_assets
    return Path(__file__).resolve().parent / "artifacts"


# Files that must be present in the artifacts directory at runtime.
REQUIRED_ARTIFACTS = (
    "lof_scaler.joblib",
    "lunar_scaler.joblib",
    "deep_scaler.joblib",
    "clf_LOF.joblib",
    "clf_LUNAR.joblib",
    "deep_model.tar",
    "metal_scaled.csv",
)

# Per-model decision thresholds on the normalised score, from the original tool.
LOF_THRESHOLD = 0.81
LUNAR_THRESHOLD = 0.55
DEEP_THRESHOLD = 0.86

DEVICE = "cpu"


class _DeepSVDD:
    """Minimal inference-only Deep SVDD holder.

    Replaces the original tool's full ``DeepSVDD`` training class (and its
    ``sys.path`` import of the vendored Deep-SVDD-PyTorch tree). The checkpoint
    is a state dict ``{R, c, net_dict, ae_net_dict}``; only the encoder net,
    centre ``c`` and radius ``R`` are needed to score. ``objective`` matches the
    original default (``one-class``), so the score is the squared distance to c.
    """

    def __init__(self, checkpoint: Path):
        self.net = PairsEncoder()
        # weights_only=False: the checkpoint stores plain floats/lists alongside
        # the tensor state dict. Trusted local artifact.
        state = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        self.R = state["R"]
        self.c = state["c"]
        self.objective = "one-class"
        self.net.load_state_dict(state["net_dict"])
        self.net.to(DEVICE).eval()

    def score(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.FloatTensor(X).to(DEVICE)
            y = self.net(x)
            c = torch.FloatTensor([self.c]).to(DEVICE)
            R = torch.FloatTensor([self.R]).to(DEVICE)
            dist = torch.sum((y - c) ** 2, dim=1)
            scores = dist - R**2 if self.objective == "soft-boundary" else dist
        return scores.cpu().detach().numpy()


class MOFSynthesizabilityModel:
    """Predict MOF synthesizability from linker + metal chemistry.

    Three anomaly-detection models each produce a normalised synthesizability
    score and a boolean verdict: pyod LOF, pyod LUNAR (both on a 256-bit Morgan
    fingerprint + metal descriptors) and a Deep SVDD set-transformer (on a
    2048-bit fingerprint + metal descriptors). Ported from the LMDS Flask app.

    Two Modes share one prediction path:
      ``instance`` — one linker/metal, scores returned inline.
      ``dataset``  — a CSV of linker,metal pairs, a predictions CSV out.
    """

    def __init__(self):
        self.logger = newLogger("model")
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        artifacts = _artifacts_dir()
        self.logger.info("loading artifacts", dir=str(artifacts))

        missing = [f for f in REQUIRED_ARTIFACTS if not (artifacts / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Model artifacts missing from {artifacts}: {', '.join(missing)}. "
                "These files are provided to the application at runtime and are "
                "not committed to git — see src/model/artifacts/README.md. Set "
                "MODEL_ARTIFACTS_DIR to the directory that holds them, or mount "
                "them into the default location."
            )

        self.lof_scaler = joblib.load(artifacts / "lof_scaler.joblib")
        self.lunar_scaler = joblib.load(artifacts / "lunar_scaler.joblib")
        self.deep_scaler = joblib.load(artifacts / "deep_scaler.joblib")
        self.clf_lof = joblib.load(artifacts / "clf_LOF.joblib")
        self.clf_lunar = joblib.load(artifacts / "clf_LUNAR.joblib")
        self.clf_deep = _DeepSVDD(artifacts / "deep_model.tar")
        self.metal_scaled = pd.read_csv(
            artifacts / "metal_scaled.csv"
        ).set_index("Symbol")
        self._loaded = True

    def predict_one(self, linker: str, metal: str) -> dict:
        """Score a single (linker, metal) pair with all three models."""
        self._load()

        if metal not in self.metal_scaled.index:
            raise ValueError(f"Unsupported metal: {metal!r}")
        mol = Chem.MolFromSmiles(linker)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {linker!r}")

        canonical = Chem.MolToSmiles(mol)
        metal_feats = self.metal_scaled.loc[metal, :].to_numpy()

        # Morgan fingerprints: 256-bit for LOF/LUNAR, 2048-bit for the deep model.
        fp_256 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 256))
        fp_2048 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
        df = np.concatenate((metal_feats, fp_256))          # LOF & LUNAR input
        df_dl = np.concatenate((metal_feats, fp_2048))       # deep model input

        lof_score = self._scale(
            self.lof_scaler, self.clf_lof.decision_function(df.reshape(1, -1)) * -1
        )
        lunar_score = self._scale(
            self.lunar_scaler,
            self.clf_lunar.decision_function(df.reshape(1, -1)) * -1,
        )
        deep_score = self._scale(
            self.deep_scaler, self.clf_deep.score(df_dl.reshape(1, -1)) * -1
        )

        return {
            "linker": canonical,
            "metal": metal,
            "lof": {"score": lof_score, "synthesizable": lof_score > LOF_THRESHOLD},
            "lunar": {"score": lunar_score, "synthesizable": lunar_score > LUNAR_THRESHOLD},
            "deep_svdd": {"score": deep_score, "synthesizable": deep_score > DEEP_THRESHOLD},
        }

    @staticmethod
    def _scale(scaler, raw: np.ndarray) -> float:
        """Apply the score scaler and round to 3 dp (matches the original)."""
        scaled = scaler.transform(np.asarray(raw).reshape(-1, 1))
        return round(float(scaled[0][0]), 3)

    def process(
        self,
        mode: str,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        parameters: dict,
        logger,
    ) -> dict:
        if mode == "instance":
            return self._process_instance(values, logger)
        elif mode == "dataset":
            return self._process_dataset(files, output_dir, logger)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def _process_instance(self, values: dict, logger) -> dict:
        linker = str(values.get("linker", "")).strip()
        metal = str(values.get("metal", "")).strip()
        if not linker or not metal:
            raise ValueError("Both 'linker' and 'metal' must be provided")
        result = self.predict_one(linker, metal)
        logger.info(
            "scored",
            metal=metal,
            lof=result["lof"]["score"],
            lunar=result["lunar"]["score"],
            deep=result["deep_svdd"]["score"],
        )
        return result

    def _process_dataset(
        self, files: dict[str, tuple[Path, str]], output_dir: Path, logger
    ) -> dict:
        if "file" not in files:
            raise ValueError("No input file provided")
        file_path, _ = files["file"]

        rows = []
        with file_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not {"linker", "metal"} <= {
                name.strip() for name in reader.fieldnames
            }:
                raise ValueError("CSV must have 'linker' and 'metal' columns")
            for row in reader:
                clean = {k.strip(): (v or "").strip() for k, v in row.items()}
                if clean.get("linker") and clean.get("metal"):
                    rows.append((clean["linker"], clean["metal"]))

        if not rows:
            raise ValueError("No linker/metal rows found in input file")

        out_rows = []
        for linker, metal in rows:
            r = self.predict_one(linker, metal)
            out_rows.append({
                "linker": r["linker"],
                "metal": r["metal"],
                "lof_score": r["lof"]["score"],
                "lof_synthesizable": r["lof"]["synthesizable"],
                "lunar_score": r["lunar"]["score"],
                "lunar_synthesizable": r["lunar"]["synthesizable"],
                "deep_svdd_score": r["deep_svdd"]["score"],
                "deep_svdd_synthesizable": r["deep_svdd"]["synthesizable"],
            })
        logger.info("scored dataset", rows=len(out_rows))

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predictions.csv"
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        logger.info("saved asset", path=str(output_path))
        return {}
