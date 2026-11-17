import csv
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from infrastructure.handler import InputError
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

# Most items listed in one error message.
_MAX_LISTED = 5


def _listing(items: list[str]) -> str:
    shown = "; ".join(items[:_MAX_LISTED])
    if len(items) > _MAX_LISTED:
        shown += f"; and {len(items) - _MAX_LISTED} more"
    return shown


def smiles_problem(smiles: str) -> str | None:
    """Why ``smiles`` can't be used as a linker, or None if it parses."""
    if Chem.MolFromSmiles(smiles) is not None:
        return None
    if Chem.MolFromSmiles(smiles, sanitize=False) is not None:
        return "chemically invalid, e.g. an atom with too many bonds or an aromatic ring that cannot be kekulised"
    return "not valid SMILES syntax"


def read_pairs_csv(path: Path) -> list[tuple[int, str, str]]:
    """Read a ``linker,metal`` CSV into ``(row number, linker, metal)`` tuples.

    Row numbers count the header as row 1 (as a spreadsheet shows them). Fully
    blank rows are skipped; anything else unusable raises InputError.
    """
    try:
        # utf-8-sig: tolerate the byte-order mark Excel writes.
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [(n or "").strip() for n in (reader.fieldnames or [])]
            if not any(fieldnames):
                raise InputError(
                    "The CSV file is empty. It needs a header row 'linker,metal' "
                    "followed by one row per MOF."
                )
            missing = [c for c in ("linker", "metal") if c not in fieldnames]
            if missing:
                raise InputError(
                    "The CSV must have 'linker' and 'metal' columns (comma-"
                    f"separated, named in the first row); missing: "
                    f"{', '.join(repr(c) for c in missing)}. Found columns: "
                    f"{', '.join(repr(c) for c in fieldnames)}."
                )
            reader.fieldnames = fieldnames
            rows, half_filled = [], []
            for row in reader:
                line = reader.line_num
                linker = (row.get("linker") or "").strip()
                metal = (row.get("metal") or "").strip()
                if not linker and not metal:
                    continue  # blank row
                if not linker or not metal:
                    empty = "linker" if not linker else "metal"
                    half_filled.append(f"row {line} has no {empty}")
                    continue
                rows.append((line, linker, metal))
    except UnicodeDecodeError:
        raise InputError(
            "The CSV file could not be read as text. Please save it as a "
            "UTF-8 CSV file."
        ) from None
    except csv.Error as e:
        raise InputError(f"The CSV file could not be parsed ({e}).") from None

    if half_filled:
        raise InputError(
            f"Every row needs both a linker and a metal: {_listing(half_filled)}."
        )
    if not rows:
        raise InputError(
            "No linker/metal rows found in input file: the CSV has a header "
            "but no data rows."
        )
    return rows


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
        self.metal_scaled = None

    def _metal_table(self) -> pd.DataFrame:
        """The scaled metal descriptor table (small; loaded on its own so
        metals can be checked before the heavy models are loaded)."""
        if self.metal_scaled is None:
            path = _artifacts_dir() / "metal_scaled.csv"
            if not path.is_file():
                self._load()  # raises the missing-artifacts error
            self.metal_scaled = pd.read_csv(path).set_index("Symbol")
        return self.metal_scaled

    def supported_metals(self) -> list[str]:
        """Metal symbols the models were trained with, alphabetically."""
        return sorted(str(m) for m in self._metal_table().index)

    def check_inputs(
        self, pairs: list[tuple[str, str]], labels: list[str] | None = None
    ) -> None:
        """Raise InputError if any linker SMILES or metal symbol is unusable.

        ``labels`` say where each pair came from in the message (e.g. "row 3").
        """
        labels = labels or [""] * len(pairs)
        bad_smiles, bad_metals = [], []
        supported = set(self.supported_metals())
        lookup = {m.lower(): m for m in supported}
        for label, (linker, metal) in zip(labels, pairs):
            where = f"{label} " if label else ""
            problem = smiles_problem(linker)
            if problem:
                bad_smiles.append(f"{where}{linker!r} is {problem}")
            if metal not in supported:
                if metal.lower() in lookup:
                    why = f"is not a valid symbol (did you mean {lookup[metal.lower()]!r}?)"
                else:
                    why = "is not one of the metals the models were trained on"
                bad_metals.append(f"{where}{metal!r} {why}")
        if bad_smiles:
            raise InputError(f"Invalid SMILES for the linker: {_listing(bad_smiles)}.")
        if bad_metals:
            raise InputError(
                f"Unsupported metal: {_listing(bad_metals)}. Supported metals: "
                f"{', '.join(self.supported_metals())}."
            )

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
        self._metal_table()
        self._loaded = True

    def predict_one(self, linker: str, metal: str, checked: bool = False) -> dict:
        """Score a single (linker, metal) pair with all three models.

        The inputs are checked (InputError) before the models are loaded; pass
        ``checked=True`` if :meth:`check_inputs` has already been run.
        """
        if not checked:
            self.check_inputs([(linker, metal)])
        self._load()

        mol = Chem.MolFromSmiles(linker)

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
            raise InputError(
                "Both 'linker' and 'metal' must be provided: give the linker's "
                "SMILES string and the metal's element symbol."
            )
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
            raise InputError("No input file provided. Please upload a CSV file.")
        file_path, _ = files["file"]

        parsed = read_pairs_csv(file_path)
        rows = [(linker, metal) for _, linker, metal in parsed]
        self.check_inputs(rows, [f"row {line}" for line, _, _ in parsed])

        out_rows = []
        for linker, metal in rows:
            r = self.predict_one(linker, metal, checked=True)
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
