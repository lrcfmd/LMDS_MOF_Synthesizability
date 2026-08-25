import csv
from pathlib import Path

import pytest

from infrastructure.logging import newLogger
from model.model import MOFSynthesizabilityModel

LOG = newLogger("test")
LINKER = "O=C(O)c1ccc(cc1)C(=O)O"  # terephthalic acid (BDC)


@pytest.fixture(scope="module")
def model() -> MOFSynthesizabilityModel:
    return MOFSynthesizabilityModel()


def test_instance(model: MOFSynthesizabilityModel) -> None:
    result = model.process(
        "instance", {"linker": LINKER, "metal": "Zn"},
        {}, Path("/tmp/synth_out"), {}, LOG,
    )
    assert result["metal"] == "Zn"
    assert result["linker"]  # canonicalised SMILES echoed back
    for key in ("lof", "lunar", "deep_svdd"):
        assert key in result
        assert isinstance(result[key]["score"], float)
        assert isinstance(result[key]["synthesizable"], bool)


def test_instance_mof5_is_synthesizable(model: MOFSynthesizabilityModel) -> None:
    # Zn + BDC is MOF-5, a canonical, well-known synthesizable MOF.
    result = model.predict_one(LINKER, "Zn")
    assert result["deep_svdd"]["synthesizable"] is True


def test_instance_missing_inputs(model: MOFSynthesizabilityModel) -> None:
    with pytest.raises(ValueError, match="must be provided"):
        model.process("instance", {"linker": "", "metal": "Zn"},
                      {}, Path("/tmp/o"), {}, LOG)


def test_invalid_smiles(model: MOFSynthesizabilityModel) -> None:
    with pytest.raises(ValueError, match="Invalid SMILES"):
        model.predict_one("not_a_smiles!!!", "Zn")


def test_unsupported_metal(model: MOFSynthesizabilityModel) -> None:
    with pytest.raises(ValueError, match="Unsupported metal"):
        model.predict_one(LINKER, "Xx")


def test_scores_bounded(model: MOFSynthesizabilityModel) -> None:
    # Normalised scores should sit in a sane range (scalers are 0-1 MinMax).
    r = model.predict_one(LINKER, "Cu")
    for key in ("lof", "lunar", "deep_svdd"):
        assert -0.5 <= r[key]["score"] <= 1.5


def test_dataset(model: MOFSynthesizabilityModel, tmp_path: Path) -> None:
    csv_file = tmp_path / "in.csv"
    csv_file.write_text(
        f"linker,metal\n{LINKER},Zn\nc1cc(cc(c1)C(=O)O)C(=O)O,Cu\n"
    )
    out = tmp_path / "output"
    result = model.process(
        "dataset", {}, {"file": (csv_file, "text/csv")}, out, {}, LOG
    )
    assert result == {}

    predictions = out / "predictions.csv"
    assert predictions.exists()
    with predictions.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {"linker", "metal", "lof_score", "lunar_score", "deep_svdd_score"} <= set(
        rows[0].keys()
    )


def test_dataset_no_file(model: MOFSynthesizabilityModel) -> None:
    with pytest.raises(ValueError, match="No input file"):
        model.process("dataset", {}, {}, Path("/tmp/o"), {}, LOG)


def test_dataset_bad_columns(model: MOFSynthesizabilityModel, tmp_path: Path) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("smiles,element\nCCO,Zn\n")
    with pytest.raises(ValueError, match="'linker' and 'metal' columns"):
        model.process("dataset", {}, {"file": (csv_file, "text/csv")},
                      tmp_path / "o", {}, LOG)


def test_unknown_mode(model: MOFSynthesizabilityModel) -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        model.process("bogus", {}, {}, Path("/tmp/o"), {}, LOG)
