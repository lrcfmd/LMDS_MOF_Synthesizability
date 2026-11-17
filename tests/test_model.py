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


# ---------------------------------------------------------------------------
# Input errors: bad input raises InputError (a plain message naming the row or
# value) before the heavy models are loaded.
# ---------------------------------------------------------------------------

import json
import re

from infrastructure.handler import InputError
from model.model import read_pairs_csv

SCHEMA = Path(__file__).resolve().parent.parent / "schema" / "model.json"
BDC = "O=C(O)c1ccc(cc1)C(=O)O"


def _fresh() -> MOFSynthesizabilityModel:
    return MOFSynthesizabilityModel()  # models not loaded yet


def _run_csv(m: MOFSynthesizabilityModel, tmp_path: Path, text, **kw) -> dict:
    f = tmp_path / "in.csv"
    if isinstance(text, bytes):
        f.write_bytes(text)
    else:
        f.write_text(text, **kw)
    return m.process("dataset", {}, {"file": (f, "text/csv")}, tmp_path / "out", {}, LOG)


@pytest.mark.parametrize("smiles, why", [
    ("c1ccccc", "not valid SMILES syntax"),
    ("C(C)(C)(C)(C)C", "chemically invalid"),
])
def test_instance_bad_smiles_is_input_error(smiles: str, why: str) -> None:
    m = _fresh()
    with pytest.raises(InputError, match=rf"'{re.escape(smiles)}' is {why}"):
        m.process("instance", {"linker": smiles, "metal": "Zn"},
                  {}, Path("/tmp/mof_out"), {}, LOG)
    assert not m._loaded


@pytest.mark.parametrize("metal, why", [
    ("zn", "did you mean 'Zn'"),
    ("Zn2+", "not one of the metals"),
    ("Pt", "not one of the metals"),
])
def test_instance_unsupported_metal_lists_supported(metal: str, why: str) -> None:
    m = _fresh()
    with pytest.raises(InputError, match=rf"'{re.escape(metal)}' .*{why}.*Supported metals: Ag, Al"):
        m.process("instance", {"linker": BDC, "metal": metal},
                  {}, Path("/tmp/mof_out"), {}, LOG)
    assert not m._loaded


@pytest.mark.parametrize("text, message", [
    ("", "CSV file is empty"),
    ("linker\nCCO\n", "missing: 'metal'. Found columns: 'linker'"),
    ("linker;metal\nCCO;Zn\n", "missing: 'linker', 'metal'"),
    ("linker,metal\n", "header but no data rows"),
    ("linker,metal\n,,\n\n", "no data rows"),
    (f"linker,metal\n{BDC},Zn\n{BDC},\n,Cu\n", "row 3 has no metal; row 4 has no linker"),
    (f"linker,metal\n{BDC},Zn\nc1ccccc,Cu\n", "row 3 'c1ccccc' is not valid SMILES"),
    (f"linker,metal\n{BDC},Zn\n{BDC},Pt\n", "Unsupported metal: row 3 'Pt'"),
])
def test_dataset_bad_csv(tmp_path: Path, text: str, message: str) -> None:
    m = _fresh()
    with pytest.raises(InputError, match=message):
        _run_csv(m, tmp_path, text)
    assert not m._loaded


def test_dataset_not_utf8(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="UTF-8"):
        _run_csv(_fresh(), tmp_path, b"linker,metal\n\xff\xfe\x00,Zn\n")


def test_dataset_excel_bom_and_blank_rows(model: MOFSynthesizabilityModel, tmp_path: Path) -> None:
    # Excel writes a byte-order mark; blank rows are skipped.
    _run_csv(model, tmp_path, f" linker , metal \n{BDC},Zn\n,\n\n{BDC},Cu\n",
             encoding="utf-8-sig")
    with (tmp_path / "out" / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["metal"] for r in rows] == ["Zn", "Cu"]


def test_read_pairs_csv_row_numbers(tmp_path: Path) -> None:
    f = tmp_path / "in.csv"
    f.write_text(f"linker,metal,notes\n{BDC},Zn,x\n\n{BDC},Cu,y\n")
    assert read_pairs_csv(f) == [(2, BDC, "Zn"), (4, BDC, "Cu")]


def test_schema_metal_enum_matches_model(model: MOFSynthesizabilityModel) -> None:
    field = json.loads(SCHEMA.read_text())["execution"]["modes"]["instance"]["inputs"]["metal"]
    assert field["enum"] == model.supported_metals()


def test_schema_example_parses_and_predicts(model: MOFSynthesizabilityModel, tmp_path: Path) -> None:
    field = json.loads(SCHEMA.read_text())["execution"]["modes"]["dataset"]["inputs"]["file"]
    f = tmp_path / "example.csv"
    f.write_text(field["example"])
    rows = read_pairs_csv(f)
    assert 3 <= len(rows) <= 5
    assert [c["name"] for c in field["columns"]] == ["linker", "metal"]
    model.check_inputs([(r[1], r[2]) for r in rows])
    _run_csv(model, tmp_path, field["example"])
    assert (tmp_path / "out" / "predictions.csv").exists()
