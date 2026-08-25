# Auto-generated from model.json - DO NOT EDIT
# Regenerate with: uv run model-types <path-to-model.json> -o src/model/types.py

from pathlib import Path
from typing import TypedDict, Literal

class InstanceInputs(TypedDict, total=False):
    # SMILES string of the organic linker
    linker: str
    # Elemental symbol of the metal, e.g. Zn, Cu, Fe (must be one of the supported metals)
    metal: str

class DatasetInputs(TypedDict, total=False):
    # CSV with 'linker' and 'metal' columns, one MOF per row
    file: tuple[Path, str]