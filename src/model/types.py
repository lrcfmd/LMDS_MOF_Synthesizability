# Auto-generated from model.json - DO NOT EDIT
# Regenerate with: uv run model-types <path-to-model.json> -o src/model/types.py

from pathlib import Path
from typing import TypedDict, Literal

class InstanceInputs(TypedDict, total=False):
    # SMILES string of the organic linker, e.g. O=C(O)c1ccc(cc1)C(=O)O
    linker: str
    # Element symbol of the metal (only metals the models were trained on)
    metal: Literal['Ag', 'Al', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'Fe', 'Ga', 'Gd', 'Hf', 'Hg', 'Ho', 'In', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'Na', 'Nd', 'Ni', 'Pb', 'Pr', 'Rb', 'Ru', 'Sb', 'Sc', 'Sm', 'Sn', 'Sr', 'Tb', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']

class DatasetInputs(TypedDict, total=False):
    # CSV of MOFs, one per row: the linker SMILES and the metal's element symbol
    file: tuple[Path, str]