"""Checkout-relative defaults; source URIs in result metadata remain relative."""
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY / 'configs/nys-georsclip.json'
JOB = REPOSITORY / 'data/nys-naip-2022-rgb-060m'
IMAGERY = JOB / 'imagery'
EMBEDDINGS = JOB / 'embeddings/georsclip-vith14-s256-st192'
RESULTS = JOB / 'search-results'


def contained(root: Path, path: Path) -> Path:
    root, path = root.resolve(), path.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Path must stay inside {root}: {path}')
    return path
