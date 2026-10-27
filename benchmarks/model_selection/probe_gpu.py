"""Minimal owned allocation probe, before loading any candidate weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch

from geospatial_search.embedding_control import gpu_access
from geospatial_search.paths import JOB

print("GPU available:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
with gpu_access(JOB):
    print("Before ordinary allocation", flush=True)
    x = torch.ones((32, 32), device="cuda")
    torch.cuda.synchronize()
    print("GPU sum:", x.sum().item(), flush=True)
