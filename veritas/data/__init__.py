"""Dataset builders register themselves on import; add new modules to _MODULES."""
from importlib import import_module

from .base import (  # noqa: F401
    DATA_DIR, PROCESSED_DIR, RAW_DIR, DatasetMeta, SplitData, TSDataset,
    available, build, load_dataset, load_processed,
)

_MODULES = [
    "synthetic",
    # healthcare
    "ptbxl", "sleepedf", "af2017",
    # activity
    "ucihar", "pamap2", "hhar", "uea",
    # industrial
    "cwru", "paderborn", "tep",
    # energy
    "ucr", "gpvs",
]

for _m in _MODULES:
    import_module(f"{__name__}.{_m}")
