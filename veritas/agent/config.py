"""Which data and claim type an agent experiment uses.

An agent experiment is named after its directory in results/agent/ (and its
bank in results/bank/). Most are named after their dataset and use single-cell
claims; a sensor-level experiment such as ``pamap2_group`` reads the
``pamap2`` data and claims one or two channel groups over the whole series.
make_views.py records this in results/agent/<name>/mode.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def agent_mode(name: str) -> tuple[str, str]:
    """(source dataset, claim type 'cell' | 'group') for an agent experiment."""
    path = ROOT / "results" / "agent" / name / "mode.json"
    if path.exists():
        m = json.loads(path.read_text())
        return m["source"], m["claims"]
    return name, "cell"
