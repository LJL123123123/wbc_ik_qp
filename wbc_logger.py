"""wbc_logger.py

CSV logging utilities for WBC.

Goal: move all CSV output from `run_wbc.py` into `wbc.py`, so logging reflects
WBC's internal targets, solver outputs, and kinematic states.

This logger is intentionally lightweight: it opens files in append mode and
writes headers once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List
import csv


@dataclass
class CsvSeries:
    path: Path
    header: List[str]
    initialized: bool = False


@dataclass
class WbcCsvLogger:
    base_dir: Path | str = Path('./debug')
    series: Dict[str, CsvSeries] = field(default_factory=dict)

    def __post_init__(self):
        # allow passing base_dir as a string for convenience
        self.base_dir = Path(self.base_dir)

    def _ensure(self, name: str, filename: str, header: Iterable[str]):
        if name in self.series:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.series[name] = CsvSeries(path=self.base_dir / filename, header=list(header), initialized=False)

    def write_row(self, name: str, filename: str, header: Iterable[str], row: List[float]):
        self._ensure(name, filename, header)
        s = self.series[name]
        if not s.initialized:
            s.path.parent.mkdir(parents=True, exist_ok=True)
            with s.path.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(s.header)
            s.initialized = True
        with s.path.open('a', newline='') as f:
            w = csv.writer(f)
            w.writerow(row)
