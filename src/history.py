from __future__ import annotations

from typing import Iterable

from slam_types import SLAMHistoryEntry


class SLAMHistory:
    """In-memory SLAM history keyed by step index."""

    def __init__(self):
        self._entries_by_step: dict[int, SLAMHistoryEntry] = {}

    def add(self, entry: SLAMHistoryEntry, *, overwrite: bool = False) -> None:
        step_index = int(entry.step_index)
        if (not overwrite) and (step_index in self._entries_by_step):
            raise KeyError(f"Step {step_index} already exists.")
        self._entries_by_step[step_index] = entry

    def get(self, step_index: int) -> SLAMHistoryEntry | None:
        return self._entries_by_step.get(int(step_index))

    def require(self, step_index: int) -> SLAMHistoryEntry:
        entry = self.get(step_index)
        if entry is None:
            raise KeyError(f"No entry for step_index={step_index}")
        return entry

    def latest(self) -> SLAMHistoryEntry | None:
        if not self._entries_by_step:
            return None
        latest_step = max(self._entries_by_step.keys())
        return self._entries_by_step[latest_step]

    def latest_or_raise(self) -> SLAMHistoryEntry:
        entry = self.latest()
        if entry is None:
            raise KeyError("No entries available.")
        return entry

    @property
    def step_indices(self) -> list[int]:
        return sorted(self._entries_by_step.keys())

    def __len__(self) -> int:
        return len(self._entries_by_step)

    def all_entries(self) -> list[SLAMHistoryEntry]:
        return [self._entries_by_step[step_index] for step_index in self.step_indices]

    def iter_entries(self) -> Iterable[SLAMHistoryEntry]:
        for step_index in self.step_indices:
            yield self._entries_by_step[step_index]
