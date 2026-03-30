from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TimingEvent:
    """Single timing sample for one named section."""

    name: str
    elapsed_ms: float
    iteration: int | None = None


class TimedSection:
    """Context manager that records runtime for a named code block."""

    def __init__(
        self,
        profiler: TimingProfiler,
        name: str,
        *,
        iteration: int | None = None,
    ) -> None:
        self._profiler = profiler
        self._name = name
        self._iteration = iteration
        self._start_ns: int | None = None

    def __enter__(self) -> TimedSection:
        if self._profiler.enabled:
            self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if (not self._profiler.enabled) or (self._start_ns is None):
            return None

        elapsed_ms = (time.perf_counter_ns() - self._start_ns) * 1e-6
        self._profiler.record(
            self._name,
            elapsed_ms,
            iteration=self._iteration,
        )
        return None


class TimingProfiler:
    """Collect timing-only runtime data for named pipeline sections."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._events: list[TimingEvent] = []

    def section(self, name: str, *, iteration: int | None = None) -> TimedSection:
        return TimedSection(self, name, iteration=iteration)

    def record(self, name: str, elapsed_ms: float, *, iteration: int | None = None) -> None:
        if not self.enabled:
            return

        self._events.append(
            TimingEvent(
                name=name,
                elapsed_ms=float(elapsed_ms),
                iteration=iteration,
            )
        )

    @property
    def events(self) -> list[TimingEvent]:
        return list(self._events)

    def filter_events(self, name: str | None = None) -> list[TimingEvent]:
        if name is None:
            return self.events
        return [event for event in self._events if event.name == name]

    def to_rows(self, name: str | None = None) -> list[dict[str, float | int | str | None]]:
        rows: list[dict[str, float | int | str | None]] = []
        cumulative_ms = 0.0

        for event_index, event in enumerate(self.filter_events(name)):
            cumulative_ms += event.elapsed_ms
            rows.append(
                {
                    "event_index": event_index,
                    "name": event.name,
                    "iteration": event.iteration,
                    "elapsed_ms": event.elapsed_ms,
                    "cumulative_ms": cumulative_ms,
                }
            )

        return rows

    def save_json(self, path: str | Path, *, name: str | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_rows(name), handle, indent=2)

        return path

    def save_csv(self, path: str | Path, *, name: str | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = self.to_rows(name)
        fieldnames = [
            "event_index",
            "name",
            "iteration",
            "elapsed_ms",
            "cumulative_ms",
        ]

        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return path
