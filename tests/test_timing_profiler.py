from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from timing_profiler import TimingProfiler


class TimingProfilerTests(unittest.TestCase):
    def test_section_records_elapsed_time_and_iteration(self) -> None:
        profiler = TimingProfiler(enabled=True)

        with profiler.section("extract_covariance", iteration=14):
            time.sleep(0.001)

        events = profiler.filter_events("extract_covariance")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].iteration, 14)
        self.assertGreater(events[0].elapsed_ms, 0.0)

    def test_rows_include_cumulative_runtime(self) -> None:
        profiler = TimingProfiler(enabled=True)
        profiler.record("extract_covariance", 10.0, iteration=1)
        profiler.record("extract_covariance", 20.0, iteration=2)

        rows = profiler.to_rows("extract_covariance")

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["elapsed_ms"], 10.0)
        self.assertAlmostEqual(rows[0]["cumulative_ms"], 10.0)
        self.assertAlmostEqual(rows[1]["cumulative_ms"], 30.0)

    def test_json_and_csv_export(self) -> None:
        profiler = TimingProfiler(enabled=True)
        profiler.record("process_step", 15.0, iteration=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = profiler.save_csv(tmp_path / "timing.csv")
            json_path = profiler.save_json(tmp_path / "timing.json")

            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("process_step", csv_path.read_text(encoding="utf-8"))

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["iteration"], 8)
            self.assertAlmostEqual(payload[0]["elapsed_ms"], 15.0)


if __name__ == "__main__":
    unittest.main()
