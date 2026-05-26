"""Compatibility entry point for running the simulated dataset."""
from __future__ import annotations

import sys

from master_code.run import main as run_main


def main() -> None:
    args = sys.argv[1:]
    if not any(arg == "--dataset" or arg.startswith("--dataset=") for arg in args):
        args = ["--dataset", "simulated", *args]

    sys.argv = [sys.argv[0], *args]
    run_main()


if __name__ == "__main__":
    main()
