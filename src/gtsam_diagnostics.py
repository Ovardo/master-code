from __future__ import annotations

import argparse
import importlib
import platform
import sys


def _normalize_expected_path(module_name: str, expected: str) -> str:
    if expected.startswith(f"{module_name}."):
        return expected[len(module_name) + 1 :]
    return expected


def _resolve_attr_path(obj: object, attr_path: str) -> object:
    current = obj
    for part in attr_path.split("."):
        current = getattr(current, part)
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that the expected GTSAM Python module and wrapper symbols are available."
    )
    parser.add_argument(
        "--module",
        default="gtsam",
        help="Python module to import. Defaults to 'gtsam'.",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        help=(
            "Attribute path that must exist after import, for example "
            "'CustomWrapper' or 'gtsam.CustomWrapper'. Repeat for multiple checks."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Module:   {args.module}")

    try:
        module = importlib.import_module(args.module)
    except Exception as exc:
        print(f"Import failed: {exc.__class__.__name__}: {exc}")
        return 1

    print(f"Location: {getattr(module, '__file__', '<unknown>')}")

    module_version = getattr(module, "__version__", None)
    if module_version is not None:
        print(f"Version:  {module_version}")

    missing: list[str] = []
    for expected in args.expect:
        attr_path = _normalize_expected_path(args.module, expected)
        try:
            _resolve_attr_path(module, attr_path)
        except AttributeError:
            missing.append(expected)
        else:
            print(f"Found:    {args.module}.{attr_path}")

    if missing:
        print("Missing expected symbols:")
        for expected in missing:
            print(f"  - {expected}")
        return 2

    print("Import check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
