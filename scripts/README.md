# Scripts

Programs in this directory use the importable `master_code` package but are not
part of its runtime API.

- `experiments/` contains repeated runs, benchmarks, and parameter sweeps.
- `plots/` contains standalone thesis figure and video generation programs.

Run scripts from the repository root with the project environment, for example:

```sh
uv run python scripts/experiments/runtime_multi_run.py --help
```
