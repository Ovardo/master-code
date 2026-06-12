# Configurations

These YAML files override defaults from `master_code.config.SlamConfig`.

- `real_default.yaml` is the baseline Victoria Park configuration.
- `sim_default.yaml` is the baseline simulated-data configuration.
- `default_config.yaml` documents the complete default schema.
- The remaining files capture specific experiments.

Pass a configuration explicitly with `run_real --config <path>` or
`run_sim --config <path>`. Every run stores its resolved configuration in the
run directory.
