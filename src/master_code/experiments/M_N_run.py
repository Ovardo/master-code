
from datetime import datetime
from master_code.config import SlamConfig
from master_code.paths import RUNS_ROOT
from master_code.run_real import run_real


def main() -> None:
    experiment_name = "M_vs_N"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = SlamConfig.load("real.yaml")
    
    runs = [
        {
            "name": "M1_N4",
            "M": 1,
        },
        {
            "name": "M2_N4",
            "M": 2,
        },
        {
            "name": "M3_N4",
            "M": 3,
        },
        {
            "name": "M4_N4",
            "M": 4,
        },
    ]

    experiment_dir = RUNS_ROOT / "experiments" / experiment_name / timestamp

    for run in runs:
        config.tentative.M = run["M"]
        output_dir = experiment_dir / run["name"]

        run_real(
            config,
            output_dir=output_dir,
            num_steps=7300,
            show_plots=False,
            save_plots=True,
        )

    print(f"Finished experiment: {experiment_dir.resolve()}")


if __name__ == "__main__":
    main()