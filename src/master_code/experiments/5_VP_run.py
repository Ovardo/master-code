from datetime import datetime
from master_code.config import SlamConfig
from master_code.paths import RUNS_ROOT
from master_code.run_real import run_real



def main() -> None:
    config = SlamConfig.load("real.yaml")
    
    run_name = "VP_BENCHMARKING"
    run_covariance_method = 'OLD'
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = RUNS_ROOT / "experiments" / run_name / run_covariance_method / run_timestamp

    runs = [1, 2, 3, 4, 5]
    
    for run in runs:
        output_dir = experiment_dir / f"run_{run}"

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