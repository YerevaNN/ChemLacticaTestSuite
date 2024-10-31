import submitit


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/saturn/app/job_%j")
    executor.update_parameters(
        name="chemlactica-saturn-app", timeout_min=120 * 60,
        nodes=1, cpus_per_task=120,
    )
    function = submitit.helpers.CommandFunction([
        'python3', 'run_docking_oracle_app.py'
    ])

    executor.submit(function)