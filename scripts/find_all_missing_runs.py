from typing import List, Dict, Any
import os
import itertools

import pandas as pd
import wandb
import yaml


def get_all_sweeps() -> List[str]:
    sweep_folder = os.path.join(os.path.dirname(__file__), "paper_hps")
    all_sweeps = [os.path.join(sweep_folder, f) for f in os.listdir(sweep_folder) if os.path.join(sweep_folder, f).endswith(".yaml")]
    return all_sweeps


def get_all_runs_from_sweep(sweep_file_loc: str) -> List[Dict[str, Any]]:
    with open(sweep_file_loc, "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    param_names = list(sweep_config["parameters"].keys())
    param_values = [[(param, v) for v in sweep_config["parameters"][param]["values"]] for param in param_names]
    all_combinations = list(itertools.product(*param_values))

    # Print all runs
    runs_params = []
    for i, params in enumerate(all_combinations):
        run = {}
        for param in params:
            run[param[0]] = param[1]
        runs_params.append(run)

    return runs_params


def find_missing_runs(expected_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    api = wandb.Api()

    runs = api.runs("almogdavid/multiset_paper_runs_exp")

    summary_list = []
    for run in runs:
        curr_run_dict = {}
        curr_run_dict.update(run.summary._json_dict)
        curr_run_dict.update({k: v for k, v in run.config.items() if not k.startswith('_')})
        curr_run_dict["name"] = run.name
        curr_run_dict["url"] = run.url
        if run.state != "finished":
            continue # Skip runs that are not finished
        try:
            summary_list.append({"dataset": curr_run_dict["dataset"],
                                 "layer": curr_run_dict["layer"],
                                 "max_neighbors": curr_run_dict["max_neighbors"],
                                 "mlp_factorization_multiset": curr_run_dict["mlp_factorization_multiset"],
                                 "multiset": curr_run_dict["multiset"],
                                 "runs": curr_run_dict["runs"],
                                 "use_attention_multiset": curr_run_dict["use_attention_multiset"]})
        except KeyError:  # Meaning the run was incomplete
            pass

    # Convert dict to list of tuples
    order = list(expected_runs[0].keys())
    expected_runs_set = set([tuple(list(((k, d[k]) for k in order))) for d in expected_runs])
    existing_runs_set = set([tuple(list(((k, d[k]) for k in order))) for d in summary_list])

    missing_runs = [dict(t) for t in expected_runs_set - existing_runs_set]
    missing_runs = sorted(missing_runs, key=lambda x: x["dataset"] + "_" + x["layer"])
    return missing_runs


def generate_run_command(run: Dict[str, Any]) -> str:
    run_command = f"scripts/run_single_exp.sh "
    for k, v in run.items():
        run_command += f"{k}={v} "
    return run_command


def main():
    all_sweeps = get_all_sweeps()
    all_runs = []
    for sweep in all_sweeps:
        sweep_runs = get_all_runs_from_sweep(sweep)
        all_runs.extend(sweep_runs)
    missing_runs = find_missing_runs(all_runs)
    if len(missing_runs):
        with open("missing_runs.sh", "w") as out_file:
            for run in missing_runs:
                run_command = generate_run_command(run)
                print(run_command)
                out_file.write(run_command + "\n")
            print(f"Found {len(missing_runs)} missing runs")


if __name__ == "__main__":
    main()
