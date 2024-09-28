import os

import pandas as pd
import wandb


def main():
    # Get all the runs
    api = wandb.Api()

    runs = api.runs("almogdavid/vpa_hps")

    summary_list = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        curr_run_dict = {}
        curr_run_dict.update(run.summary._json_dict)
        curr_run_dict.update({k: v for k, v in run.config.items() if not k.startswith('_')})
        curr_run_dict["name"] = run.name
        curr_run_dict["url"] = run.url

        summary_list.append(curr_run_dict)

    runs_df = pd.DataFrame.from_dict(summary_list)
    
    results = []

    for ds_name, ds_df in runs_df.groupby("dataset_name"):
        for agg_name, agg_df in ds_df.groupby("agg"):
            for _, row in agg_df.iterrows():
                test_acc_mean = row["final_test_acc_mean"]
                test_acc_std = row["final_test_acc_std"]
                layer = row["name"].split("_")[0]
                results.append({"dataset": ds_name,
                                "layer": layer,
                                "agg": agg_name,
                                "test_acc_mean": test_acc_mean,
                                "test_acc_std": test_acc_std})

    results = pd.DataFrame.from_dict(results)

    best_res_df = results.groupby(["layer", "dataset"]).max("test_acc_mean")

    if os.path.exists("vpa_results.xlsx"):
        os.remove("vpa_results.xlsx")
    with pd.ExcelWriter("vpa_results.xlsx", engine="openpyxl") as writer:
        best_res_df.to_excel(writer, sheet_name="vap_results")


if __name__ == "__main__":
    main()