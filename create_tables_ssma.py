import csv
import os

import numpy as np
import pandas as pd
import wandb

BEST_RESULTS_0 = ["54.83±7.55 62.29±9.33 89.79±6.71 76.28±3.19 75.2±2.9 0.280±0.0", # GCN
                  "56.67±3.72 66.41±5.69 89.19±4.58 80.18±0.1 74.5±4.14 0.223±0.02", # GAT
                  "52.50±8.43 61.64±6.80 88.80±11.80 75.28±4.80 72.8±4.92 0.235±0.00", # GAT2
                  "51.69±8.04 61.28±9.23 90.51±6.97 75.19±4.73 74.1±5.02 0.222±0.00", # GIN
                  "49.17±3.15 63.02±4.93 86.07±7.95 75.56±4.24 71.1±4.79 0.22±0.00",] # GPS

BEST_RESULTS_1 = ["63.3±1.42 0.26±0.02 66.3±0.48 72.3±3.94 0.79±0.01 0.23±0.0", # GCN
                  "63.6±0.47 0.26±0.01 66.6±0.78 67.3±5.81 0.79±0.01 0.22±0.0", # GAT
                  "63.7±1.13 0.26±0.01 64.7±0.62 66.4±3.70 0.79±0.01 0.22±0.0", # GAT2
                  "62.5±1.37 0.26±0.02 66.4±1.52 67.0±5.79 0.78±0.01 0.22±0.0", # GIN
                  "60.34±1.49 0.27±0.01 66.71±0.73 67.62±5.46 0.78±0.01 0.22±0.0", #GPS
                  ]

BEST_RESULTS = [BEST_RESULTS_0, BEST_RESULTS_1]

ALL_DS_LIST = [["enzymes", "ptc_mr", "mutag", "proteins", "imdb_binary", "zinc"],
               ["peptides_func", "peptides_struct", "ogbn_arxiv", "ogbn_products", "ogbg_molhiv", "ogbg_molpcba"]]

LAYERS = ["gcn", "gat", "gat2", "gin", "graphgps"]


def handle_curr_aggr(curr_agg_df, curr_agg):
    for ds_idx, ds_list in enumerate(ALL_DS_LIST):
        data = [["layer"] + ds_list]
        for layer_idx, layer in enumerate(LAYERS):
            curr_layer_data = [layer]
            for curr_ds in ds_list:
                row = curr_agg_df[(curr_agg_df["layer"] == layer) & (curr_agg_df["dataset"] == curr_ds)]
                if len(row) < 1:
                    print(f"Skipping {curr_ds} - {layer} (no data)")
                    curr_layer_data.append("N/A")
                    continue
                elif len(row) > 1:
                    print(f"Skipping {curr_ds} - {layer} (multiple rows)")
                    curr_layer_data.append("N/A")
                    continue
                else:
                    row = row.iloc[0]
                    value_k = [k for k, v in row.items() if k.startswith("best") and isinstance(v, float) and not np.isnan(v)]
                    assert 0 < len(value_k) <= 2, "Should have at most two values"
                    metric_name = value_k[0].split("best_")[1].split("_test")[0]
                    mean_val = row[f"best_{metric_name}_test_mean"]
                    std_val = row[f"best_{metric_name}_test_std"]

                    if std_val == 'NaN' or np.isnan(std_val):
                        std_val = 0

                    if metric_name not in ("MAE") and curr_ds != "ogbg_molpcba":
                        mean_val *= 100
                        std_val *= 100

                    curr_val_str = f"{mean_val:.2f}±{std_val:.2f}"
                    curr_layer_data.append(curr_val_str)
            data.append(curr_layer_data)
            data.append([f"{layer} + SSMA"] + BEST_RESULTS[ds_idx][layer_idx].split(" "))

        # Write CSV to disk
        os.makedirs("ssma_exp_tables", exist_ok=True)
        with open(f'ssma_exp_tables/{curr_agg}_{ds_idx}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

        # Calc improvements in percentage
        improv = []
        for curr_idx in range(1, len(data), 2):
            row_without = data[curr_idx]
            row_with = data[curr_idx + 1]

            try:
                values_without = np.asarray([float(s.split("±")[0]) for s in row_without[1:]])
                values_with = np.asarray([float(s.split("±")[0]) for s in row_with[1:]])
            except ValueError:
                continue

            curr_improv = values_with/values_without
            curr_improv[-1] = 1 - curr_improv[-1]
            curr_improv[:-1] = curr_improv[:-1] - 1
            improv.append(curr_improv)

        total = np.mean(improv, axis=0)
        if ds_idx == 0:
            print(f"Improvements for {curr_agg}: {total}")


def main():
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    sweep = api.sweep("almogdavid/ssma/po26orbl")
    runs = sweep.runs

    run_data = []
    for run in runs:
        if run.state != "finished":
            continue

        curr_run_data = {k: run.config[k] for k in ('aggr', 'layer', 'dataset')}
        curr_run_data.update({k: run.summary[k] for k in ("best_accuracy_test_mean", "best_accuracy_test_std", "best_MAE_test_mean", "best_MAE_test_std", "best_ap_test_mean", "best_ap_test_std") if k in run.summary})

        if not any(k.startswith("best") for k in curr_run_data):
            continue

        # .name is the human-readable name of the run.
        run_data.append(curr_run_data)

    runs_df = pd.DataFrame.from_dict(run_data)

    # Filter runs which with SSMA
    runs_df = runs_df[["SSMA" not in s for s in runs_df["aggr"].astype(str)]]

    # Per aggregation
    for curr_agg, curr_agg_df in runs_df.groupby("aggr"):
        print(f"Aggregation: {curr_agg}")
        handle_curr_aggr(curr_agg_df, curr_agg)


if __name__ == "__main__":
    main()