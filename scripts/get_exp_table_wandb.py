import argparse
import os
from collections import OrderedDict
from typing import Dict, Tuple, List

import numpy as np
import wandb
import pandas as pd
from tqdm import tqdm


def get_all_runs(project) -> pd.DataFrame:
    api = wandb.Api()

    runs = api.runs(project)

    summary_list = []
    for run in tqdm(runs, "Fetching runs"):
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        curr_run_dict = {}
        curr_run_dict.update(run.summary._json_dict)
        curr_run_dict.update({k: v for k, v in run.config.items() if not k.startswith('_')})
        curr_run_dict["name"] = run.name
        curr_run_dict["url"] = run.url
        curr_run_dict["run"] = run

        summary_list.append(curr_run_dict)

    runs_df = pd.DataFrame.from_dict(summary_list)
    return runs_df


def get_ds_metric_name(runs_df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    ds_metric_map = {}
    for _, row in runs_df.iterrows():
        valid_metric_col = [col for col in row.index if col.startswith("best_") and col.endswith("_test_mean") and not np.isnan(row[col])]
        if len(valid_metric_col) == 0:
            continue
        assert len(valid_metric_col) == 1, f"There should be exactly one valid metric, found: {valid_metric_col}"
        valid_metric_col = valid_metric_col[0]
        ds_name = row["dataset"]
        metric_type = {"MAE": "min",
                       "ap": "max",
                       "rocauc": "max",
                       "acc": "max",
                       "accuracy": "max"}[valid_metric_col.split("_")[1]]
        ds_metric_map[ds_name] = (valid_metric_col, metric_type)
    return ds_metric_map


def get_best_run(runs_df: pd.DataFrame, ds_metric_name: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    with_ssma = runs_df[runs_df["ssma"] == 'true']
    no_ssma = runs_df[runs_df["ssma"] == 'false']

    # Get best run for each dataset
    runs = []
    for curr_df, is_ssma in ((with_ssma, True), (no_ssma, False)):
        for ds_name, ds_df in curr_df.groupby("dataset"):
            if ds_name not in ds_metric_name:
                continue
            for v in ds_df.groupby("layer"):
                layer_name, layer_df = v
                metric_name, metric_type = ds_metric_name[ds_name]

                expected_metric_name = metric_name

                best_run_idx = layer_df[expected_metric_name].agg(f"idx{metric_type}")
                if np.isnan(best_run_idx): # No valid run
                    continue
                best_run = layer_df.loc[best_run_idx]
                best_run[metric_name] = best_run[expected_metric_name]
                runs.append(best_run)
    runs_df = pd.DataFrame.from_records(runs)
    return runs_df


def color_result_table(row: pd.Series, ds_metric_name: Dict[str, Tuple[str, str]], ds_name: str) -> List[str]:
    res = []
    for layer_name, ssma in row.keys():
        val_without_ssma = row[(layer_name, "false")]
        val_with_ssma = row[(layer_name, "true")]

        if (ssma == "false" or
                ds_name not in ds_metric_name or
                val_with_ssma in (np.nan, None) or
                val_without_ssma in (np.nan, None)):
            curr_color = ""
            res.append(curr_color)
            continue

        _, metric_type = ds_metric_name[ds_name]

        val_with_ssma = float(val_with_ssma.split("±")[0])
        val_without_ssma = float(val_without_ssma.split("±")[0])

        if metric_type == "min" and val_with_ssma < val_without_ssma:
            curr_color = "background-color: green"
        elif metric_type == "max" and val_with_ssma > val_without_ssma:
            curr_color = "background-color: green"
        else:
            curr_color = "background-color: red"
        res.append(curr_color)

    return res


def prepare_output(args: argparse.Namespace, runs_df: pd.DataFrame, ds_metric_name: Dict[str, Tuple[str, str]]) -> None:
    agg_runs = []
    result_table_columns = []
    for layer in args.layers:
        for ssma in ('false', 'true'):
            result_table_columns.append((layer, ssma))
    result_table_columns = pd.MultiIndex.from_tuples(result_table_columns)
    result_table_indexes = args.datasets

    result_table = pd.DataFrame(index=result_table_indexes, columns=result_table_columns)

    for ds in args.datasets:
        for layer in args.layers:
            for ssma in ('true', 'false'):
                curr_df = runs_df[(runs_df["dataset"] == ds) & (runs_df["layer"] == layer) & (runs_df["ssma"] == ssma)]
                if len(curr_df) == 0:
                    print(f"No runs for dataset {ds}, layer {layer}, ssma {ssma}")
                    curr_row = {"dataset": ds, "layer": layer, "ssma": ssma}
                else:
                    assert len(curr_df) == 1, f"Should have exactly one run, found {len(curr_df)}"
                    curr_test_metric = curr_df.iloc[0][ds_metric_name[ds][0]]
                    curr_test_metric_std = curr_df.iloc[0][ds_metric_name[ds][0].replace("mean", "std")]
                    if curr_test_metric_std == 'NaN':
                        curr_test_metric_std = 0
                    result_table.loc[ds, (layer, ssma)] = f"{curr_test_metric:.4f}±{curr_test_metric_std:.4f}"
                    curr_row = OrderedDict()

                    for n in ("dataset", "layer", "hidden_dim", "depth", "ssma", "max_neighbors", "mlp_compression", "use_attention"):
                        curr_row[n] = curr_df.iloc[0][n]
                    curr_row["test_metric"] = curr_test_metric
                    curr_row["url"] = curr_df.iloc[0]["url"]

                    curr_run_obj = curr_df.iloc[0]["run"]
                    command = f"python {curr_run_obj.metadata['codePath']} {' '.join(curr_run_obj.metadata['args'])}"
                    curr_row["command"] = command

                    if curr_row["ssma"] == "false":
                        for n in ("max_neighbors", "mlp_compression", "use_attention"):
                            curr_row[n] = ""
                agg_runs.append(curr_row)
    runs_df = pd.DataFrame.from_dict(agg_runs)

    # Color the result table cells
    result_table = result_table.style.apply(lambda x: color_result_table(x, ds_metric_name, x.name), axis=1)

    # Generate the output excel
    if os.path.isfile(args.output):
        os.remove(args.output)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        result_table.to_excel(writer, sheet_name="results")
        runs_df.to_excel(writer, sheet_name="runs")

    # Generate output text (shell script for running the best runs)
    runs_df_ssma = runs_df[runs_df["ssma"] == "true"]
    with open(args.output.replace(".xlsx", ".sh"), "w") as f:
        for ds_name, ds_df in runs_df_ssma.groupby("dataset"):
            f.write("#" * 10 + f" {ds_name} " + "#" * 10 + "\n")
            for layer_name, layer_df in ds_df.groupby("layer"):
                assert len(layer_df) == 1
                layer_df = layer_df.iloc[0]
                f.write(f"# {layer_name} #\n")
                f.write(str(layer_df["command"]) + "\n")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",
                        type=str,
                        nargs="+",
                        default=["almogdavid/ssma"],
                        help="Project is specified by <entity/project-name>")
    parser.add_argument("--datasets",
                        type=str,
                        nargs="+",
                        default=("ogbg_molhiv", "ogbg_molpcba", "mutag", "enzymes", "proteins", "imdb_binary", "ptc_mr", "zinc" ,"peptides_func", "peptides_struct", "ogbn_arxiv", "ogbn_products"),
                        help="The datasets to grab")
    parser.add_argument("--layers",
                        type=str,
                        nargs="+",
                        default=("gcn", "gat", "gat2", "gin", "graphgps", "pna"),
                        help="The layers to grab")
    parser.add_argument("--output",
                        type=str,
                        default="experiments.xlsx",
                        help="The output file")
    args = parser.parse_args()

    runs_df = []
    for proj in args.project:
        runs_df.append(get_all_runs(proj))
    runs_df = pd.concat(runs_df)
    ds_metric_name = get_ds_metric_name(runs_df)
    runs_df = get_best_run(runs_df, ds_metric_name)
    prepare_output(args, runs_df, ds_metric_name)


if __name__ == "__main__":
    main()
