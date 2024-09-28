import argparse

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt


def get_all_runs(ds: str) -> pd.DataFrame:
    api = wandb.Api()

    sweep = api.sweep("almogdavid/multiset_paper_runs_exp/1yk1e461")# todo: Remove names from the project

    runs = sweep.runs

    summary_list = []
    for run in runs:
        curr_run_dict = {}
        curr_run_dict.update(run.summary._json_dict)
        curr_run_dict.update({k: v for k, v in run.config.items() if not k.startswith('_')})
        curr_run_dict["name"] = run.name
        curr_run_dict["url"] = run.url

        summary_list.append(curr_run_dict)

    runs_df = pd.DataFrame.from_dict(summary_list)
    runs_df = runs_df[runs_df["dataset"] == ds]
    return runs_df


def get_relevant_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    best_runs = []
    runs_df["aggr"] = runs_df["aggr"].astype(str)
    for df_groups, group_df in runs_df.groupby(["layer_type", "aggr", "hidden_dim"]):
        group_df = group_df[group_df["best_MAE_test_mean"].isna() == False]
        if group_df.empty:
            print(f"Empty group: {df_groups}")
            continue
        #best_run = group_df.loc[group_df["best_accuracy_test_mean"].idxmax()]
        best_run = group_df.loc[group_df["best_MAE_test_mean"].idxmin()]
        best_runs.append(best_run)

    relevant_runs = pd.DataFrame.from_records(best_runs)
    return relevant_runs


def plot_ablation_data(runs_df: pd.DataFrame, ds_name) -> None:
    runs_df["hidden_dim"] = runs_df["hidden_dim"].apply(lambda x: np.log2(x))

    # Get unique values for layer_type and use_attention
    layer_types = runs_df['layer_type'].unique()

    # Generate colors dynamically
    num_layers = len(layer_types)

    # Create a colormap with unique colors for each combination of layer_type and use_attention
    color_map = plt.cm.get_cmap('Set1', num_layers)
    colors = {layer_types[i]: color_map(i) for i in range(num_layers)}

    # Plotting
    fig, ax = plt.subplots()

    #runs_df["best_accuracy_test_mean"] = runs_df["best_accuracy_test_mean"] * 100

    print(f"Dataset: {ds_name}")
    for idx, (layer_type, relevant_df) in enumerate(runs_df.groupby(['layer_type', 'aggr'])):
        layer, aggr = layer_type
        linestyle = '-' if aggr == "multiset" else '--'
        color = colors[layer]
        ax.plot(relevant_df['hidden_dim'], relevant_df['best_MAE_test_mean'], linestyle=linestyle,
                color=color, label=f'{layer}{"-deepset" if "multiset" in aggr else ""}')
        plt.scatter(relevant_df['hidden_dim'], relevant_df['best_MAE_test_mean'], color=color)

        str_to_print = f'{layer}-{aggr}: '
        for i, row in relevant_df.sort_values(by='hidden_dim', ascending=True).iterrows():
            str_to_print += f"{row['hidden_dim']}={row['best_MAE_test_mean']:.2f}, "
        print(str_to_print)

    # Legend outside of the figure
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save plot to PDF
    plt.savefig(f"circular_conv_ablation_{ds_name}.pdf", bbox_inches='tight')


def main():
    for ds in ["zinc"]:
        runs_df = get_all_runs(ds)
        #runs_df = runs_df[runs_df["layer_type"] != "pna"]
        runs_df = get_relevant_runs(runs_df)
        plot_ablation_data(runs_df, ds)


if __name__ == "__main__":
    main()