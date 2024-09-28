import argparse

import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt


def get_all_runs() -> pd.DataFrame:
    api = wandb.Api()
    summary_list = []
    for sweep_id in ("7rb6mf1w", "wt439iqx"):
        sweep = api.sweep(f"almogdavid/multiset_paper_runs_exp/{sweep_id}")
        runs = sweep.runs

        for run in runs:
            curr_run_dict = {}
            curr_run_dict.update(run.summary._json_dict)
            curr_run_dict.update({k: v for k, v in run.config.items() if not k.startswith('_')})
            curr_run_dict["name"] = run.name
            curr_run_dict["url"] = run.url

            summary_list.append(curr_run_dict)

    runs_df = pd.DataFrame.from_dict(summary_list)
    return runs_df


def get_relevant_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    best_runs = []
    runs_df["use_multiset"] = runs_df["aggr"].apply(lambda x: "multiset" in x)
    for df_groups, group_df in runs_df.groupby(["layer_type", "use_multiset", "parameter_budget"]):
        group_df = group_df[group_df["best_MAE_test_mean"].isna() == False]
        if group_df.empty:
            print(f"Empty group: {df_groups}")
            continue
        best_run = group_df.loc[group_df["best_MAE_test_mean"].idxmin()]
        best_runs.append(best_run)

    relevant_runs = pd.DataFrame.from_records(best_runs)
    relevant_runs = relevant_runs[["use_multiset", "layer_type", "parameter_budget", "best_MAE_test_mean"]] # Just for clarity
    return relevant_runs


def plot_ablation_data(runs_df: pd.DataFrame) -> None:
    runs_df.to_csv("circular_conv_ablation_params.csv", index=False)
    runs_df["parameter_budget"] = runs_df["parameter_budget"].apply(lambda x: np.log2(x))

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

    print(f"Dataset: ZINC")
    for idx, (layer_type, relevant_df) in enumerate(runs_df.groupby(['layer_type', 'use_ssma'])):
        layer, use_ssma = layer_type
        linestyle = '-' if use_ssma else '--'
        color = colors[layer]
        ax.plot(relevant_df['parameter_budget'], relevant_df['best_MAE_test_mean'], linestyle=linestyle,
                color=color, label=f'{layer}{"-deepset" if use_ssma else ""}')
        ax.set_xlabel('Parameter budget (log2)')
        ax.set_ylabel('MAE')
        plt.scatter(relevant_df['parameter_budget'], relevant_df['best_MAE_test_mean'], color=color)

        str_to_print = f'{layer}-{"deepset" if use_ssma else "add"}: '
        for i, row in relevant_df.sort_values(by='parameter_budget', ascending=True).iterrows():
            str_to_print += f"{row['parameter_budget']}={row['best_MAE_test_mean']:.2f}, "
        print(str_to_print)

    # Legend outside of the figure
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save plot to PDF
    plt.savefig(f"circular_conv_ablation_parameters_zinc.pdf", bbox_inches='tight')


def main():
    runs_df = get_all_runs()
    runs_df = get_relevant_runs(runs_df)
    plot_ablation_data(runs_df)


if __name__ == "__main__":
    main()