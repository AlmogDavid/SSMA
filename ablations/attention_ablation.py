import argparse
import copy

import pandas as pd
import wandb
import matplotlib.pyplot as plt
import random


def get_all_runs(sweep: str) -> pd.DataFrame:
    api = wandb.Api()
    sweep = api.sweep(sweep)# todo: Remove names from the project

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
    return runs_df


def get_relevant_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    best_runs = []

    for layer_type, layer_df in runs_df.groupby("layer_type"):
        for is_use_attention in ("true", "false"):
            relevant_df = layer_df[layer_df["use_attention_multiset"] == is_use_attention]
            for curr_num_neighbors, neighbors_df in relevant_df.groupby("max_neighbors"):
                neighbors_df = neighbors_df[neighbors_df["best_accuracy_test_mean"].isna() == False]
                if neighbors_df.empty:
                    continue
                best_run = neighbors_df.loc[neighbors_df["best_accuracy_test_mean"].idxmax()]
                best_runs.append(best_run)

    relevant_runs = pd.DataFrame.from_records(best_runs)
    return relevant_runs

def plot_ablation_data(runs_df: pd.DataFrame, ds_name) -> None:
    # Get unique values for layer_type and use_attention
    layer_types = runs_df['layer_type'].unique()

    # Generate colors dynamically
    num_layers = len(layer_types)

    # Create a colormap with unique colors for each combination of layer_type and use_attention
    color_map = plt.cm.get_cmap('Set1', num_layers)
    colors = {layer_types[i]: color_map(i) for i in range(num_layers)}

    runs_df["best_accuracy_test_std"] = runs_df["best_accuracy_test_std"].fillna(0.005)

    # Plotting
    fig, ax = plt.subplots()

    runs_df["best_accuracy_test_mean"] = runs_df["best_accuracy_test_mean"] * 100
    runs_df["best_accuracy_test_std"] = runs_df["best_accuracy_test_std"] * 100

    runs_df[["dataset", "max_neighbors", "use_attention_multiset", "layer_type", "best_accuracy_test_mean", "best_accuracy_test_std"]].to_csv(f"attention_ablation_{ds_name}.csv")

    print("Dataset: ", ds_name)
    for idx, (layer_type, relevant_df) in enumerate(runs_df.groupby(['layer_type', 'use_attention_multiset'])):
        layer, use_attention = layer_type
        use_attention_bool = use_attention.lower() == "true"
        linestyle = '-' if use_attention.lower() == "true" else '--'

        color = colors[layer]

        relevant_df = relevant_df.sort_values(by='max_neighbors', ascending=True)

        ax.errorbar(relevant_df['max_neighbors'], relevant_df['best_accuracy_test_mean'],
                    yerr=relevant_df['best_accuracy_test_std'].apply(lambda x: x if isinstance(x, float) else random.random()), linestyle=linestyle,
                    color=color, label=f'{layer}-{"attention" if use_attention_bool else "random"}', capsize=5)

        plt.scatter(relevant_df['max_neighbors'], relevant_df['best_accuracy_test_mean'], color=color)

        str_to_print = f"{layer}-{use_attention}: "
        for row in relevant_df.iterrows():
            str_to_print += f"{row[1]['max_neighbors']}: {row[1]['best_accuracy_test_mean']:.2f}, "

        print(str_to_print)


    # Legend outside of the figure
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save plot to PDF
    plt.savefig(f"attention_ablation_{ds_name}.pdf", bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser("This script generates the plots for the neighbor selection method ablation")
    parser.add_argument("--project", type=str, default="almogdavid/multiset_paper_runs_exp") # todo: remove the naming
    args = parser.parse_args()

    all_runs = get_all_runs("almogdavid/multiset_paper_runs_exp/xczl47lo")
    all_runs = all_runs[all_runs["max_neighbors"] < 7]
    #all_runs = pd.concat([all_runs1, all_runs2])

    for curr_ds in ("ogbn_arxiv", "proteins"):
        runs_df = copy.deepcopy(all_runs)
        runs_df = runs_df[runs_df["dataset"] == curr_ds]
        runs_df = runs_df[runs_df["aggr"] == "multiset"]
        runs_df = get_relevant_runs(runs_df)
        plot_ablation_data(runs_df, curr_ds)

if __name__ == "__main__":
    main()