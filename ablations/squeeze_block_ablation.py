import argparse
import copy
import random

import pandas as pd
import wandb
import matplotlib.pyplot as plt


def get_all_runs(sweep:str ) -> pd.DataFrame:

    api = wandb.Api()

    sweep = api.sweep(sweep)

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

    for _, neighbors_df in runs_df.groupby(["mlp_factorization_multiset", "layer_type"]):
        neighbors_df = neighbors_df[neighbors_df["best_accuracy_test_mean"].isna() == False]
        best_run = neighbors_df.loc[neighbors_df["best_accuracy_test_mean"].idxmax()]
        best_runs.append(best_run)

    relevant_runs = pd.DataFrame.from_records(best_runs)
    return relevant_runs


def plot_ablation_data(runs_df: pd.DataFrame, ds_name, ax) -> None:

    # Get unique values for layer_type and use_attention
    layer_types = runs_df['layer_type'].unique()

    # Generate colors dynamically
    num_layers = len(layer_types)

    # Create a colormap with unique colors for each combination of layer_type and use_attention
    color_map = plt.cm.get_cmap('Set1', num_layers)
    colors = {layer_types[i]: color_map(i) for i in range(num_layers)}

    # Plotting
    # fig, ax = plt.subplots()
    print("Dataset: ", ds_name)
    for layer_type, relevant_df in runs_df.groupby('layer_type'):
        color = colors[layer_type]
        relevant_df = relevant_df.sort_values(by='mlp_factorization_multiset', ascending=True)
        relevant_df['best_accuracy_test_mean'] = relevant_df['best_accuracy_test_mean'] * 100
        relevant_df['best_accuracy_test_std'] = relevant_df['best_accuracy_test_std'] * 100

        ax.errorbar(relevant_df['mlp_factorization_multiset'], relevant_df['best_accuracy_test_mean'],
                    yerr=relevant_df['best_accuracy_test_std'].apply(lambda x: x if isinstance(x, float) else random.random()),
                    color=color, capsize=5)

        ax.scatter(relevant_df['mlp_factorization_multiset'], relevant_df['best_accuracy_test_mean'], color=color, label=f'{layer_type}')

        str_to_print = f"{layer_type}: "
        for _, row in relevant_df.iterrows():
            str_to_print += f"{row['mlp_factorization_multiset']}={row['best_accuracy_test_mean']:.2f}, "

        print(str_to_print)

    # Legend outside of the figure
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # # Save plot to PDF
    # plt.savefig(f"squeeze_block_ablation_{ds_name}.pdf", bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser("This script generates the plots for the squeeze block ablation")
    parser.add_argument("--project", type=str, default="almogdavid/multiset_paper_runs_exp") # todo: remove the naming
    args = parser.parse_args()

    # Plotting
    all_runs = get_all_runs("almogdavid/multiset_paper_runs_exp/zk8ewolc")

    all_runs.sort_values(["dataset", "layer_type", "mlp_factorization_multiset"], inplace=True)
    all_runs[["dataset", "layer_type", "num_params", "mlp_factorization_multiset", "num_params", "best_accuracy_test_mean", "best_accuracy_test_std"]].to_csv(f"squeeze_block_ablation.csv")

    num_ds = all_runs["dataset"].nunique()

    fig, ax = plt.subplots(1, num_ds, sharey=False, sharex=True, figsize=(10*num_ds, 6))
    ax[0].set_ylabel("Test Accuracy")

    for idx, curr_ds in enumerate(all_runs["dataset"].unique()):
        runs_df = copy.deepcopy(all_runs)
        ax[idx].set_title(curr_ds)
        runs_df = get_relevant_runs(runs_df)
        ax[idx].set_xlabel("Factorization ratio")
        ax[idx].set_xscale('log')
        plot_ablation_data(runs_df, curr_ds, ax[idx])

    plt.subplots_adjust(wspace=0.1)
    fig.legend(loc='center left', fontsize='large')
    # Save plot to PDF
    plt.savefig(f"squeeze_block_ablation.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()