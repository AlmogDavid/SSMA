import argparse
import os

import numpy as np
import pandas as pd
import torch_geometric
import yaml
from torch_geometric.transforms import AddRandomWalkPE

from data.datasets import BaseDataset


def analyze_single_ds(ds: BaseDataset):
    num_nodes = 0
    num_edges = 0
    num_graphs = 0
    num_cls = 0
    in_degrees = []
    if ds.is_graph_level():
        splits = ("train", "valid", "test")
    else:
        splits = ("all", )
    for curr_ds in [ds.get_split(s) for s in splits]:
        num_graphs += len(curr_ds)
        for data in curr_ds:
            d_num_nodes = data.num_nodes if data.x is None else data.x.size(0)
            d_degree = torch_geometric.utils.degree(data.edge_index[1], num_nodes=d_num_nodes)
            in_degrees.append(d_degree.numpy())
            num_nodes += d_num_nodes
            num_edges += data.edge_index.size(1)
            num_cls = max(num_cls, data.y.max().item())

    stat = {"avg_nodes": num_nodes / num_graphs,
            "avg_edges": num_edges / num_graphs,
            "num_graphs": num_graphs,
            "in_dim": ds.node_dim(),
            "edge_dim": ds.edge_dim(),
            "num_classes": ds.output_dim()}

    # Compute degree statistics
    in_degrees = np.concatenate(in_degrees)
    avg = np.mean(in_degrees)
    std = np.std(in_degrees)
    max_ = np.max(in_degrees)
    min_ = np.min(in_degrees)
    q25, q50, q75, q90, q95, q99 = np.quantile(in_degrees, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    deg_stat = {"avg": avg, "std": std, "max": max_, "min": min_, "q25": q25, "q50": q50, "q75": q75, "q90": q90,
            "q95": q95, "q99": q99}
    stat.update({f"deg_{k}": v for k,v in deg_stat.items()})

    return stat


def main():
    dataset_map = BaseDataset.get_all_ds()

    parser = argparse.ArgumentParser(description='Get statistics of datasets')
    parser.add_argument('--datasets', default=tuple(dataset_map.keys()), type=str, nargs='+', help='The datasets to get statistics of')
    parser.add_argument('--output', type=str, default="ds_statistics.xlsx", help='The output file to save the statistics to')
    parser.add_argument('--prepare', action='store_true', help='Whether to prepare the datasets (compute PE)')
    args = parser.parse_args()

    assert args.output.endswith('.xlsx'), 'The output file must be an excel file (must end with .xlsx)'

    for ds in args.datasets:
        assert ds in dataset_map, f"The dataset {ds} is not a valid dataset ({list(dataset_map.keys())})"

    agg_values = []

    pre_transform = None
    if args.prepare:
        # Open base.yaml and read default pe value from there
        with open(os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"), 'r') as file:
            yaml_content = yaml.safe_load(file)
            pe_depth = yaml_content["rw_pe_depth_gps"]
        pre_transform = AddRandomWalkPE(walk_length=pe_depth, attr_name='pe')  # Should be synced with train.py action

    for ds_name in args.datasets:
        print(f"Working on DS: {ds_name}")
        # Load reg
        ds = dataset_map[ds_name]()
        if pre_transform is not None:  # Load again with PE
            print("Loading with PE")
            ds = dataset_map[ds_name](pre_transform=pre_transform)
        curr_row = analyze_single_ds(ds)
        curr_row["name"] = ds_name
        agg_values.append(curr_row)

    df = pd.DataFrame.from_dict(agg_values)
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('name')))
    df = df[cols]
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    numeric_cols = df.select_dtypes(include=[np.number])
    df[numeric_cols.columns] = numeric_cols.map(lambda x: round(x, 2))
    print(df)

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ds_statistics")


if __name__ == "__main__":
    main()
