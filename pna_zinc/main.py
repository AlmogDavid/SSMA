import argparse
import copy
import os.path as osp

import torch
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pna_zinc.zinc_config import net_params, optimization_params
from pna_zinc.utils import compute_degree_stats, compue_batch_graph_norms
from pna_zinc.models import PNANet

def main(args: argparse.Namespace):
    exp_params = vars(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
    train_dataset = ZINC(path, subset=True, split='train')
    val_dataset = ZINC(path, subset=True, split='val')
    test_dataset = ZINC(path, subset=True, split='test')

    # Dataloaders
    batch_size = exp_params["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Degree stats
    exp_params["avg_d"] = compute_degree_stats(train_loader)

    # Model - linear search for best width
    hidden_dim = exp_params["hidden_dim"]
    edge_dim = exp_params["edge_dim"]
    tried_hidden = set()

    data_sample = next(iter(train_loader)).to(device)
    graph_norms_sample = compue_batch_graph_norms(data_sample)
    while hidden_dim not in tried_hidden:
        exp_params["hidden_dim"] = hidden_dim
        exp_params["edge_dim"] = edge_dim
        tried_hidden.add(hidden_dim)
        model = PNANet(exp_params).to(device)
        model(data_sample.x.squeeze(), data_sample.batch, data_sample.edge_index, graph_norms_sample, data_sample.edge_attr)# Initialize lazy modules
        model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f'Number of trainable parameters: {model_num_params} hidden_dim: {exp_params["hidden_dim"]} edge_dim: {exp_params["edge_dim"]}')

        if not exp_params["use_ssma"]:
            print("Not running using SSMA, not searching for better capacity")
            break

        if model_num_params < 100000:
            hidden_dim += 5
            edge_dim += 5
        else:
            hidden_dim -= 5
            edge_dim -= 5

        edge_dim = max(0, edge_dim)
        hidden_dim = max(0, hidden_dim)

    # Optimization
    init_lr = exp_params["init_lr"]
    weight_decay = exp_params["weight_decay"]
    lr_reduce_factor = exp_params["lr_reduce_factor"]
    lr_schedule_patience = exp_params["lr_schedule_patience"]
    min_lr = exp_params["min_lr"]

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=lr_reduce_factor,
                                  patience=lr_schedule_patience,
                                  min_lr=min_lr)

    def train(epoch):
        model.train()

        total_loss = 0
        for data in train_loader:
            graph_norms = compue_batch_graph_norms(data)
            data = data.to(device)
            graph_norms = graph_norms.to(device)
            optimizer.zero_grad()
            out = model(data.x.squeeze(), data.batch, data.edge_index, graph_norms, data.edge_attr)
            loss = (out.squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()

        total_error = 0
        for data in loader:
            graph_norms = compue_batch_graph_norms(data)
            data = data.to(device)
            graph_norms = graph_norms.to(device)
            out = model(data.x.squeeze(), data.batch, data.edge_index, graph_norms, data.edge_attr)
            total_error += (out.squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)

    best_val_mae = None
    best_val_test_mae = None
    best_loss_mae = None
    best_epoch_idx = -1

    num_epochs = exp_params["epochs"]
    with tqdm(range(1, num_epochs + 1), unit='epoch') as t:
        for epoch in t:
            loss = train(epoch)
            val_mae = test(val_loader)
            test_mae = test(test_loader)
            scheduler.step(val_mae)
            if not best_val_mae or val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_test_mae = test_mae
                best_loss_mae = loss
                best_epoch_idx = epoch

            if epoch - best_epoch_idx > exp_params["early_stopping"]:
                print("Early stopping, did not improve for more than 50 epochs")
                break

            t.set_description(f'Epoch {epoch:02d}  | Loss: {loss:.4f} | Val: {val_mae:.4f} | Test: {test_mae:.4f}')

    print(f"\nTEST MAE CORRESPONDING TO BEST VAL MAE: {best_val_test_mae:.4f}")
    return {"val_mae": best_val_mae, "test_mae": best_val_test_mae, "train_mae": best_loss_mae}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PNA model on the ZINC dataset.')
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to average the results over.")
    parser.add_argument("--use_ssma", type=str, default="false", help="Whether to use the SSMA implementation.")
    parser.add_argument("--mlp_compression", type=float, default=1.0, help="Compression rate of the MLP in SSMA.")
    parser.add_argument("--use_attention", type=str, default="false", help="Whether to use attention in SSMA.")
    parser.add_argument("--max_neighbors", type=int, default=2, help="Maximum number of neighbors to consider in SSMA layer.")

    for k, v in net_params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v, help=f"Parameter {k} of the model.")

    for k,v in optimization_params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v, help=f"Parameter {k} of the optimization.")

    args = parser.parse_args()
    args.use_ssma = args.use_ssma.lower() == "true"
    args.use_attention = args.use_attention.lower() == "true"

    wandb.init("ssma")

    agg_metrics = []
    try:
        for run_idx in range(args.runs):
            metrics = main(copy.deepcopy(args))
            agg_metrics.append(metrics)
    except KeyboardInterrupt:
        print('-=' * 45 + "-")
        print('Exiting from training early because of KeyboardInterrupt')
        print('-=' * 45 + "-")

    avg_metrics = {k + "_mean": sum(m[k] for m in agg_metrics) / len(agg_metrics) for k in agg_metrics[0]}
    std_metrics = {k + "_std": sum((m[k] - avg_metrics[k + "_mean"]) ** 2 for m in agg_metrics) / len(agg_metrics) for k in agg_metrics[0]}

    metrics = {**avg_metrics, **std_metrics}
    print("Summary:")
    for k,v in metrics.items():
        print(f"{k}: {v}")
    wandb.log(metrics)
    wandb.finish()


