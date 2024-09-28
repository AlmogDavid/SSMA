# Taken from: https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/examples/graph_gps.py

import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch_geometric.transforms as T
import wandb
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention

from ssma import SSMA

HEADS_GPS = 4


class GPS(torch.nn.Module):
    def __init__(self, channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any], run_args: argparse.Namespace):
        super().__init__()

        self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if run_args.use_ssma:
                aggr = SSMA(in_dim=channels,
                            num_neighbors=run_args.max_neighbors_ssma,
                            mlp_compression=run_args.compression_ssma,
                            use_attention=run_args.attention_ssma)
            else:
                aggr = "add"
            mp_module = GINEConv(nn, aggr=aggr)
            if run_args.use_ssma:
                mp_module.register_propagate_forward_pre_hook(aggr.pre_aggregation_hook)
            conv = GPSConv(channels, mp_module, heads=HEADS_GPS,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x = global_add_pool(x, batch)
        return self.mlp(x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


def get_num_trainable_params(model: nn.Module) -> int:
    num_trainable_params = 0
    for name, param in model.named_parameters():
        try:
            if param.requires_grad:
                num_trainable_params += param.numel()
        except ValueError:
            print(f"Failed to compute number of parameters for parameter named {name}")
            raise
    return num_trainable_params


def create_model(args: argparse.Namespace, train_loader: DataLoader) -> GPS:
    data = next(iter(train_loader))
    if not args.use_pe:
        data.pe = torch.zeros(data.num_nodes, 20)
    print("Searching for highest hidden dimension with the parameter budget")
    low = 32
    high = 128
    mid = 0
    while low < high:
        new_mid = (low + high) // 2
        new_mid = new_mid - new_mid % HEADS_GPS

        if new_mid == mid:  # No progress
            break
        mid = new_mid

        model = GPS(channels=mid, pe_dim=28, num_layers=10, attn_type=args.attn_type,
                    attn_kwargs=attn_kwargs, run_args=args)
        out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                    data.batch)
        num_params = get_num_trainable_params(model)
        print(f"Trying width: {mid}, num params: {num_params}")
        if num_params > args.parameter_budget:
            high = mid
        else:
            low = mid + 1
    print(f"Selected model width: {mid}")
    return model.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--attn_type', default='multihead',
        help="Global attention type such as 'multihead' or 'performer'.")
    parser.add_argument('--use_pe', type=str, default="false")
    parser.add_argument('--use_ssma', type=str, default="false")
    parser.add_argument('--attention_ssma', type=str, default="false")
    parser.add_argument('--compression_ssma', type=float, default=0.1)
    parser.add_argument('--max_neighbors_ssma', type=int, default=2)
    parser.add_argument('--parameter_budget', type=int, default=500000)
    args = parser.parse_args()
    print("Input arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    wandb.init(config=vars(args))

    args.use_pe = args.use_pe.lower() == "true"
    args.use_ssma = args.use_ssma.lower() == "true"
    args.attention_ssma = args.attention_ssma.lower() == "true"

    if not args.use_ssma:
        if not (args.compression_ssma == 0.1 and args.attention_ssma == False and args.max_neighbors_ssma == 2):
            print("Skipping run since this run is duplicated")
            exit(0)

    if args.use_pe:
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ZINC-PE')
    else:
        transform = None
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ZINC')

    train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
    val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
    test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_kwargs = {'dropout': 0.5}
    model = create_model(args, train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
    #                               min_lr=0.00001)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, 1e-3, 1, 50),
                                         CosineAnnealingLR(optimizer, 2000)], milestones=[50])


    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            model.redraw_projection.redraw_projections()
            if not args.use_pe:
                data.pe = torch.zeros(data.num_nodes, 20).to(device)

            out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                        data.batch)
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
            data = data.to(device)
            if not args.use_pe:
                data.pe = torch.zeros(data.num_nodes, 20).to(device)
            out = model(data.x, data.pe, data.edge_index, data.edge_attr,
                        data.batch)
            total_error += (out.squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)


    for epoch in range(1, 2001):
        loss = train()
        val_mae = test(val_loader)
        test_mae = test(test_loader)
        # scheduler.step(val_mae)
        scheduler.step()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
              f'Test: {test_mae:.4f}')

    wandb.log({"loss": loss, "val_mae": val_mae, "test_mae": test_mae})
    wandb.finish()
