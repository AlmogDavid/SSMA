import argparse
import copy
import inspect
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from omegaconf import DictConfig, OmegaConf
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing, GINConv, GATConv, PNAConv, GPSConv, GATv2Conv, GINEConv
from torch_scatter import scatter_sum, scatter_max, scatter_mean

from data.datasets import BaseDataset
from models.custom_modules import CustomLSTMAgg
from models.gen_agg.gen_agg import GenAgg
from ssma import SSMA
from utils import get_all_subclasses


class MLP(nn.Module):
    """
    This code taken from DGL gin layer implementation
    """
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super().__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class MLPReadout(nn.Module):
    """
    MLP Readout module.
    This code was taken from: https://github.com/YardenAdi-1/benchmarking-gnns/blob/4de55cdd9bfad3d4123ce15df7af29e83ecc7c8f/layers/mlp_readout_layer.py
    """

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class PYGMPNet(nn.Module):
    """
    Message Passing net based on pytorch geometric
    """

    def __init__(self, cfg: DictConfig, ds: BaseDataset):
        super().__init__()

        self.cfg = cfg
        hidden_dim = self.cfg.hidden_dim
        if self._is_uses_pe():
            self.cfg.hidden_dim -= cfg.rw_pe_depth_gps  # Modify for first embedding layer

        n_layers = cfg.depth
        self.batch_norm = cfg.batch_norm
        self.residual = cfg.residual
        self.readout = cfg.readout
        self.mid_net_dropout = nn.Dropout(cfg.dropout)

        layer_list = [self._get_layer(cfg, hidden_dim, hidden_dim, ds) for idx in range(n_layers)]
        self.layers = nn.ModuleList(layer_list)

        if self.batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.LazyBatchNorm1d() for _ in range(len(self.layers))])
        else:
            self.batch_norm_layers = nn.ModuleList([nn.Identity() for _ in range(len(self.layers))])

        self.MLP_layer = MLPReadout(hidden_dim, ds.output_dim())

        if cfg.layer_type == "graphgps" and cfg.use_pe_gps:
            self.pe_norm = BatchNorm1d(cfg.rw_pe_depth_gps)

    def _get_layer(self, cfg: DictConfig, in_dim: int, out_dim: int, ds) -> MessagePassing:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = argparse.Namespace(**cfg)
        aggr = copy.deepcopy(cfg.aggr)
        if isinstance(aggr, (tuple, list)) and len(aggr) == 1:
            aggr = aggr[0]
        layer_type = cfg.layer_type.lower()

        if isinstance(aggr, (list, tuple)):
            initialize_ssma = "SSMA" in aggr
            if initialize_ssma:
                aggr.remove("SSMA")
        else:
            initialize_ssma = aggr == "SSMA"
        if initialize_ssma:
            mlp_compression = cfg.mlp_compression
            num_neighbors_ssma = cfg.max_neighbors
            use_attention = cfg.use_attention
            temp = cfg.temp
            learn_affine = cfg.learn_affine
            ssma_params = dict(in_dim=out_dim,
                               mlp_compression=mlp_compression,
                               num_neighbors=num_neighbors_ssma,
                               use_attention=use_attention,
                               temp=temp,
                               learn_affine=learn_affine)
            if layer_type == "pna":
                ssma_params.update(dict(n_heads=cfg.towers_pna))

            if layer_type == "gat":
                n_heads = cfg.n_heads_gat
                ssma_params.update(dict(n_heads=n_heads, in_dim=out_dim // n_heads))

            if layer_type == "gat2":
                n_heads = cfg.n_heads_gat2
                ssma_params.update(dict(n_heads=n_heads, in_dim=out_dim // n_heads))

            ms_aggr = SSMA(**ssma_params)
            if layer_type == "pna":
                if isinstance(aggr, (list, tuple)):
                    aggr = [ms_aggr] + list(aggr)
                else:
                    aggr = ms_aggr
            else:
                aggr = ms_aggr
        elif aggr == "LSTM":
            aggr = CustomLSTMAgg(in_channels=in_dim, out_channels=out_dim)
        elif aggr == "GenAgg":
            aggr = GenAgg()
        if layer_type == "gcn":
            layer = GCNConv(in_channels=in_dim,
                            out_channels=out_dim,
                            add_self_loops=False,
                            aggr=aggr)
        elif layer_type == "gin":
            n_mlp_layers = cfg.n_mlp_GIN
            train_eps = cfg.learn_eps_GIN
            mlp = MLP(n_mlp_layers, in_dim, in_dim, out_dim)
            layer = GINConv(nn=mlp, aggr=aggr, train_eps=train_eps)
        elif layer_type == "gat":
            n_heads = cfg.n_heads_gat
            edge_dim = cfg.hidden_dim if cfg.use_edge_feat else None

            layer = GATConv(in_channels=in_dim,
                            out_channels=out_dim // n_heads,
                            add_self_loops=False,
                            heads=n_heads,
                            edge_dim=edge_dim,
                            aggr=aggr)
        elif layer_type == "gat2":
            n_heads = cfg.n_heads_gat2
            edge_dim = cfg.hidden_dim if cfg.use_edge_feat else None
            layer = GATv2Conv(in_channels=in_dim,
                              out_channels=out_dim // n_heads,
                              add_self_loops=False,
                              heads=n_heads,
                              edge_dim=edge_dim,
                              aggr=aggr)
        elif layer_type == "pna":
            if not isinstance(aggr, (list, tuple)):
                aggr = [aggr]
            scalers = cfg.scalers_pna
            towers = cfg.towers_pna
            deg = torch.as_tensor(cfg.avg_d_pna)  # Calculated at runtime by the calling script
            edge_dim = cfg.hidden_dim if cfg.use_edge_feat else None
            layer = PNAConv(in_channels=in_dim,
                            out_channels=out_dim,
                            aggregators=aggr,
                            scalers=scalers,
                            towers=towers,
                            edge_dim=edge_dim,
                            deg=deg)
        elif layer_type == "graphgps":
            assert in_dim == out_dim, "GraphGPS requires in_dim == out_dim"
            nn = Sequential(
                Linear(in_dim, in_dim),
                ReLU(),
                Linear(in_dim, in_dim),
            )
            if cfg.use_edge_feat and ds.edge_dim() > 0:
                mp_module = GINEConv(nn, aggr=aggr, edge_dim=cfg.hidden_dim)
            else:
                mp_module = GINConv(nn, aggr=aggr)
            layer = GPSConv(channels=in_dim,
                            conv=mp_module,
                            attn_type="multihead",
                            attn_kwargs={'dropout': 0.5},
                            heads=cfg.n_heads_gps)
        else:
            raise ValueError(f"Unsupported layer: {layer_type}")

        if initialize_ssma:
            if isinstance(layer, GPSConv):
                register_func = layer.conv.register_propagate_forward_pre_hook
            else:
                register_func = layer.register_propagate_forward_pre_hook
            register_func(ms_aggr.pre_aggregation_hook)

        return layer

    def _subsample_data(self, data: Data) -> Data:
        if "SSMA" in self.cfg.aggr and not self.cfg.use_attention:  # Random sample maximum number of neighbors
            edge_index = data.edge_index
            tgt_nodes = edge_index[1]
            unique_tgt_nodes = torch.unique(tgt_nodes)
            selected_edges = []
            for curr_tgt_node in unique_tgt_nodes:
                valid_edges = torch.argwhere(tgt_nodes == curr_tgt_node).reshape([-1])
                if len(valid_edges) > self.cfg.max_neighbors:
                    valid_edges = valid_edges[torch.randperm(len(valid_edges))[:self.cfg.max_neighbors]]
                selected_edges.append(valid_edges)

            selected_edges = torch.cat(selected_edges)
            edge_index = edge_index[:, selected_edges]
            data.edge_index = edge_index
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[selected_edges]
        return data

    def _is_uses_pe(self) -> bool:
        return self.cfg.layer_type == "graphgps" and self.cfg.use_pe_gps

    def forward(self, pyg_data: Data):
        has_pe = self._is_uses_pe()
        assert not (hasattr(pyg_data, "pe") ^ has_pe), "PE feature should be present if and only if the model uses it"
        pyg_data = self._subsample_data(pyg_data)
        if "LSTM" == self.cfg.aggr:
            pyg_data = pyg_data.sort(sort_by_row=False)  # Done for LSTM usage
        x = pyg_data.x

        edge_index = pyg_data.edge_index
        if hasattr(self, "embedding_e"):
            edge_attr = self.embedding_e(pyg_data.edge_attr)

        x = self.embedding_h(x).reshape(x.size(0), -1)

        if has_pe:
            pe_norm = self.pe_norm(pyg_data.pe)
            x = torch.cat([x, pe_norm], dim=-1)

        for layer_idx, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norm_layers)):
            x_in = x

            layer_call_args = dict(x=x, edge_index=edge_index)
            if hasattr(self, "embedding_e"):
                layer_call_args.update(dict(edge_attr=edge_attr))

            x = layer(**layer_call_args)
            x = batch_norm(x)
            x = F.relu(x)

            if self.residual and x.size(1) == x_in.size(1):  # Ignore last layer
                x = x + x_in

            x = self.mid_net_dropout(x)

        # Aggregate all nodes in a graph
        if self.readout == "sum":
            xg = scatter_sum(x, pyg_data.batch, dim=0)
        elif self.readout == "max":
            xg = scatter_max(x, pyg_data.batch, dim=0)[0]
        elif self.readout == "mean":
            xg = scatter_mean(x, pyg_data.batch, dim=0)
        elif self.readout is None:
            xg = x
        else:
            raise NotImplementedError(f"Unsupported readout: {self.readout}")

        pred = self.MLP_layer(xg)
        return pred

    @staticmethod
    def get_all_models() -> Dict[str, Type["PYGMPNet"]]:
        all_sons = get_all_subclasses(PYGMPNet)
        return {cls.__name__: cls for cls in all_sons}


class RegMPNet(PYGMPNet):
    def __init__(self, cfg: DictConfig, ds: BaseDataset):
        super().__init__(cfg=cfg, ds=ds)
        hidden_dim = cfg.hidden_dim
        self.embedding_h = nn.LazyLinear(hidden_dim)
        if cfg.use_edge_feat and ds.edge_dim() > 0:
            layer_call_parameters = inspect.signature(self.layers[0].forward).parameters
            if "edge_attr" in layer_call_parameters or isinstance(self.layers[0], GPSConv):
                self.embedding_e = nn.LazyLinear(hidden_dim)


class MolEmbNet(PYGMPNet):
    """
    Network for molecules datasets, generating embeddings using nn.Embeddings layers
    """

    def __init__(self, cfg: DictConfig, ds: BaseDataset):
        super().__init__(cfg=cfg, ds=ds)
        train_ds = ds.get_split("train")
        num_atom_type = torch.max(torch.stack([torch.max(train_ds[i].x) for i in range(len(train_ds))])).item() + 1
        hidden_dim = cfg.hidden_dim
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if cfg.use_edge_feat and ds.edge_dim() > 0:
            layer_call_parameters = inspect.signature(self.layers[0].forward).parameters
            if "edge_attr" in layer_call_parameters or isinstance(self.layers[0], GPSConv):
                num_bond_type = torch.max(
                    torch.stack([torch.max(train_ds[i].edge_attr) for i in range(len(train_ds))])).item() + 1
                self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)


class OGBGNet(PYGMPNet):
    """
    Network for ogbg datasets, using the ogb library for encoding atoms and bonds
    """

    def __init__(self, cfg: DictConfig, ds: BaseDataset):
        super().__init__(cfg=cfg, ds=ds)
        hidden_dim = cfg.hidden_dim
        self.embedding_h = AtomEncoder(hidden_dim)

        if cfg.use_edge_feat and ds.edge_dim() > 0:
            layer_call_parameters = inspect.signature(self.layers[0].forward).parameters
            if "edge_attr" in layer_call_parameters or isinstance(self.layers[0], GPSConv):
                self.embedding_e = BondEncoder(hidden_dim)


class OGBNNet(PYGMPNet):
    """
    Network for ogbn datasets
    """

    def __init__(self, cfg: DictConfig, ds: BaseDataset):
        assert cfg.readout is None, "When usign OGBN models, graph readout should be None"
        super().__init__(cfg=cfg, ds=ds)
        self.embedding_h = nn.LazyLinear(cfg.hidden_dim)  # Project to network dimension
