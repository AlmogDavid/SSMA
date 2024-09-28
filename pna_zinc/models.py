import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ssma import SSMA
from pna_zinc.pna_conv import PNALayer
from pna_zinc.aux_layers import MLPReadout

"""
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
    Architecture follows that in https://github.com/graphdeeplearning/benchmarking-gnns
"""


class PNANet(nn.Module):

    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.towers = net_params['towers']
        self.divide_input_first = net_params['divide_input_first']
        self.divide_input_last = net_params['divide_input_last']
        self.edge_feat = net_params['edge_feat']
        self.avg_d = net_params['avg_d']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, edge_dim)

        self.layers = nn.ModuleList()
        ssma_creation_callback = lambda: self._create_ssma_agg(in_dim=hidden_dim // self.towers,
                                                               mlp_compression=net_params["mlp_compression"],
                                                               use_attention=net_params["use_attention"],
                                                               max_neighbors=net_params["max_neighbors"])
        for _ in range(n_layers - 1):
            curr_l = PNALayer(in_dim=hidden_dim, out_dim=hidden_dim,
                              graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                              residual=self.residual,
                              avg_d=self.avg_d, towers=self.towers, edge_features=self.edge_feat,
                              edge_dim=edge_dim, divide_input=self.divide_input_first,
                              pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers,
                              ssma_callback=ssma_creation_callback if net_params["use_ssma"] else None)
            self.layers.append(curr_l)

        self.layers.append(PNALayer(in_dim=hidden_dim, out_dim=out_dim,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual,
                                    avg_d=self.avg_d, towers=self.towers, divide_input=self.divide_input_last,
                                    edge_features=self.edge_feat, edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers,
                                    ssma_callback=ssma_creation_callback if net_params["use_ssma"] else None))

        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

    def _create_ssma_agg(self, in_dim, mlp_compression, use_attention, max_neighbors):
        return SSMA(in_dim=in_dim,
                    num_neighbors=max_neighbors,
                    mlp_compression=mlp_compression,
                    use_attention=use_attention)

    def forward(self, x, batch, edge_index, snorm_n, edge_attr):
        x = self.embedding_h(x)
        if self.edge_feat:
            edge_attr = self.embedding_e(edge_attr)

        for conv_layer in self.layers:
            x = conv_layer(x, edge_index, snorm_n, edge_attr)

        if self.readout == "sum":
            x = global_add_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "mean":
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)  # default readout is mean nodes

        return self.MLP_layer(x)

    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss
