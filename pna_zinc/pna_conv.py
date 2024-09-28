import torch
from torch import nn

import torch_scatter

from pna_zinc.aux_layers import FCLayer, MLP
from torch_geometric.nn.conv import MessagePassing

NUM_AGGREGATORS = 4
NUM_SCALERS = 3

class PNALayer(nn.Module):

    def __init__(self, in_dim, out_dim, avg_d, graph_norm, batch_norm, towers=1,
                 pretrans_layers=1, posttrans_layers=1, divide_input=True, residual=False, edge_features=False,
                 edge_dim=0, ssma_callback=None):
        """
        :param in_dim:              size of the input per node
        :param out_dim:             size of the output per node
        :param aggregators:         set of aggregation function identifiers
        :param scalers:             set of scaling functions identifiers
        :param avg_d:               average degree of nodes in the training set, used by scalers to normalize
        :param graph_norm:          whether to use graph normalisation
        :param batch_norm:          whether to use batch normalisation
        :param towers:              number of towers to use
        :param pretrans_layers:     number of layers in the transformation before the aggregation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        :param divide_input:        whether the input features should be split between towers or not
        :param residual:            whether to add a residual connection
        :param edge_features:       whether to use the edge features
        :param edge_dim:            size of the edge features
        """
        super().__init__()

        self.divide_input = divide_input
        self.input_tower = in_dim // towers if divide_input else in_dim
        self.output_tower = out_dim // towers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(PNATower(in_dim=self.input_tower, out_dim=self.output_tower,
                                        avg_d=avg_d, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, batch_norm=batch_norm,
                                        graph_norm=graph_norm, edge_features=edge_features, edge_dim=edge_dim, ssma_callback=ssma_callback))
        # mixing network
        self.mixing_network = FCLayer(out_dim, out_dim, activation='LeakyReLU')

    def forward(self, x, edge_index, snorm_n, edge_attr):
        h_in = x  # for residual connection

        if self.divide_input:
            h_cat = torch.cat(
                [tower(
                    x[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower],
                    edge_index,
                    snorm_n,
                    edge_attr)
                for n_tower, tower
                in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(x, edge_index, snorm_n, edge_attr) for tower in self.towers], dim=1)

        h_out = self.mixing_network(h_cat)

        if self.residual:
            h_out = h_in + h_out  # residual connection
        return h_out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)
    

class PNATower(MessagePassing):
    def __init__(self, in_dim, out_dim, graph_norm, batch_norm, avg_d,
                 pretrans_layers, posttrans_layers, edge_features, edge_dim, ssma_callback):
        self._ssma_agg = None
        if ssma_callback is None:
            super().__init__(node_dim=0)
        else:
            super().__init__(node_dim=0)
            self._ssma_agg = ssma_callback()


        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                            out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        curr_num_agg = NUM_AGGREGATORS + (self._ssma_agg is not None)
        self.posttrans = MLP(in_size=(curr_num_agg * NUM_SCALERS + 1) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d

    
    def message(self, x_i, x_j,
                edge_attr):

        h = x_i  # Dummy.
        if edge_attr is not None:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        h = self.pretrans(h)
        return h
    
    def aggregate(self, inputs, index):
        sums = torch_scatter.scatter_add(inputs, index, dim=0)
        maxs = torch_scatter.scatter_max(inputs, index, dim=0)[0]
        means = torch_scatter.scatter_mean(inputs, index, dim=0)
        var = torch.relu(torch_scatter.scatter_mean(inputs ** 2, index, dim=0) - means ** 2)
        
        aggrs = [sums, maxs, means, var]

        if self._ssma_agg is not None:
            ssma_agg_res = self._ssma_agg(inputs, index)
            aggrs.append(ssma_agg_res)

        c_idx = index.bincount().float().view(-1, 1)
        l_idx = torch.log(c_idx + 1.)
        
        amplification_scaler = [a * (l_idx / self.avg_d["log"]) for a in aggrs]
        attenuation_scaler = [a * (self.avg_d["log"] / l_idx) for a in aggrs]
        combinations = torch.cat(aggrs + amplification_scaler + attenuation_scaler, dim=1)
        return combinations

    def posttrans_nodes(self, x):
        return self.posttrans(x)

    def forward(self, x, edge_index, snorm_n, edge_attr):
        if self._ssma_agg is not None:
            self._ssma_agg.pre_aggregation_hook(self, (edge_index, None, {"x": x}))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = torch.cat([x, out], dim=1)
        x = self.posttrans_nodes(x)

        # graph and batch normalization
        if self.graph_norm:
            x = x * snorm_n
        if self.batch_norm:
            x = self.batchnorm_h(x)
        return x
