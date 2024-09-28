import torch
from torch.utils.data import DataLoader

from torch_geometric.utils import degree

def compute_degree_stats(train_loader: DataLoader) -> dict:
    degress = torch.cat([degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_loader])
    return dict(lin=torch.mean(degress),
                exp=torch.mean(torch.exp(torch.div(1, degress)) - 1),
                log=torch.mean(torch.log(degress + 1)))


def compue_batch_graph_norms(data):
    nodes_per_graph = torch.bincount(data.batch)
    norm_per_graph = 1 / torch.sqrt(nodes_per_graph)
    graph_norms = norm_per_graph[data.batch].unsqueeze(1)
    return graph_norms
