import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import OptTensor


class SSMADense(torch.nn.Module):
    """
    Sequential Signal Mixing Aggregation (SSMA) method for MPGNNs working on dense adjacency matrices.
    """

    def __init__(self,
                 in_dim: int,
                 num_neighbors: int,
                 mlp_compression: float = 1.0,
                 n_heads: int = 1,
                 temp: float = 1.0,
                 learn_affine: bool = False):
        """
        :param in_dim: The input dimension of the node features
        :param num_neighbors: Maximal number of neighbors to aggregate for each node
        :param mlp_compression: The compression ratio for the last MLP, if less than 1.0, the MLP will be factorized
        :param n_heads: Number of attention heads to use
        :param temp: The attention temperature to use
        :param learn_affine: If True, will learn the affine transformation, otherwise will use a fixed one.
        """
        super().__init__()

        self._in_dim = in_dim
        self._max_neighbors = num_neighbors
        self._mlp_compression = mlp_compression
        self._n_heads = n_heads
        self._learn_affine = learn_affine

        att_groups = n_heads * num_neighbors

        self.attn_l = nn.LazyLinear(att_groups, bias=True)
        self.attn_r = nn.LazyLinear(att_groups, bias=True)
        self._neighbor_att_temp = temp
        self._edge_attention_ste = None

        m1 = self._max_neighbors + 1
        m2 = int((in_dim - 1) * self._max_neighbors + 1)
        self._m1 = m1
        self._m2 = m2

        # Set frozen affine layer
        self._affine_layer = nn.Linear(in_features=in_dim, out_features=self._m1 * self._m2, bias=True)
        aff_w = torch.zeros(in_dim, self._m1 * self._m2)
        aff_b = torch.zeros(self._m1 * self._m2, dtype=torch.float32)
        aff_w[:in_dim, :in_dim] = -torch.eye(in_dim, dtype=torch.float32)
        aff_b[self._m2] = 1
        self._affine_layer.weight.data = aff_w.T
        self._affine_layer.bias.data = aff_b
        if not learn_affine:
            for p in self._affine_layer.parameters():
                p.requires_grad = False

        if mlp_compression < 1.0:  # Perform matrix factorization
            T = (mlp_compression * (self._m1 * self._m2 * in_dim)) / (self._m1 * self._m2 + in_dim)
            T = int(np.ceil(T))
            self._mlp = nn.Sequential(
                nn.Linear(in_features=self._m1 * self._m2, out_features=T),
                nn.Linear(in_features=T, out_features=in_dim)
            )
        else:
            self._mlp = nn.Linear(in_features=self._m1 * self._m2, out_features=in_dim, bias=True)

    def _compute_attention(self,
                           x: Tensor, adj: Tensor,
                           mask: OptTensor = None) -> Tensor:

        # Compute attention weights as in GAT, each input is devided to groups, each group has its own attention per number of neighbors
        # So we have  #groups * #neighbors attention weights
        B, N, _ = adj.size()
        x_l = self.attn_l(x)
        x_r = self.attn_r(x)

        x_l = x_l.reshape(B, N, self._max_neighbors, -1)  # [B, N, #neighbors, #groups]
        x_r = x_r.reshape(B, N, self._max_neighbors, -1)  # [B, N, #neighbors, #groups]

        # Compute softmax over the neighbors based on the attention weights and the graph topology
        edge_attention_scores = F.leaky_relu(
            x_l.unsqueeze(1) + x_r.unsqueeze(2)) / self._neighbor_att_temp
        if mask is not None:
            attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [B, N, N]
            attention_mask = attention_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, N, 1, 1]
            edge_attention_scores = edge_attention_scores.masked_fill(attention_mask == 0, -1e9)
        edge_attention_ste = F.softmax(edge_attention_scores, dim=2)  # [B, N, N, #neighbors, #groups]

        # Take the dense adjacenecy matrix into account
        edge_attention_ste = torch.clamp(edge_attention_ste * adj.reshape(B, N, N, 1, 1), min=1e-6)
        edge_attention_ste = edge_attention_ste / edge_attention_ste.sum(dim=2, keepdim=True)

        return edge_attention_ste

    def forward(self, x: Tensor, adj: Tensor,
                mask: OptTensor = None) -> Tensor:

        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        # Create index and new x based on the attention weights
        edge_att = self._compute_attention(x, adj, mask)  # [B, N, N, #neighbors, #groups]
        x = x.reshape(B, N, self._n_heads, -1)  # [B, N, #groups, group_size]
        x = torch.einsum('bjgk,bijng->bingk', x, edge_att)  # [B, N, #neighbors, #groups, group_size]
        x = x.reshape(B, N, self._max_neighbors, -1)  # [B, N, #neighbors, F]

        # Perform affine transformation
        x_aff = self._affine_layer(x)

        # Compute FFT
        x_aff = x_aff.reshape(B, N, self._max_neighbors, self._m1, self._m2)
        x_fft = torch.fft.fft2(x_aff)

        # Aggregate neighbors
        x_fft_abs = x_fft.abs()
        x_fft_abs_log = (x_fft_abs + 1e-6).log()
        x_fft_angle = x_fft.angle()

        x_fft_abs_agg = x_fft_abs_log.mean(dim=2).exp()
        x_fft_angle_agg = x_fft_angle.sum(dim=2)
        x_fft_agg = torch.polar(abs=x_fft_abs_agg, angle=x_fft_angle_agg)

        # Perform IFFT
        x_agg_comp = torch.fft.ifft2(x_fft_agg)
        x_agg = x_agg_comp.real

        # Perform MLP
        x_agg = x_agg.reshape(B, N, -1)
        x_agg_transformed = self._mlp(x_agg)

        if mask is not None:
            x_agg_transformed = x_agg_transformed * mask.view(B, N, 1).to(x_agg_transformed.dtype)
        return x_agg_transformed

    def __repr__(self) -> str:
        return "".join((f'SSMADense(in_dim={self._in_dim},'
                        f'num_neighbors={self._max_neighbors},'
                        f'mlp_compression={self._mlp_compression},'
                        f'n_heads={self._n_heads},'
                        f'temp={self._neighbor_att_temp},'
                        f'learn_affine={self._learn_affine})'))
