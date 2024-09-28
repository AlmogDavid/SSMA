from typing import Optional

from torch import Tensor
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn import LSTMAggregation


class CustomLSTMAgg(LSTMAggregation):
    """
    Simple modification to LSTM aggregation to support inputs with more than 2 dimensions (from attention layers)
    """

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
            self,
            x: Tensor,
            index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = -2,
            max_num_elements: Optional[int] = None,
    ) -> Tensor:
        orig_x_shape = x.shape
        x = x.view(x.shape[0], -1)
        x_agg = super().forward(x, index, ptr, dim_size, dim, max_num_elements)
        x_agg = x_agg.view(x_agg.shape[0], *orig_x_shape[1:])
        return x_agg
