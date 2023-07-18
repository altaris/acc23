"""Various stuff that don't fit in other modules"""

from typing import Dict, List

import torch
from torch import Tensor, nn
from transformers.activations import get_activation


class MLP(nn.Sequential):
    """A MLP (yep)"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        layer_norm: bool = True,
        activation: str = "gelu",
        dropout: float = 0.0,
        is_head: bool = False,
    ):
        """
        Args:
            in_dim (int):
            hidden_dims (List[int]): The last element is the output vector
                dimension
            layer_norm (bool):
            activation (str):
            dropout (float):
            is_head (bool): If set to `True`, there will be no layer
                normalization, activation, nor dropout after the last dense
                layer
        """
        ns = [in_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(1, len(ns)):
            a, b = ns[i - 1], ns[i]
            layers.append(nn.Linear(a, b))
            if not (is_head and i == len(ns) - 1):
                # The only time we don't add ln/act/drop is after the last
                # layer if the MLP is a head
                if layer_norm:
                    layers.append(nn.LayerNorm(b))
                layers.append(get_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


def concat_tensor_dict(d: Dict[str, Tensor]) -> Tensor:
    """
    Converts a dict of tensors to a tensor. The tensors (of the dict) are
    concatenated along the last axis. If they are 1-dimensional, they are
    unsqueezed along a new axis at the end. I know it's not super clear, so
    here is an example:

    >>> d = {"foo": torch.ones(3), "bar": torch.zeros((3, 2))}
    >>> concat_tensor_dict(d)
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]])

    Note that the `foo` tensor has been unsqueezed.
    """

    def _maybe_unsqueeze(t: Tensor) -> Tensor:
        return t if t.ndim > 1 else t.unsqueeze(-1)

    return torch.concatenate(list(map(_maybe_unsqueeze, d.values())), dim=-1)
