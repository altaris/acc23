class MyModule(nn.Module):
    """My torch module"""

    def __init__(self) -> None:
        super().__init__()

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        pass




class VisionTransformerBlock(nn.Module):
    """
    Vision transformer block which actually includes the 'transformer' part of
    this whole endeavor in the form of a multihead attention module.
    """

    norm_1: nn.Module
    norm_2: nn.Module
    mha: nn.Module
    mlp: nn.Module

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            embed_dim (int):
            num_heads (int):
            dropout (float): Applied in both the MHA module and the MLP head
            activation (str): Defaults to `gelu`
        """
        super().__init__()
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor) -> Tensor:
        u = self.norm_1(x)
        x = self.mha(u, u, u, need_weights=False)[0] + x
        x = self.mlp(self.norm_2(x)) + x
        return x
