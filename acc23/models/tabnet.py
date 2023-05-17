"""
Implementation of TabNet from

    Arik, Sercan Ö., and Tomas Pfister. "Tabnet: Attentive interpretable
    tabular learning." Proceedings of the AAAI Conference on Artificial
    Intelligence. Vol. 35. No. 8. 2021.

See also:
    https://towardsdatascience.com/implementing-tabnet-in-pytorch-fc977c383279
"""
__docformat__ = "google"

from typing import Tuple

import torch
from torch import Tensor, nn


class GhostBatchNormalization(nn.Module):
    """
    Ghost batch normalization from

        Hoffer, Elad, Itay Hubara, and Daniel Soudry. "Train longer, generalize
        better: closing the generalization gap in large batch training of
        neural networks." Advances in neural information processing systems 30
        (2017).
    """

    norm: nn.BatchNorm1d
    virtual_batch_size: int

    def __init__(
        self,
        num_features: int,
        virtual_batch_size: int = 128,
        momentum: float = 0.01,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features, momentum=momentum)
        self.virtual_batch_size = virtual_batch_size

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        n_chunks = x.shape[0] // self.virtual_batch_size
        chunks = [x] if n_chunks == 0 else torch.chunk(x, n_chunks, dim=0)
        xns = [self.norm(c) for c in chunks]
        return torch.cat(xns, 0)


class GLUBlock(nn.Module):
    """
    A GLU block is simply a fully connected - ghost batch normalization - GLU
    activation sequence
    """

    linear: nn.Module
    norm: nn.Module
    activation: nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        virtual_batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * out_features)
        self.norm = GhostBatchNormalization(
            2 * out_features, virtual_batch_size
        )
        self.activation = nn.GLU()

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class AttentiveTransformer(nn.Module):
    """Attentive transformer"""

    gamma: float
    linear: nn.Module
    norm: nn.Module

    def __init__(
        self,
        in_features: int,
        n_a: int,
        gamma: float = 1.5,
        virtual_batch_size: int = 128,
    ) -> None:
        """
        Args:
            in_features (int): Number of features of the tabular data
            n_a (int): Dimension of the attribute vector
            gamma (float): Relaxation parameter, denoted by $\\gamma$ in the
                paper
            virtual_batch_size (int): See `GhostBatchNormalization`
        """
        super().__init__()
        self.gamma = gamma
        self.linear = nn.Linear(n_a, in_features)
        self.norm = GhostBatchNormalization(in_features, virtual_batch_size)

    def forward(
        self, a: Tensor, prior_scales: Tensor, **__
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            a (Tensor): The attribute tensor with shape `(N, n_a)`
            prior_scales (Tensor): The prior scales tensor with shape
                `(N, in_features)`

        Returns:
            1. Mask with shape `(N, in_features)`
            2. Updated prior scales, also with shape `(N, in_features)`
        """
        a = self.linear(a)
        a = self.norm(a)
        m = sparsemax(a * prior_scales)
        prior_scales *= self.gamma - m
        return m, prior_scales


class FeatureTransformerSharedBlock(nn.Module):
    """Block that is shared among all feature transformers"""

    gb_1: nn.Module
    gb_2: nn.Module

    def __init__(
        self,
        in_features: int,
        n_a: int,
        n_d: int,
        virtual_batch_size: int = 128,
    ) -> None:
        """
        Args:
            in_features (int): Number of features of the tabular data, denoted
                by $D$ in the paper
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector
            virtual_batch_size: See `GhostBatchNormalization`
        """
        super().__init__()
        self.gb_1 = GLUBlock(in_features, n_a + n_d, virtual_batch_size)
        self.gb_2 = GLUBlock(n_a + n_d, n_a + n_d, virtual_batch_size)

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """
        Args:
            x (Tensor): Masked feature vector with shape `(N, in_features)`

        Returns:
            A `(N, n_a + n_d)` tensor, denoted by $\\mathbf{[a[i], d[i]]}$ in
            the paper.
        """
        s = Tensor([0.5], device=x.device).sqrt()
        x = self.gb_1(x)
        x = s * (x + self.gb_2(x))
        return x


class FeatureTransformerDependentBlock(nn.Module):
    """Block that is exclusive to each decision step's feature transformer"""

    gb_1: nn.Module
    gb_2: nn.Module

    def __init__(
        self,
        n_a: int,
        n_d: int,
        virtual_batch_size: int = 128,
    ) -> None:
        """
        Args:
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector
            virtual_batch_size: See `GhostBatchNormalization`
        """
        super().__init__()
        self.gb_1 = GLUBlock(n_a + n_d, n_a + n_d, virtual_batch_size)
        self.gb_2 = GLUBlock(n_a + n_d, n_a + n_d, virtual_batch_size)

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """Takes and returns a `(N, n_a + n_d)` tensor"""
        s = Tensor([0.5], device=x.device).sqrt()
        x = s * (x + self.gb_1(x))
        x = s * (x + self.gb_2(x))
        return x


class FeatureTransformer(nn.Module):
    """Feature transformer"""

    shared_block: nn.Module
    dependent_block: nn.Module

    def __init__(
        self,
        n_a: int,
        n_d: int,
        shared_block: FeatureTransformerSharedBlock,
        virtual_batch_size: int = 128,
    ) -> None:
        """
        Args:
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector
            shared_block: Feature transformer block to be shared across all
                decision steps.
        """
        super().__init__()
        self.shared_block = shared_block
        self.dependent_block = FeatureTransformerDependentBlock(
            n_a, n_d, virtual_batch_size
        )

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        x = self.shared_block(x)
        x = self.dependent_block(x)
        return x


class DecisionStep(nn.Module):
    """A single TabNet decision step"""

    activation: nn.Module
    attentive_transformer: nn.Module
    feature_transformer: nn.Module
    n_a: int

    def __init__(
        self,
        in_features: int,
        n_a: int,
        n_d: int,
        shared_feature_transformer_block: FeatureTransformerSharedBlock,
        gamma: float = 1.5,
        virtual_batch_size: int = 128,
    ) -> None:
        """
        Args:
            in_features (int): Number of features of the tabular data, denoted
                by $D$ in the paper
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector
            shared_feature_transformer_block: Feature transformer block to be
                shared across all decision steps.
            gamma (float): Relaxation parameter, denoted by $\\gamma$ in the
                paper
            virtual_batch_size: See `GhostBatchNormalization`
        """
        super().__init__()
        self.activation = nn.ReLU()
        self.attentive_transformer = AttentiveTransformer(
            in_features, n_a, gamma, virtual_batch_size
        )
        self.feature_transformer = FeatureTransformer(
            n_a,
            n_d,
            shared_feature_transformer_block,
            virtual_batch_size,
        )
        self.n_a = n_a

    # pylint: disable=missing-function-docstring
    def forward(
        self,
        x: Tensor,
        a: Tensor,
        prior_scales: Tensor,
        **__,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x (Tensor): The (batch-normalized) input tabular data, denoted by
                $\\mathbf{f}$ in the paper, with shape `(N, in_features)`
            a (Tensor): Attribute vector of the previous decision step, denoted
                by $\\mathbf{a[i-1]}$ in the paper, whith shape `(N, n_a)`
            prior_scales (Tensor):, denoted by $\\mathbf{P[i-1]}$ in the paper,
                with shape `(N, in_features)`

        Returns:
            1. The attribute vector $\\mathbf{a[i]}$ of shape `(N, n_a)`
            2. The output vector $\\mathbf{d[i]}$ of shape `(N, n_d)`
            3. Updated prior scales (for the attentive transformer)
               $\\mathbf{P[i]}$ of shape `(N, in_features)`
            4. The feature mask $\\mathbf{M[i]}$ of shape `(N, in_features)`
        """
        m, prior_scales = self.attentive_transformer(a, prior_scales)
        y = self.feature_transformer(x * m)
        a, d = y[:, : self.n_a], y[:, self.n_a :]
        d = self.activation(d)
        return a, d, prior_scales, m


class TabNetEncoder(nn.Module):
    """TabNet encoder"""

    decision_steps: nn.ModuleList
    feature_transformer: nn.Module
    linear: nn.Module
    norm: nn.Module
    n_a: int
    n_d: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_a: int,
        n_d: int,
        n_decision_steps: int = 3,
        gamma: float = 1.5,
        virtual_batch_size: int = 128,
    ) -> None:
        """
        Args:
            in_features (int): Number of features of the tabular data, denoted
                by $D$ in the paper
            out_features (int):
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector. The paper recommends
                `n_a = n_d`.
            n_decision_steps (int): The paper recommends between 3 and 10
            gamma (float): Relaxation parameter for the attentive transformers.
                The paper recommends larger values if the number of decision
                steps is large.
            virtual_batch_size: See `GhostBatchNormalization`
        """
        super().__init__()
        shared_block = FeatureTransformerSharedBlock(
            in_features, n_a, n_d, virtual_batch_size
        )
        self.feature_transformer = FeatureTransformer(
            n_a, n_d, shared_block, virtual_batch_size
        )
        self.decision_steps = nn.ModuleList(
            [
                DecisionStep(in_features, n_a, n_d, shared_block, gamma)
                for _ in range(n_decision_steps)
            ]
        )
        self.norm = GhostBatchNormalization(in_features, virtual_batch_size)
        self.linear = nn.Linear(n_d, out_features)
        self.n_a, self.n_d = n_a, n_d

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x (Tensor): Unnormalized tabular data with shape `(B, in_features)`

        Returns:
            1. Output tensor with shape `(B, n_d)`
            2. Feature attribute tensor with shape `(B, in_features)`
            3. The sparse regularization term $L_{sparse}$
        """
        x = self.norm(x)
        a = self.feature_transformer(x)[:, : self.n_a]
        prior_scales = torch.ones_like(x, dtype=x.dtype, device=x.device)
        ms, mfs, ds = [], [], []
        for step in self.decision_steps:
            a, d, prior_scales, m = step(x, a, prior_scales)
            ms.append(m)
            mfs.append(d.sum(dim=-1).unsqueeze(dim=-1) * m)
            ds.append(d)
        d = torch.stack(ds).sum(dim=0)
        m_agg = torch.stack(mfs).sum(dim=0)
        m_agg /= m_agg.sum()
        y = self.linear(d)
        l = torch.stack(list(map(sparse_regularization_term, ms))).mean()
        return y, m_agg, l


class TabNetDecoder(nn.Module):
    """TabNet decoder"""

    steps: nn.ModuleList

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_a: int,
        n_d: int,
        n_steps: int = 3,
        virtual_batch_size: int = 128,
    ) -> None:
        """
        Args:
            in_features (int): Number of features of the encoded representation
            out_features (int):
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector. The paper recommends
                `n_a = n_d`.
            n_steps (int): The paper recommends between 3 and 10
            virtual_batch_size: See `GhostBatchNormalization`
        """
        super().__init__()
        shared_block = FeatureTransformerSharedBlock(
            in_features, n_a, n_d, virtual_batch_size
        )
        self.steps = nn.ModuleList(
            [
                nn.Sequential(
                    FeatureTransformer(
                        n_a, n_d, shared_block, virtual_batch_size
                    ),
                    nn.Linear(n_a + n_d, out_features),
                )
                for _ in range(n_steps)
            ]
        )

    # pylint: disable=missing-function-docstring
    def forward(self, z: Tensor, *_, **__) -> Tensor:
        xs = map(lambda step: step(z), self.steps)
        x = torch.stack(list(xs)).sum(dim=0)
        return x


def sparse_regularization_term(m: Tensor) -> Tensor:
    """
    Sparse regularization term, which is essentially the entropy of the mask
    matrix `m` (more precisely, it's the average entropy of the rows of `m`).
    The log is clamped in $[-100, 0]$ for numerical stability.
    """
    e = -m * m.log().clamp(-100, 0)
    return e.mean(dim=0).sum()


def sparsemax(z: Tensor) -> Tensor:
    """
    Sparsemax activation of

        Martins, Andre, and Ramon Astudillo. "From softmax to sparsemax: A
        sparse model of attention and multi-label classification."
        International conference on machine learning. PMLR, 2016.
    """
    s, _ = z.sort(dim=-1)  # z: (B, N), s: (B, N), s_{ij} = (z_i)_{(j)}
    r = torch.arange(0, z.shape[-1])  # r = [0, ..., N-1]
    # w (B, N), w_{ik} = 1 if 1 + k z_{(k)} > sum_{j<=k} z_{(j)}, 0 otherwise
    w = (1 + r * s - s.cumsum(dim=-1) > 0).to(int)
    # r * w (B, N), (r * w)_{ik} = k if w_{ik} = 1, 0 otherwise. Therefore:
    # k_z (B,), (k_z)_i = largest index k st. w_{ik} > 0, i.e. st
    # 1 + k z_{(k)} > sum_{j<=k} z_{(j)}. Looks weird but trust me bro =)
    k_z = (r * w).max(dim=-1).values
    # m (B, N), m_{ik} = 1 if k <= (k_z)_i, 0 otherwise. Therefore:
    # m_z (B,), (m_z)_i = sum_{j <= (k_z)_i} z_{(j)}
    m = Tensor([[1] * int(k) + [0] * (z.shape[-1] - int(k)) for k in k_z + 1])
    m_z = (m * s).sum(dim=-1)
    tau_z = (m_z + 1) / k_z
    # p: (B, N), p_{ik} = [ z_{ik} - (tau_z)_i ]_+
    p = (z - tau_z.unsqueeze(-1)).clamp(0)
    return p
