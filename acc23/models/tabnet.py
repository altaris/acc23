"""
Implementation of TabNet from

    Arik, Sercan Ö., and Tomas Pfister. "Tabnet: Attentive interpretable
    tabular learning." Proceedings of the AAAI Conference on Artificial
    Intelligence. Vol. 35. No. 8. 2021.

See also:
    https://towardsdatascience.com/implementing-tabnet-in-pytorch-fc977c383279
"""
__docformat__ = "google"

from collections import OrderedDict
from dataclasses import dataclass
from math import sqrt
from typing import Tuple

import torch
from torch import Tensor, nn
from transformers.activations import get_activation

# class GhostBatchNormalization(nn.Module):
#     """
#     Ghost batch normalization from

#         Hoffer, Elad, Itay Hubara, and Daniel Soudry. "Train longer, generalize
#         better: closing the generalization gap in large batch training of
#         neural networks." Advances in neural information processing systems 30
#         (2017).
#     """

#     norm: nn.BatchNorm1d
#     virtual_batch_size: int

#     def __init__(
#         self,
#         num_features: int,
#         momentum: float = 0.01,
#     ):
#         super().__init__()
#         self.norm = nn.BatchNorm1d(num_features, momentum=momentum)
#         self.virtual_batch_size = virtual_batch_size

#     # pylint: disable=missing-function-docstring
#     def forward(self, x: Tensor, *_, **__) -> Tensor:
#         n_chunks = x.shape[0] // self.virtual_batch_size
#         chunks = [x] if n_chunks == 0 else torch.chunk(x, n_chunks, dim=0)
#         xns = [self.norm(c) for c in chunks]
#         return torch.cat(xns, 0)


class Sparsemax(nn.Module):
    """
    Sparsemax activation of

        Martins, Andre, and Ramon Astudillo. "From softmax to sparsemax: A
        sparse model of attention and multi-label classification."
        International conference on machine learning. PMLR, 2016.
    """

    def forward(self, z: Tensor) -> Tensor:
        s, _ = z.sort(dim=-1)  # z: (B, N), s: (B, N), s_{ij} = (z_i)_{(j)}
        with torch.no_grad():
            r = torch.arange(0, z.shape[-1])  # r = [0, ..., N-1]
            r = r.to(z.device)
            # w (B, N), w_{ik} = 1 if 1 + k z_{(k)} > sum_{j<=k} z_{(j)}, 0 otherwise
            w = (1 + r * s - s.cumsum(dim=-1) > 0).to(int)
            # r * w (B, N), (r * w)_{ik} = k if w_{ik} = 1, 0 otherwise. Therefore:
            # k_z (B,), (k_z)_i = largest index k st. w_{ik} > 0, i.e. st
            # 1 + k z_{(k)} > sum_{j<=k} z_{(j)}. Looks weird but trust me bro =)
            k_z = (r * w).max(dim=-1).values
            # m (B, N), m_{ik} = 1 if k <= (k_z)_i, 0 otherwise. Therefore:
            # m_z (B,), (m_z)_i = sum_{j <= (k_z)_i} z_{(j)}
            m = Tensor(
                [
                    [1] * int(k) + [0] * (z.shape[-1] - int(k))
                    for k in k_z + 1
                ]  # TODO: int(k) generates a warning...
            )
            m = m.to(z.device)
        m_z = (m * s).sum(dim=-1)
        tau_z = (m_z + 1) / k_z
        # p: (B, N), p_{ik} = [ z_{ik} - (tau_z)_i ]_+
        p = (z - tau_z.unsqueeze(-1)).clamp(0)
        return p


class SparseLoss(nn.Module):
    """
    Mean sparse regularization term of a batch of mask matrix. The sparse
    regularization term, or sparse loss, of a single mask matrix `m` is the
    average entropy of the rows of `m`.

    At least that's what the paper says it should be, but with large values of
    `n_a` and `n_d` this value is very large. In pactice, this returns
    $L_{sparse} / D$, where $D$ is the number of tabular features.
    """

    def forward(self, m: Tensor) -> Tensor:
        e = -m * (m + 1e-10).log()
        return e.mean().clamp(-100)
        # return e.sum(-1).mean()


class AttentiveTransformer(nn.Module):
    """Attentive transformer"""

    gamma: float
    linear: nn.Module
    norm: nn.Module
    activation: nn.Module

    def __init__(
        self,
        in_features: int,
        n_a: int,
        gamma: float = 1.5,
    ) -> None:
        """
        Args:
            in_features (int): Number of features of the tabular data
            n_a (int): Dimension of the attribute vector
            gamma (float): Relaxation parameter, denoted by $\\gamma$ in the
                paper
        """
        super().__init__()
        self.gamma = gamma
        self.linear = nn.Linear(n_a, in_features)
        self.norm = nn.BatchNorm1d(in_features)
        self.activation = Sparsemax()

    def forward(self, a: Tensor, ps: Tensor, **__) -> Tuple[Tensor, Tensor]:
        """
        Args:
            a (Tensor): The attribute tensor with shape `(N, n_a)`
            ps (Tensor): The prior scales tensor with shape
                `(N, in_features)`

        Returns:
            1. Mask with shape `(N, in_features)`
            2. Updated prior scales, also with shape `(N, in_features)`
        """
        a = self.linear(a)
        a = self.norm(a)
        m = self.activation(a * ps)
        return m, ps * (self.gamma - m.detach())


class GLUBlock(nn.Module):
    """
    A GLU block is simply a fully connected - batch normalization - GLU
    activation sequence
    """

    linear: nn.Module
    norm: nn.Module
    activation: nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * out_features)
        self.norm = nn.BatchNorm1d(2 * out_features)
        self.activation = nn.GLU()

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class FeatureTransformerSharedBlock(nn.Module):
    """Block that is shared among all feature transformers"""

    gb_1: nn.Module
    gb_2: nn.Module

    def __init__(self, in_features: int, n_a: int, n_d: int) -> None:
        """
        Args:
            in_features (int): Number of features of the tabular data, denoted
                by $D$ in the paper
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector
        """
        super().__init__()
        self.gb_1 = GLUBlock(in_features, n_a + n_d)
        self.gb_2 = GLUBlock(n_a + n_d, n_a + n_d)

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """
        Args:
            x (Tensor): Masked feature vector with shape `(N, in_features)`

        Returns:
            A `(N, n_a + n_d)` tensor, denoted by $\\mathbf{[a[i], d[i]]}$ in
            the paper.
        """
        x = self.gb_1(x)
        x = sqrt(0.5) * (x + self.gb_2(x))
        return x


class FeatureTransformerDependentBlock(nn.Module):
    """Block that is exclusive to each decision step's feature transformer"""

    gb_1: nn.Module
    gb_2: nn.Module

    def __init__(
        self,
        n_a: int,
        n_d: int,
    ) -> None:
        """
        Args:
            n_a (int): Dimension of the feature vector (to be passed down to
                the next decision step)
            n_d (int): Dimension of the output vector
        """
        super().__init__()
        self.gb_1 = GLUBlock(n_a + n_d, n_a + n_d)
        self.gb_2 = GLUBlock(n_a + n_d, n_a + n_d)

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        """Takes and returns a `(N, n_a + n_d)` tensor"""
        x = sqrt(0.5) * (x + self.gb_1(x))
        x = sqrt(0.5) * (x + self.gb_2(x))
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
        self.dependent_block = FeatureTransformerDependentBlock(n_a, n_d)

    # pylint: disable=missing-function-docstring
    def forward(self, x: Tensor, *_, **__) -> Tensor:
        x = self.shared_block(x)
        x = self.dependent_block(x)
        return x


@dataclass
class DecisionStepOutput(OrderedDict):
    """
    Structure to store the output of DecisionStep.forward

    TODO: Get rid of this, it doesn't simplify the graph of the model
    """

    a: Tensor
    """Attention vector"""
    d: Tensor
    """Decision vector"""
    ps: Tensor
    """Prior scales"""
    mf: Tensor
    """Mask features"""
    sl: Tensor
    """Sparse loss"""


class DecisionStep(nn.Module):
    """A single TabNet decision step"""

    activation: nn.Module
    attentive_transformer: nn.Module
    feature_transformer: nn.Module
    n_a: int
    sparse_loss: nn.Module

    def __init__(
        self,
        in_features: int,
        n_a: int,
        n_d: int,
        shared_feature_transformer_block: FeatureTransformerSharedBlock,
        gamma: float = 1.5,
        activation: str = "relu",
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
            activation (str): Defaults to relu
        """
        super().__init__()
        self.activation = get_activation(activation)
        self.attentive_transformer = AttentiveTransformer(
            in_features, n_a, gamma
        )
        self.feature_transformer = FeatureTransformer(
            n_a,
            n_d,
            shared_feature_transformer_block,
        )
        self.sparse_loss = SparseLoss()
        self.n_a, self.n_d = n_a, n_d

    def forward(self, x: Tensor, a: Tensor, ps: Tensor) -> DecisionStepOutput:
        """
        Args:
            x (Tensor): The (batch-normalized) input tabular data,
              denoted by $\\mathbf{f}$ in the paper, with shape `(N,
              in_features)`
            a (Tensor): Attribute vector of the previous decision step,
              denoted by $\\mathbf{a[i-1]}$ in the paper, whith shape `(N,
              n_a)`
            ps (Tensor): Prior scales, denoted by $\\mathbf{P[i-1]}$ in
              the paper, with shape `(N, in_features)`

        Returns:
            1. The attribute vector $\\mathbf{a[i]}$ of shape `(N, n_a)`
            2. The decision vector $\\mathbf{d[i]}$ of shape `(N, n_d)`
            3. Updated prior scales (for the attentive transformer)
               $\\mathbf{P[i]}$ of shape `(N, in_features)`
            4. The mask features $\\mathbf{\\eta[i] \\otimes \\M[i]}$
            5. The sparse loss
        """
        m, ps = self.attentive_transformer(a, ps)
        y = self.feature_transformer(x * m)
        a, d = y.split([self.n_a, self.n_d], dim=-1)
        d = self.activation(d)
        mf = d.sum(dim=-1).unsqueeze(dim=-1) * m
        sl = self.sparse_loss(m)
        # sl = (m**2).sum(-1).sum(-1)
        return DecisionStepOutput(a=a, d=d, ps=ps, mf=mf, sl=sl)


class TabNetEncoder(nn.Module):
    """TabNet encoder"""

    decision_steps: nn.ModuleList
    first_feature_transformer: nn.Module
    linear: nn.Module
    n_a: int
    n_d: int
    norm: nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_a: int,
        n_d: int,
        n_decision_steps: int = 3,
        gamma: float = 1.5,
        activation: str = "relu",
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
            activation (str): Defaults to relu
        """
        super().__init__()
        shared_block = FeatureTransformerSharedBlock(in_features, n_a, n_d)
        self.first_feature_transformer = FeatureTransformer(
            n_a, n_d, shared_block
        )
        self.decision_steps = nn.ModuleList(
            [
                DecisionStep(
                    in_features,
                    n_a,
                    n_d,
                    shared_block,
                    gamma,
                    activation,
                )
                for _ in range(n_decision_steps)
            ]
        )
        self.norm = nn.BatchNorm1d(in_features)
        self.linear = nn.Linear(n_d, out_features, bias=False)
        self.n_a, self.n_d = n_a, n_d

    def forward(self, x: Tensor, *_, **__) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x (Tensor): Unnormalized tabular data with shape `(N, in_features)`

        Returns:
            1. Output tensor with shape `(N, n_d)`
            2. Feature attribute tensor with shape `(N, in_features)`
            3. The sparse regularization term $L_{sparse}$
        """
        x = self.norm(x)
        a = self.first_feature_transformer(x)[:, : self.n_a]
        ps = torch.ones_like(x)
        ps.requires_grad = False
        d, mf, sl = Tensor([0]).to(x), Tensor([0]).to(x), Tensor([0]).to(x)
        for step in self.decision_steps:
            output = step(x, a, ps)
            a = output.a
            d += output.d
            ps = output.ps
            mf += output.mf
            sl += output.sl
        d = self.linear(d)
        mf = mf / (mf.sum() + 1e-5)
        sl /= len(self.decision_steps)
        return d, mf, sl


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
        """
        super().__init__()
        shared_block = FeatureTransformerSharedBlock(in_features, n_a, n_d)
        self.steps = nn.ModuleList(
            [
                nn.Sequential(
                    FeatureTransformer(n_a, n_d, shared_block),
                    nn.Linear(n_a + n_d, out_features),
                )
                for _ in range(n_steps)
            ]
        )

    # pylint: disable=missing-function-docstring
    def forward(self, z: Tensor, *_, **__) -> Tensor:
        xs = list(map(lambda step: step(z), self.steps))
        x = torch.stack(xs).sum(dim=0)
        return x
