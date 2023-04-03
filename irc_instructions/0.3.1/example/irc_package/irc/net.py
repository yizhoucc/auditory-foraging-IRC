import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, TensorDataset
from gym.spaces import MultiDiscrete, Box
from typing import Optional, Union
from collections.abc import Callable, Iterable

from jarvis.config import Config
from jarvis.utils import create_mlp_layers, sgd_optimizer

from . import rcParams
from .distribution import BaseDistribution, _create_distribution
from .alias import Array, Module, Tensor, VarSpace
from .utils import sub2ind, get_dtype, train_and_summarize


class BaseDistributionNet(Module):
    r"""Base class for a distribution net.

    The distribution net models a conditional distribution P(x|y1, y2, ...) by
    a standard deep neural network. The distribution parameter is first computed
    from input ys, then used to characterize a distribution for x.

    """

    def __init__(self,
        x_space: VarSpace,
        y_spaces: Iterable[VarSpace],
        p_x: Union[BaseDistribution, dict, None] = None,
    ):
        r"""
        Args
        ----
        x_space:
            Space for variable x in P(x|y1, y2, ...).
        y_spaces:
            Spaces for variables y in P(x|y1, y2, ...).
        p_x:
            Distribution or a configuration specifying one for variable x, used
            only as a dummy to interpret distribution parameters.

        """
        super().__init__()
        self.x_space = x_space
        self.y_spaces = y_spaces
        self.p_x = _create_distribution(p_x, self.x_space)

    def forward(self, *ys: Iterable[Tensor]) -> Tensor:
        r"""Returns the distribution parameters.

        Args
        ----
        ys:
            The list of input tensors. Each element is a 2D tensor of shape
            (num_samples, y_dim).

        Returns
        -------
        param_vecs: (num_samples, param_dim)
            Parameter vectors for `p_x`.

        """
        raise NotImplementedError

    def _get_input_tensor(self, *ys: Iterable[Tensor]) -> Tensor:
        r"""Converts and concatenates input tensors.

        MultiDiscrete space inputs are converted to a list of one-hot vectors,
        while Box space inputs are unchanged. All inputs are later concatenated
        to form a single input tensor.

        Args
        ----
        ys:
            The list of input tensors. See `forward` for more details.

        Returns
        -------
        out: (num_samples, in_features)
            Converted and concatenated input tensor, used to feed into the deep
            neural network.

        """
        out = []
        for y_space, y in zip(self.y_spaces, ys):
            if isinstance(y_space, MultiDiscrete):
                for i, dim in enumerate(y_space.nvec):
                    out.append(one_hot(y[:, i], dim).to(torch.float))
            if isinstance(y_space, Box):
                out.append(y)
        out = torch.cat(out, dim=1)
        return out

    def _train(self,
        dset: Dataset,
        get_param_and_loss: Callable[..., tuple[Tensor, Tensor]],
        ws: Optional[Array] = None,
        *,
        l2_reg: Optional[float] = None,
        lr: Optional[float] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> dict:
        r"""Trains the distribution net.

        This method will be called to either estimate from data pairs or regress
        toward a target net.

        Args
        ----
        dset:
            Dataset for network training, provided by `estimate` or `regress`.
        get_param_and_loss:
            A function that returns two tensors: `param_vecs` and `raw_loss`.
            `param_vecs` is the distribution parameters of data pair samples in
            a batch, with shape (batch_size, param_dim). `raw_loss` is a scalar
            tensor, either the negative log likelihood loss in `estimate` or the
            KL divergence loss in `regress`.
        ws, l2_reg, lr, num_epochs, batch_size:
            Optimization parameters, similar to those in
            `BaseDistribution.estimate`.
        kwargs:
            Additional arguments for `train_and_summarize`.

        Returns
        -------
        train_stats:
            The training statistics returned by `train_and_summarize`, with some
            additional information.

        """
        _rcParams = Config(rcParams.get('net.BaseDistributionNet._train'))
        l2_reg = l2_reg or _rcParams.l2_reg
        lr = lr or _rcParams.lr
        kwargs = Config(kwargs)
        kwargs.fill({
            'num_epochs': num_epochs or _rcParams.num_epochs,
            'batch_size': batch_size or _rcParams.batch_size,
        })

        def get_loss(*batch):
            param_vecs, raw_loss = get_param_and_loss(*batch)
            l2_loss = param_vecs.pow(2).mean()
            loss = (1-l2_reg)*raw_loss+l2_reg*l2_loss
            return loss
        optimizer = sgd_optimizer(self, lr=lr)
        device = next(iter(self.parameters())).device

        stats = {'l2_reg': l2_reg, 'lr': lr}
        stats.update(train_and_summarize(
            dset, get_loss, optimizer, ws=ws, device=device, **kwargs,
        ))
        return stats

    def estimate(self,
        xs: Array,
        ys: Iterable[Array],
        **kwargs,
    ) -> dict:
        r"""Estimates p(x|y) from samples.

        Args
        ----
        xs: (num_samples, x_dim)
            x data samples.
        ys: [(num_samples, y_dim)]
            y data samples. Similar to `ys` in `forward`, except as Array.
        kwargs:
            Additional arguments for `_train`.

        Returns
        -------
        estimate_stats:
            See `_train` for more details.

        """
        _rcParams = Config(rcParams.get('net.BaseDistributionNet.estimate'))
        kwargs = Config(kwargs)
        kwargs.fill(_rcParams)
        dset = TensorDataset(
            torch.tensor(xs, dtype=get_dtype(self.x_space)),
            *[
                torch.tensor(ys[i], dtype=get_dtype(self.y_spaces[i]))
                for i in range(len(ys))
            ],
        )
        def get_param_and_loss(xs, *ys):
            param_vecs = self.forward(*ys)
            nll_loss = -torch.cat([
                self.p_x.loglikelihoods(xs[i][None], param_vecs[i])
                for i in range(len(xs))
            ]).mean()
            return param_vecs, nll_loss
        return self._train(dset, get_param_and_loss, **kwargs)

    def regress(self,
        targets: Array,
        ys: Iterable[Array],
        **kwargs,
    ) -> dict:
        r"""Regresses p(x|y) to targets.

        Args
        ----
        targets: (num_samples, param_dim)
            Target distribution parameters.
        ys: [(num_samples, y_dim)]
            y data samples. Similar to `ys` in `forward`, except as Array.
        kwargs:
            Additional arguments for `_train`.

        Returns
        -------
        regress_stats:
            See `_train` for more details.

        """
        kwargs = Config(kwargs)
        kwargs.fill(rcParams.net.BaseDistributionNet.get('regress'))
        dset = TensorDataset(
            torch.tensor(targets, dtype=torch.float),
            *[
                torch.tensor(ys[i], dtype=get_dtype(self.y_spaces[i]))
                for i in range(len(ys))
            ],
        )
        def get_param_and_loss(param_vecs_p, *ys):
            param_vecs_q = self.forward(*ys)
            kl_loss = torch.stack([
                self.p_x.kl_loss(param_vecs_p[i], param_vecs_q[i])
                for i in range(len(param_vecs_p))
            ]).mean()
            return param_vecs_q, kl_loss
        return self._train(dset, get_param_and_loss, **kwargs)


class MultiLayerPerceptronNet(BaseDistributionNet):
    r"""Distribution net with MLP as backbone."""

    def __init__(self,
        x_space, y_spaces, p_x=None,
        mlp_features: Optional[Iterable[int]] = None,
        **kwargs,
    ):
        r"""
        Args
        ----
        mlp_features:
            Hidden layer features of MLP.
        kwargs:
            Additional arguments for `create_mlp_layers`.

        """
        _rcParams = Config(rcParams.get('net.MultiLayerPerceptronNet._init_'))
        super().__init__(x_space, y_spaces, p_x)
        in_features = 0
        for y_space in self.y_spaces:
            if isinstance(y_space, MultiDiscrete):
                in_features += sum(y_space.nvec)
            if isinstance(y_space, Box):
                y_dim, = y_space.shape
                in_features += y_dim
        out_features = self.p_x.param_dim
        if mlp_features is None:
            mlp_features = [int(_rcParams._expanse*(in_features*out_features)**0.5)]
        self.layers = create_mlp_layers(
            in_features, out_features, mlp_features, **kwargs,
        )

    def forward(self, *ys: tuple[Tensor]) -> Tensor:
        out = self._get_input_tensor(*ys)
        for layer in self.layers:
            out = layer(out)
        return out


class CompleteEmbedWrapper(BaseDistributionNet):
    r"""Wrapper for complete embedding of conditioned variable.

    When there is only one conditioned variable and its space is MultiDiscrete,
    use a complete embedding to get a single one-hot vector instead of a
    concatenation of several one-hot vectors.

    """

    def __init__(self,
        x_space, y_spaces, p_x=None,
        net: Union[BaseDistributionNet, dict, None] = None,
    ):
        r"""
        Args
        ----
        net:
            The original distribution net to be wrapped.

        """
        super().__init__(x_space, y_spaces, p_x)

        _y_spaces = [
            MultiDiscrete([np.prod(y_space.nvec)]) if isinstance(y_space, MultiDiscrete)
            else y_space for y_space in self.y_spaces
        ]
        self.net = _create_net(net, self.x_space, _y_spaces, self.p_x)

    def forward(self, *ys):
        return self.net.forward(*[
            sub2ind(ys[i], y_space.nvec)[:, None] if isinstance(y_space, MultiDiscrete)
            else ys[i] for i, y_space in enumerate(self.y_spaces)
        ])


def _create_net(
    net: Union[BaseDistributionNet, dict, None],
    x_space: VarSpace,
    y_spaces: Iterable[VarSpace],
    p_x: Union[BaseDistribution, dict, None] = None,
) -> BaseDistributionNet:
    p_x = _create_distribution(p_x, x_space)
    if net is None or isinstance(net, dict):
        net = Config(net)
        net.fill(rcParams.net._create_net)
        net = net.instantiate(x_space, y_spaces, p_x)
    else:
        assert isinstance(net, BaseDistributionNet)
        assert net.x_space==x_space
        assert tuple(net.y_spaces)==tuple(y_spaces)
        assert net.p_x.param_dim==p_x.param_dim
    return net
